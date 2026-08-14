//! Executor-neutral bounded scheduling over the synchronous exactness core.

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::{EncodeOptions, Encoding, Error, ErrorCode, Result, Session, TokenPatch, Tokenizer};

/// What happens only when a request returns an execution error after its own
/// ordered GPU -> CPU -> HF fallback chain is exhausted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceFailurePolicy {
    #[default]
    FallbackOnly,
    RetryEligiblePeer,
}

/// Resource limits for an executor-neutral serving queue.
#[derive(Debug, Clone)]
pub struct ServingLimits {
    pub max_queued_requests: usize,
    pub max_queued_bytes: usize,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
    /// Maximum queued or executing operations for one stateful session.
    pub max_session_requests: usize,
    pub batch_window: Duration,
    pub worker_threads: usize,
}

impl Default for ServingLimits {
    fn default() -> Self {
        Self {
            max_queued_requests: 4096,
            max_queued_bytes: 256 * 1024 * 1024,
            max_batch_rows: 512,
            max_batch_bytes: 4_000_000,
            max_session_requests: 64,
            batch_window: Duration::from_micros(200),
            worker_threads: 1,
        }
    }
}

/// Separately-accounted queue and engine timings for one response.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ServingTimings {
    pub queue: Duration,
    pub engine: Duration,
    /// Time spent constructing a request-owned zero-copy result view.
    pub materialization: Duration,
    /// Time in the stateful store operation. Stateless rows report zero.
    pub store: Duration,
    pub total: Duration,
    pub batch_rows: usize,
    pub device_index: usize,
}

/// A value produced by the bounded scheduling layer.
#[derive(Debug)]
pub struct ServingResponse<T> {
    pub value: T,
    pub timings: ServingTimings,
}

/// Builder requiring at least one already verified tokenizer/device handle.
#[derive(Debug)]
pub struct ServingPoolBuilder {
    tokenizers: Vec<Tokenizer>,
    limits: ServingLimits,
    failure_policy: DeviceFailurePolicy,
}

impl ServingPoolBuilder {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizers: vec![tokenizer],
            limits: ServingLimits::default(),
            failure_policy: DeviceFailurePolicy::default(),
        }
    }

    /// Add an independently validated device route for the same exact artifact.
    pub fn device(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizers.push(tokenizer);
        self
    }

    pub fn limits(mut self, limits: ServingLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn device_failure_policy(mut self, policy: DeviceFailurePolicy) -> Self {
        self.failure_policy = policy;
        self
    }

    pub fn build(self) -> Result<ServingPool> {
        validate_limits(&self.limits)?;
        let first = self.tokenizers.first().ok_or_else(|| {
            Error::new(
                ErrorCode::ConfigInvalid,
                "serving pool requires one tokenizer",
            )
        })?;
        for tokenizer in &self.tokenizers[1..] {
            if tokenizer.family() != first.family()
                || tokenizer.artifact().identity() != first.artifact().identity()
            {
                return Err(Error::new(
                    ErrorCode::ConfigInvalid,
                    "multi-device tokenizers must bind the same canonical artifact",
                ));
            }
        }
        let shared = Arc::new(Shared {
            tokenizers: self.tokenizers.into(),
            limits: self.limits.clone(),
            failure_policy: self.failure_policy,
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                queued_bytes: 0,
                shutdown: false,
            }),
            ready: Condvar::new(),
            next_device: AtomicUsize::new(0),
        });
        let owner = Arc::new(PoolOwner {
            shared: Arc::clone(&shared),
            workers: Mutex::new(Vec::with_capacity(self.limits.worker_threads)),
            stopped: AtomicBool::new(false),
        });
        let mut workers = owner
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for index in 0..self.limits.worker_threads {
            let worker_shared = Arc::clone(&shared);
            workers.push(
                std::thread::Builder::new()
                    .name(format!("toktier-serving-{index}"))
                    .spawn(move || worker_loop(worker_shared))
                    .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?,
            );
        }
        drop(workers);
        Ok(ServingPool { owner })
    }
}

/// Cloneable submission handle. Dropping the final pool drains accepted work.
#[derive(Clone)]
pub struct ServingPool {
    owner: Arc<PoolOwner>,
}

impl std::fmt::Debug for ServingPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServingPool")
            .field("devices", &self.owner.shared.tokenizers.len())
            .field("limits", &self.owner.shared.limits)
            .finish_non_exhaustive()
    }
}

impl ServingPool {
    pub fn builder(tokenizer: Tokenizer) -> ServingPoolBuilder {
        ServingPoolBuilder::new(tokenizer)
    }

    /// Exact configured worker bound. Device count never silently raises it.
    pub fn worker_threads(&self) -> usize {
        self.owner.shared.limits.worker_threads
    }

    pub fn submit(&self, text: impl Into<String>) -> Result<ServingRequest<Encoding>> {
        self.submit_with(text, EncodeOptions::default(), None)
    }

    pub fn submit_with(
        &self,
        text: impl Into<String>,
        options: EncodeOptions,
        deadline_after: Option<Duration>,
    ) -> Result<ServingRequest<Encoding>> {
        let text = text.into();
        let device_index = self
            .owner
            .shared
            .next_device
            .fetch_add(1, Ordering::Relaxed)
            % self.owner.shared.tokenizers.len();
        let completion = Arc::new(Completion::new());
        let now = Instant::now();
        let job = Job::Encode(EncodeJob {
            text,
            options,
            device_index,
            queued_at: now,
            deadline: deadline_after.and_then(|value| now.checked_add(value)),
            completion: Arc::clone(&completion),
        });
        self.owner.shared.enqueue(job)?;
        Ok(ServingRequest { completion })
    }

    /// Open a session pinned to one observable device route for its lifetime.
    pub fn open_session(&self, name: impl Into<String>) -> Result<ServingSession> {
        let device_index = self
            .owner
            .shared
            .next_device
            .fetch_add(1, Ordering::Relaxed)
            % self.owner.shared.tokenizers.len();
        let session = self.owner.shared.tokenizers[device_index].open_session(name)?;
        Ok(ServingSession {
            pool: self.clone(),
            device_index,
            session: Arc::new(Mutex::new(session)),
            pending: Arc::new(AtomicUsize::new(0)),
            sequencer: Arc::new(SessionSequencer::new()),
        })
    }

    /// Drain accepted jobs and join all workers. Further submissions fail.
    pub fn shutdown(self) {
        self.owner.stop();
    }
}

/// Single-writer session adapter. Appends are serialized by its private lock;
/// no store transaction or lock is held across a Future poll/await point.
#[derive(Debug, Clone)]
pub struct ServingSession {
    pool: ServingPool,
    device_index: usize,
    session: Arc<Mutex<Session>>,
    pending: Arc<AtomicUsize>,
    sequencer: Arc<SessionSequencer>,
}

impl ServingSession {
    pub fn submit_seed(&self, text: impl Into<String>) -> Result<ServingRequest<Encoding>> {
        reserve_session_slot(
            &self.pending,
            self.pool.owner.shared.limits.max_session_requests,
        )?;
        let text = text.into();
        let completion = Arc::new(Completion::new());
        let now = Instant::now();
        let ticket = self.sequencer.reserve();
        let job = Job::Seed(SeedJob {
            text,
            ticket,
            device_index: self.device_index,
            queued_at: now,
            deadline: None,
            session: Arc::clone(&self.session),
            pending: Arc::clone(&self.pending),
            sequencer: Arc::clone(&self.sequencer),
            completion: Arc::clone(&completion),
        });
        if let Err(error) = self.pool.owner.shared.enqueue(job) {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            self.sequencer.cancel(ticket);
            return Err(error);
        }
        Ok(ServingRequest { completion })
    }

    pub fn submit_append(&self, suffix: impl Into<String>) -> Result<ServingRequest<TokenPatch>> {
        self.submit_append_with_deadline(suffix, None)
    }

    pub fn submit_append_with_deadline(
        &self,
        suffix: impl Into<String>,
        deadline_after: Option<Duration>,
    ) -> Result<ServingRequest<TokenPatch>> {
        reserve_session_slot(
            &self.pending,
            self.pool.owner.shared.limits.max_session_requests,
        )?;
        let suffix = suffix.into();
        let completion = Arc::new(Completion::new());
        let now = Instant::now();
        let ticket = self.sequencer.reserve();
        let job = Job::Append(AppendJob {
            suffix,
            ticket,
            device_index: self.device_index,
            queued_at: now,
            deadline: deadline_after.and_then(|value| now.checked_add(value)),
            session: Arc::clone(&self.session),
            pending: Arc::clone(&self.pending),
            sequencer: Arc::clone(&self.sequencer),
            completion: Arc::clone(&completion),
        });
        if let Err(error) = self.pool.owner.shared.enqueue(job) {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            self.sequencer.cancel(ticket);
            return Err(error);
        }
        Ok(ServingRequest { completion })
    }
}

/// Executor-neutral request: both `Future` and blocking wait are available.
pub struct ServingRequest<T> {
    completion: Arc<Completion<T>>,
}

impl<T> std::fmt::Debug for ServingRequest<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ServingRequest")
            .field("completed", &self.completion.done.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl<T> ServingRequest<T> {
    /// Request cancellation before execution. If execution has already begun,
    /// the operation may still commit and cancellation makes no rollback claim.
    pub fn cancel(&self) -> bool {
        if self.completion.done.load(Ordering::Acquire) {
            return false;
        }
        !self.completion.cancelled.swap(true, Ordering::AcqRel)
    }

    pub fn wait(self) -> Result<ServingResponse<T>> {
        let mut slot = self
            .completion
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while slot.is_none() {
            slot = self
                .completion
                .ready
                .wait(slot)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        slot.take().expect("completion was signalled")
    }

    pub fn wait_timeout(self, timeout: Duration) -> Result<ServingResponse<T>> {
        let slot = self
            .completion
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut slot, timed) = self
            .completion
            .ready
            .wait_timeout_while(slot, timeout, |value| value.is_none())
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if timed.timed_out() && slot.is_none() {
            self.completion.cancelled.store(true, Ordering::Release);
            return Err(Error::new(
                ErrorCode::DeadlineExceeded,
                "serving request did not complete before the wait timeout; already-started work may still commit",
            ));
        }
        slot.take().expect("completion was signalled")
    }
}

impl<T> Future for ServingRequest<T> {
    type Output = Result<ServingResponse<T>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut result = self
            .completion
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(result) = result.take() {
            return Poll::Ready(result);
        }
        *self
            .completion
            .waker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(context.waker().clone());
        Poll::Pending
    }
}

impl<T> Drop for ServingRequest<T> {
    fn drop(&mut self) {
        if !self.completion.done.load(Ordering::Acquire) {
            self.completion.cancelled.store(true, Ordering::Release);
        }
    }
}

struct Completion<T> {
    result: Mutex<Option<Result<ServingResponse<T>>>>,
    ready: Condvar,
    waker: Mutex<Option<Waker>>,
    cancelled: AtomicBool,
    done: AtomicBool,
}

impl<T> Completion<T> {
    fn new() -> Self {
        Self {
            result: Mutex::new(None),
            ready: Condvar::new(),
            waker: Mutex::new(None),
            cancelled: AtomicBool::new(false),
            done: AtomicBool::new(false),
        }
    }

    fn complete(&self, result: Result<ServingResponse<T>>) {
        let result = if self.cancelled.load(Ordering::Acquire) {
            Err(Error::new(
                ErrorCode::RequestCancelled,
                "request was cancelled; already-started work may still have committed",
            ))
        } else {
            result
        };
        *self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
        self.done.store(true, Ordering::Release);
        self.ready.notify_all();
        if let Some(waker) = self
            .waker
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            waker.wake();
        }
    }
}

struct EncodeJob {
    text: String,
    options: EncodeOptions,
    device_index: usize,
    queued_at: Instant,
    deadline: Option<Instant>,
    completion: Arc<Completion<Encoding>>,
}

struct AppendJob {
    suffix: String,
    ticket: u64,
    device_index: usize,
    queued_at: Instant,
    deadline: Option<Instant>,
    session: Arc<Mutex<Session>>,
    pending: Arc<AtomicUsize>,
    sequencer: Arc<SessionSequencer>,
    completion: Arc<Completion<TokenPatch>>,
}

struct SeedJob {
    text: String,
    ticket: u64,
    device_index: usize,
    queued_at: Instant,
    deadline: Option<Instant>,
    session: Arc<Mutex<Session>>,
    pending: Arc<AtomicUsize>,
    sequencer: Arc<SessionSequencer>,
    completion: Arc<Completion<Encoding>>,
}

enum Job {
    Encode(EncodeJob),
    Seed(SeedJob),
    Append(AppendJob),
}

impl Job {
    fn bytes(&self) -> usize {
        match self {
            Self::Encode(job) => job.text.len(),
            Self::Seed(job) => job.text.len(),
            Self::Append(job) => job.suffix.len(),
        }
    }
}

struct QueueState {
    jobs: VecDeque<Job>,
    queued_bytes: usize,
    shutdown: bool,
}

struct Shared {
    tokenizers: Arc<[Tokenizer]>,
    limits: ServingLimits,
    failure_policy: DeviceFailurePolicy,
    state: Mutex<QueueState>,
    ready: Condvar,
    next_device: AtomicUsize,
}

impl Shared {
    fn enqueue(&self, job: Job) -> Result<()> {
        let bytes = job.bytes();
        if bytes > self.limits.max_batch_bytes || bytes > self.limits.max_queued_bytes {
            return Err(Error::new(
                ErrorCode::QueueFull,
                "request exceeds the configured queue/batch byte bound",
            ));
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown {
            return Err(Error::new(
                ErrorCode::RuntimeShutdown,
                "serving pool is shutting down",
            ));
        }
        if state.jobs.len() >= self.limits.max_queued_requests
            || state.queued_bytes.saturating_add(bytes) > self.limits.max_queued_bytes
        {
            return Err(Error::new(ErrorCode::QueueFull, "serving queue is full"));
        }
        state.queued_bytes += bytes;
        state.jobs.push_back(job);
        self.ready.notify_one();
        Ok(())
    }
}

struct PoolOwner {
    shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    stopped: AtomicBool,
}

impl PoolOwner {
    fn stop(&self) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.shutdown = true;
            self.shared.ready.notify_all();
        }
        for worker in self
            .workers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
        {
            let _ = worker.join();
        }
    }
}

impl Drop for PoolOwner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn worker_loop(shared: Arc<Shared>) {
    loop {
        let first = {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while state.jobs.is_empty() && !state.shutdown {
                state = shared
                    .ready
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if state.jobs.is_empty() && state.shutdown {
                return;
            }
            let job = state.jobs.pop_front().expect("queue is not empty");
            state.queued_bytes = state.queued_bytes.saturating_sub(job.bytes());
            job
        };
        match first {
            Job::Encode(first) => execute_encode_batch(&shared, first),
            Job::Seed(job) => execute_seed(job),
            Job::Append(job) => execute_append(job),
        }
    }
}

fn execute_encode_batch(shared: &Arc<Shared>, first: EncodeJob) {
    if should_skip(&first.completion, first.deadline) {
        return;
    }
    let device = first.device_index;
    let options = first.options;
    let batch_deadline = Instant::now() + shared.limits.batch_window;
    let mut bytes = first.text.len();
    let mut jobs = vec![first];
    while jobs.len() < shared.limits.max_batch_rows && Instant::now() < batch_deadline {
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let position = state.jobs.iter().position(|job| match job {
            Job::Encode(job) => {
                job.device_index == device
                    && job.options == options
                    && bytes.saturating_add(job.text.len()) <= shared.limits.max_batch_bytes
            }
            Job::Seed(_) | Job::Append(_) => false,
        });
        if let Some(position) = position {
            let job = state
                .jobs
                .remove(position)
                .expect("position came from queue");
            state.queued_bytes = state.queued_bytes.saturating_sub(job.bytes());
            drop(state);
            if let Job::Encode(job) = job {
                if !should_skip(&job.completion, job.deadline) {
                    bytes += job.text.len();
                    jobs.push(job);
                }
            }
            continue;
        }
        let remaining = batch_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let (_state, timeout) = shared
            .ready
            .wait_timeout(state, remaining)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if timeout.timed_out() {
            break;
        }
    }

    let texts = jobs.iter().map(|job| job.text.as_str()).collect::<Vec<_>>();
    let started = Instant::now();
    let mut result = shared.tokenizers[device].encode_batch_with(&texts, options);
    let mut used_device = device;
    if result.is_err()
        && shared.failure_policy == DeviceFailurePolicy::RetryEligiblePeer
        && shared.tokenizers.len() > 1
    {
        for offset in 1..shared.tokenizers.len() {
            let candidate = (device + offset) % shared.tokenizers.len();
            result = shared.tokenizers[candidate].encode_batch_with(&texts, options);
            if result.is_ok() {
                used_device = candidate;
                break;
            }
        }
    }
    let engine = started.elapsed();
    match result {
        Ok(batch) => {
            for (row, job) in jobs.into_iter().enumerate() {
                let materialization_started = Instant::now();
                let value = batch
                    .row_buffer(row)
                    .map(|buffer| Encoding::from_buffer(buffer, batch.executions()[row].clone()));
                let materialization = materialization_started.elapsed();
                let finished = Instant::now();
                let response = value.map(|value| ServingResponse {
                    value,
                    timings: ServingTimings {
                        queue: started.saturating_duration_since(job.queued_at),
                        engine,
                        materialization,
                        store: Duration::ZERO,
                        total: finished.saturating_duration_since(job.queued_at),
                        batch_rows: batch.len(),
                        device_index: used_device,
                    },
                });
                job.completion.complete(response);
            }
        }
        Err(error) => {
            for job in jobs {
                job.completion
                    .complete(Err(Error::new(error.code(), error.message().to_owned())));
            }
        }
    }
}

fn execute_append(job: AppendJob) {
    let turn = job.sequencer.wait_for(job.ticket);
    if should_skip(&job.completion, job.deadline) {
        job.pending.fetch_sub(1, Ordering::AcqRel);
        return;
    }
    let started = Instant::now();
    // Cancellation after this point does not claim to cancel a committed
    // store transaction: append executes exactly once under the session lock.
    let result = job
        .session
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .append_observed(&job.suffix);
    let finished = Instant::now();
    job.pending.fetch_sub(1, Ordering::AcqRel);
    drop(turn);
    job.completion.complete(result.map(|(value, store)| {
        ServingResponse {
            value,
            timings: ServingTimings {
                queue: started.saturating_duration_since(job.queued_at),
                engine: finished
                    .saturating_duration_since(started)
                    .saturating_sub(store),
                materialization: Duration::ZERO,
                store,
                total: finished.saturating_duration_since(job.queued_at),
                batch_rows: 1,
                device_index: job.device_index,
            },
        }
    }));
}

fn execute_seed(job: SeedJob) {
    let turn = job.sequencer.wait_for(job.ticket);
    if should_skip(&job.completion, job.deadline) {
        job.pending.fetch_sub(1, Ordering::AcqRel);
        return;
    }
    let started = Instant::now();
    let result = job
        .session
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .seed_observed(&job.text);
    let finished = Instant::now();
    job.pending.fetch_sub(1, Ordering::AcqRel);
    drop(turn);
    job.completion.complete(result.map(|(value, store)| {
        ServingResponse {
            value,
            timings: ServingTimings {
                queue: started.saturating_duration_since(job.queued_at),
                engine: finished
                    .saturating_duration_since(started)
                    .saturating_sub(store),
                materialization: Duration::ZERO,
                store,
                total: finished.saturating_duration_since(job.queued_at),
                batch_rows: 1,
                device_index: job.device_index,
            },
        }
    }));
}

fn should_skip<T>(completion: &Completion<T>, deadline: Option<Instant>) -> bool {
    let cancelled = completion.cancelled.load(Ordering::Acquire);
    let expired = deadline.is_some_and(|deadline| Instant::now() >= deadline);
    if cancelled || expired {
        completion.complete(Err(Error::new(
            if expired {
                ErrorCode::DeadlineExceeded
            } else {
                ErrorCode::RequestCancelled
            },
            if expired {
                "request deadline elapsed before execution"
            } else {
                "request was cancelled before execution"
            },
        )));
        true
    } else {
        false
    }
}

fn validate_limits(limits: &ServingLimits) -> Result<()> {
    if limits.max_queued_requests == 0
        || limits.max_queued_bytes == 0
        || limits.max_batch_rows == 0
        || limits.max_batch_bytes == 0
        || limits.max_session_requests == 0
        || limits.worker_threads == 0
        || limits.max_batch_bytes > limits.max_queued_bytes
    {
        return Err(Error::new(
            ErrorCode::ConfigInvalid,
            "serving limits must be non-zero and batch bytes may not exceed queue bytes",
        ));
    }
    Ok(())
}

fn reserve_session_slot(pending: &AtomicUsize, maximum: usize) -> Result<()> {
    pending
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            (current < maximum).then_some(current + 1)
        })
        .map(|_| ())
        .map_err(|_| {
            Error::new(
                ErrorCode::QueueFull,
                "stateful session reached its configured in-flight request bound",
            )
        })
}

/// FIFO admission for one stateful session. The global pool may dequeue
/// several operations concurrently, but only the next ticket may touch the
/// session. Dropping a turn advances the queue even for a cancelled or
/// expired operation, so later appends cannot deadlock behind it.
#[derive(Debug)]
struct SessionSequencer {
    issued: AtomicU64,
    state: Mutex<SessionSequenceState>,
    ready: Condvar,
}

#[derive(Debug)]
struct SessionSequenceState {
    next: u64,
    cancelled: BTreeSet<u64>,
}

impl SessionSequencer {
    fn new() -> Self {
        Self {
            issued: AtomicU64::new(0),
            state: Mutex::new(SessionSequenceState {
                next: 0,
                cancelled: BTreeSet::new(),
            }),
            ready: Condvar::new(),
        }
    }

    fn reserve(&self) -> u64 {
        self.issued.fetch_add(1, Ordering::Relaxed)
    }

    fn cancel(&self, ticket: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cancelled.insert(ticket);
        advance_cancelled(&mut state);
        drop(state);
        self.ready.notify_all();
    }

    fn wait_for(&self, ticket: u64) -> SessionTurn<'_> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.next != ticket {
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        SessionTurn {
            sequencer: self,
            state: Some(state),
        }
    }
}

struct SessionTurn<'a> {
    sequencer: &'a SessionSequencer,
    state: Option<std::sync::MutexGuard<'a, SessionSequenceState>>,
}

impl Drop for SessionTurn<'_> {
    fn drop(&mut self) {
        if let Some(mut state) = self.state.take() {
            state.next = state.next.saturating_add(1);
            advance_cancelled(&mut state);
            drop(state);
            self.sequencer.ready.notify_all();
        }
    }
}

fn advance_cancelled(state: &mut SessionSequenceState) {
    while state.cancelled.remove(&state.next) {
        state.next = state.next.saturating_add(1);
    }
}
