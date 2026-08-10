use std::ops::Deref;
use std::sync::Arc;

use crate::{Error, ErrorCode, ExecutionFacts, Result};

/// Immutable, cheaply cloneable ownership of one continuous token allocation.
/// A borrowed slice cannot outlive its owning buffer:
///
/// ```compile_fail
/// use toktier::TokenBuffer;
/// fn invalid<'a>() -> &'a [u32] {
///     TokenBuffer::new(vec![1, 2, 3]).as_slice()
/// }
/// ```
///
/// Construction adopts the given vector's allocation as the single
/// shared owner (no element copy), and internal session results share
/// the store's own allocation the same way, so a retained buffer keeps
/// observing exactly the memory it was created over.
#[derive(Debug, Clone, Default)]
pub struct TokenBuffer {
    values: Arc<Vec<u32>>,
    start: usize,
    end: usize,
}

impl TokenBuffer {
    pub fn new(ids: impl Into<Vec<u32>>) -> Self {
        let values = Arc::new(ids.into());
        let end = values.len();
        Self {
            values,
            start: 0,
            end,
        }
    }

    /// Share the store's immutable ID range without copying it.
    pub(crate) fn from_shared(ids: toktier_store_core::SharedIds) -> Self {
        let (values, start, end) = ids.into_parts();
        Self { values, start, end }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.values[self.start..self.end]
    }

    pub fn into_vec(self) -> Vec<u32> {
        self.as_slice().to_vec()
    }

    pub(crate) fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start > end || end > self.as_slice().len() {
            return Err(Error::new(
                ErrorCode::Internal,
                "token-buffer slice is outside its parent",
            ));
        }
        Ok(Self {
            values: Arc::clone(&self.values),
            start: self.start + start,
            end: self.start + end,
        })
    }
}

impl From<Vec<u32>> for TokenBuffer {
    fn from(ids: Vec<u32>) -> Self {
        Self::new(ids)
    }
}

impl PartialEq for TokenBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for TokenBuffer {}

impl AsRef<[u32]> for TokenBuffer {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

impl Deref for TokenBuffer {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// One tokenization result and its typed execution facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    ids: TokenBuffer,
    offsets: Option<Arc<[(u32, u32)]>>,
    execution: ExecutionFacts,
}

impl Encoding {
    pub(crate) fn new(
        ids: Vec<u32>,
        offsets: Option<Vec<(u32, u32)>>,
        execution: ExecutionFacts,
    ) -> Self {
        Self {
            ids: ids.into(),
            offsets: offsets.map(Arc::from),
            execution,
        }
    }

    pub(crate) fn from_buffer(ids: TokenBuffer, execution: ExecutionFacts) -> Self {
        Self {
            ids,
            offsets: None,
            execution,
        }
    }

    pub fn ids(&self) -> &[u32] {
        self.ids.as_slice()
    }

    pub fn offsets(&self) -> Option<&[(u32, u32)]> {
        self.offsets.as_deref()
    }

    pub fn execution(&self) -> &ExecutionFacts {
        &self.execution
    }

    pub fn token_buffer(&self) -> &TokenBuffer {
        &self.ids
    }

    pub fn into_token_buffer(self) -> TokenBuffer {
        self.ids
    }
}

/// Ragged batch stored as one values allocation plus monotone row offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaggedEncoding {
    values: TokenBuffer,
    offsets: Arc<[u64]>,
    executions: Arc<[ExecutionFacts]>,
}

impl RaggedEncoding {
    pub(crate) fn from_rows(rows: Vec<(Vec<u32>, ExecutionFacts)>) -> Result<Self> {
        let total = rows.iter().try_fold(0usize, |acc, (ids, _)| {
            acc.checked_add(ids.len())
                .ok_or_else(|| Error::new(ErrorCode::InvalidArgument, "batch token count overflow"))
        })?;
        let mut values = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut executions = Vec::with_capacity(rows.len());
        offsets.push(0);
        for (ids, execution) in rows {
            values.extend_from_slice(&ids);
            offsets.push(u64::try_from(values.len()).map_err(|_| {
                Error::new(ErrorCode::InvalidArgument, "batch token count exceeds u64")
            })?);
            executions.push(execution);
        }
        Ok(Self {
            values: values.into(),
            offsets: offsets.into(),
            executions: executions.into(),
        })
    }

    pub fn values(&self) -> &[u32] {
        self.values.as_slice()
    }

    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validated `(start, end)` value bounds of one row.
    fn bounds(&self, row: usize) -> Result<(usize, usize)> {
        let start = *self.offsets.get(row).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                format!("batch row {row} is out of range"),
            )
        })?;
        let end = *self.offsets.get(row + 1).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidArgument,
                format!("batch row {row} is out of range"),
            )
        })?;
        let start = usize::try_from(start)
            .map_err(|_| Error::new(ErrorCode::Internal, "batch offset exceeds usize"))?;
        let end = usize::try_from(end)
            .map_err(|_| Error::new(ErrorCode::Internal, "batch offset exceeds usize"))?;
        Ok((start, end))
    }

    pub fn row(&self, row: usize) -> Result<&[u32]> {
        let (start, end) = self.bounds(row)?;
        Ok(&self.values[start..end])
    }

    pub fn executions(&self) -> &[ExecutionFacts] {
        &self.executions
    }

    /// One row as a zero-copy view over the batch's shared values allocation.
    pub fn row_buffer(&self, row: usize) -> Result<TokenBuffer> {
        let (start, end) = self.bounds(row)?;
        self.values.slice(start, end)
    }

    pub fn into_rows(self) -> Vec<Vec<u32>> {
        (0..self.len())
            .map(|row| self.row(row).expect("validated ragged offsets").to_vec())
            .collect()
    }
}

/// Exact suffix replacement produced by an incremental append.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenPatch {
    keep_tokens: u64,
    replacement_ids: TokenBuffer,
    revision: u64,
    token_count: u64,
    execution: ExecutionFacts,
}

impl TokenPatch {
    pub(crate) fn new(
        keep_tokens: u64,
        replacement_ids: Vec<u32>,
        revision: u64,
        token_count: u64,
        execution: ExecutionFacts,
    ) -> Self {
        Self {
            keep_tokens,
            replacement_ids: replacement_ids.into(),
            revision,
            token_count,
            execution,
        }
    }

    pub const fn keep_tokens(&self) -> u64 {
        self.keep_tokens
    }

    pub fn replacement_ids(&self) -> &[u32] {
        &self.replacement_ids
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub fn execution(&self) -> &ExecutionFacts {
        &self.execution
    }
}
