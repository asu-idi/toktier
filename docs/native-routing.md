# Native routing

TokTier 0.1.1 moves the allocation-sensitive part of per-input routing into
`toktier-routing-core`, without changing the public `RoutePlan`, fallback
order, reason codes, or 64 KiB crossover.

## What is native now

For each plain input, one private PyO3 call now:

1. borrows CPython's cached UTF-8 view instead of allocating a complete
   `bytes` copy;
2. selects the first backend whose byte threshold is satisfied; and
3. runs the added-token frontend's one-/two-byte necessary-condition gate.

A negative literal result is final. A positive result is only a candidate and
still runs the exact Hugging Face added-vocabulary splitter, so the native gate
cannot admit an accelerated input that the prior frontend would have routed to
the reference backend. Scanners without a sound native prefilter are treated as
"always candidate" and keep the previous exact path.

The same crate also owns the frozen O/S/L/N/M BPE synchronizing-transition
predicate used by the session store. Certified corrected-Gigatoken sessions
pass the digest-checked property table into the native encoder adapter once.
The store can then seal a long stable prefix without copying every token ID and
span into Python. It retains both the longest-added-literal end guard and
enough mutable tail for the next certified repair window. Uncertified engines
still never seal.

## Control-plane profile

`tools/profile_routing.py` compares this selector with the allocation shape of
the 0.1.0 no-added-literal route. It measures only the UTF-8 threshold plus the
necessary-condition gate; it is not a tokenizer or end-to-end API benchmark.

Recorded on an Intel Core Ultra 9 285K / Python 3.12.3 host (the RTX 5090 test
machine), seven repeats, median microseconds per call:

| Input | 0.1.0 allocation shape | Native selector | Speedup |
|---:|---:|---:|---:|
| 32 bytes | 0.186 us | 0.051 us | 3.67x |
| 1,500 bytes | 0.593 us | 0.055 us | 10.81x |
| 65,536 bytes | 15.698 us | 0.369 us | 42.52x |
| 4,000,000 bytes | 2,969.548 us | 51.634 us | 57.51x |

Reproduce the profile from an installed source checkout:

```bash
python tools/profile_routing.py --repeats 7
```

The profile deliberately separates control-plane time from backend execution,
content-index hashing, store persistence, and conversion of a full result into
the public Python `tuple[int, ...]`. In particular, returning a million-token
`Encoding.ids` necessarily costs more than the bounded internal repair itself.

## Compatibility and remaining work

The native selector is an internal implementation detail. Python still owns
artifact acquisition, registry projection, immutable plan construction,
fallback invocation/accounting, and public result conversion. Content-prefix
indexing and backend calls can still cross Python, and the corrected-Gigatoken
append adapter still calls into its private Python extension wrapper.

The longer-term one-call, no-callback request pipeline remains tracked in
[`ROADMAP.md`](../ROADMAP.md). Moving those pieces requires independent
differential and hardware evidence; 0.1.1 does not claim that larger migration
is complete.
