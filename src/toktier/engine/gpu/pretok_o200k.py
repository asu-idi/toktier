r"""Piece starts for the o200k pre-tokenization band, computed on the GPU.

The band
--------
"o200k" is the splitter group whose pattern has seven leftmost-first
alternatives. Families in the band differ in exactly two parameters: the
maximum number of digits in a piece and whether the contraction suffix
exists at all.

    A1  [^\r\n\p{L}\p{N}]? [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ (?i:'s|'t|...)?
    A2  [^\r\n\p{L}\p{N}]? [Lu Lt Lm Lo M]+ [Ll Lm Lo M]* (?i:'s|'t|...)?
    A3  \p{N}{1,digits_max}
    A4  ' ?[^\s\p{L}\p{N}]+[\r\n/]*     (the slash is absorbed with CR/LF)
    A5  \s*[\r\n]+      A6  \s+(?!\S)      A7  \s+

What this module computes
-------------------------
:class:`GpuPretokO200k` turns a codepoint tensor into a boolean mask of
piece starts: element i is true exactly when a piece of the
leftmost-first match sequence begins at i. There is a single-document
entry (``starts``) and a batched entry (``starts_batched``) whose
semantics are "call ``starts`` on each document", implemented by
truncating every cross-character and cross-run propagation at document
heads.

Three structures separate this band from the plain GPT-style band, and
they are what most of the code below is about:

1. **Case backtracking is local.** Inside a letter run (subclasses
   U = Lu/Lt, L = Ll, C = Lm/Lo/M), a piece starts at a U-only character
   whose previous character is either L-only or a C playing a lower
   role. lower_role(p) holds when, inside the same contiguous {L,C}
   segment, an L-only character occurs at or before p, or when p is the
   last C of the whole letter run and no L-only follows it: that is
   where the greedy U* backtrack lands.
2. **Contraction suffix chains.** A letter piece may end with
   (?i:'s|'t|...), eating an apostrophe plus one or two letters of the
   following run. A run that is swallowed whole makes the next
   apostrophe lose its suffix eligibility, because that piece has
   already used its suffix slot; this produces alternating chains of the
   "'t't't" shape, which are resolved in order (a sparse, exact loop off
   the vectorised path). An affected run recomputes its internal rules
   from its effective start, with hasL / lastL / lastC all measured over
   the effective interval.
3. **{CR, LF, /} absorption crosses runs.** The A4 tail [\r\n/]* absorbs,
   from the end of a punct run, the longest contiguous {crlf, slash}
   segment. That segment can span a whitespace run and a slash punct run,
   wholly or partly. Absorbed characters never start a piece, and the
   whitespace / punct runs take part in the remaining rules from their
   effective start.

Semantics are pinned to the reference tokenizer's own splitter: every
rule here was derived from, and differentially checked against, the
reference engine's regex splitter applied to the same artifact. Where a
shape cannot be decided by the vectorised rules, this module recomputes
it with the sequential matcher below rather than approximating it, which
is the same discipline as the serial fallback of the parallel
tokenization algorithm: a narrow trigger surface, correctness
unconditional.

Class table
-----------
The constructor takes an already loaded and verified class table; it
never probes or builds one. The seven values are:

    P  everything else (the A4 punct class)     U  Lu, Lt
    L  Ll                                       C  Lm, Lo
    N  \p{N}                                    S  the splitter's own
    M  \p{M}                                       whitespace set

Marks get a class of their own because they carry a threefold identity:
they belong to the A1/A2 letter classes, to the A4 punct class and to the
prefix class. The vectorised rules give them the letter role by default;
next to a punct character the sparse local fallback resolves them
exactly.

The masks behind that table must come from the reference tokenizer engine
itself, never from another Unicode table. Two incidents made this a rule.
A table built from the standard library's Unicode 15.0 data lacked
codepoints the reference engine (Unicode 16.0) knows, and the missing
ones produced extra splits. A table rebuilt from a regex package that was
ahead at Unicode 17.0 skewed the other way, calling characters letters
that the reference engine treats as unassigned. Table generation
therefore probes the reference engine, and the loader checks the digest
that the kernel certificate binds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .options import DEFAULT_GPU_OPTIONS, GpuOptions

__all__ = ["GpuPretokO200k", "doc_start_arrays"]

#: Class values in the o200k class table.
P_, U_, L_, C_, N_, S_, M_ = 0, 1, 2, 3, 4, 5, 6

#: Contraction suffixes, as codepoints for the tensor path: one-letter
#: forms ('s 't 'm 'd) and two-letter forms ('re 've 'll).
CONTR1 = {0x73, 0x74, 0x6D, 0x64}
CONTR2 = {(0x72, 0x65), (0x76, 0x65), (0x6C, 0x6C)}

#: The same suffixes as characters, for the host matcher.
CONTR_ONE = {"s", "t", "m", "d"}
CONTR_TWO = {("r", "e"), ("v", "e"), ("l", "l")}

#: LATIN SMALL LETTER LONG S, the one non-ASCII folding source.
LONG_S = 0x17F


def _fold_ch(ch: str) -> str:
    """Simple case folding, as the (?i:) contraction group needs it.

    For the eight target lowercase letters {s,t,r,e,v,m,l,d} the only
    folding sources are A-Z and the long s (U+017F) -> "s". Probes
    against the reference splitter settled this: an apostrophe followed
    by the long s matches the suffix, while the other case-like variants
    do not. ``str.lower()`` does not fold the long s, and using it was a
    latent divergence in an earlier host implementation.
    """
    if "A" <= ch <= "Z":
        return ch.lower()
    if ord(ch) == LONG_S:
        return "s"
    return ch


def _fold_t(x: torch.Tensor) -> torch.Tensor:
    """Tensor form of :func:`_fold_ch` over a codepoint tensor."""
    y = torch.where((x >= 65) & (x <= 90), x + 32, x)
    return torch.where(x == 0x17F, torch.full_like(y, 0x73), y)


# ---- scan-free propagation primitives ------------------------------
# torch's cummax / cummin are slow scan kernels: measured at 64M
# elements, one call costs about 105 ms, against 0.66 ms for a cumsum of
# the same length. The 18 scans this file used to contain accounted for
# about 87% of the GPU time of starts(). The two primitives below rewrite
# them with cumsum plus gather; the element-for-element equality argument
# is recorded at every call site.


def _fill_last(
    mark: torch.Tensor, vals: torch.Tensor, default: int, ar: torch.Tensor
) -> torch.Tensor:
    """Position i takes ``vals`` at the nearest mark <= i, else ``default``.

    Equal element for element to ``cummax(where(mark, vals, default))`` if
    and only if ``vals`` is non-decreasing along the mark order and is at
    least ``default``. Every call site in this file satisfies that (an
    arange is strictly increasing, a prefix count is non-decreasing); the
    argument is spelled out at each call site.
    """
    idx = mark.long().cumsum(0) - 1        # index of the nearest mark, -1 = none
    src = vals[mark]
    if src.numel() == 0:
        return torch.full_like(ar, default)
    out = src[idx.clamp(min=0)]
    return torch.where(idx >= 0, out, torch.full_like(ar, default))


def _first_brk(
    brk: torch.Tensor, ar: torch.Tensor, n: int, strictly_after: bool
) -> torch.Tensor:
    """Position i takes the first break at (or after) i; ``n`` when none.

    Equal element for element to the flip / cummin / flip family it
    replaces: break positions are strictly increasing, so the reverse
    minimum is the nearest forward break.
    """
    cnt = brk.long().cumsum(0)
    idx = cnt if strictly_after else cnt - brk.long()
    pos = ar[brk]
    count = pos.numel()
    if count == 0:
        return torch.full_like(ar, n)
    out = pos[idx.clamp(max=count - 1)]
    return torch.where(idx < count, out, torch.full_like(ar, n))


def _runs(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contiguous true segments: ``(head, rid, rs_of, re_of)``.

    ``rid`` is meaningless outside the mask, and rs/re are only valid
    inside it. Scan-free: rs_of gathers head positions by run id, which
    equals ``cummax(where(head, ar, 0))`` because an arange is strictly
    increasing (so the running maximum is the last head) and the default
    0 for rid < 0 matches the original; re_of is the first non-mask
    position at or after i.
    """
    n = mask.numel()
    head = mask.clone()
    head[1:] &= ~mask[:-1]
    rid = head.long().cumsum(0) - 1
    ar = torch.arange(n, device=mask.device)
    heads_pos = ar[head]
    if heads_pos.numel():
        rs_of = torch.where(
            rid >= 0, heads_pos[rid.clamp(min=0)], torch.zeros_like(ar)
        )
    else:
        rs_of = torch.zeros_like(ar)
    re_of = _first_brk(~mask, ar, n, strictly_after=False)
    return head, rid, rs_of, re_of


def _runs_b(
    mask: torch.Tensor, doc_head: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched :func:`_runs`: a document's first character forces a new run.

    Every rule that is expressed per run therefore stops at document
    boundaries by construction. rs/re are only valid inside the mask, as
    in :func:`_runs`.
    """
    n = mask.numel()
    head = mask.clone()
    head[1:] &= ~mask[:-1]
    head |= mask & doc_head
    rid = head.long().cumsum(0) - 1
    ar = torch.arange(n, device=mask.device)
    heads_pos = ar[head]
    if heads_pos.numel():
        rs_of = torch.where(
            rid >= 0, heads_pos[rid.clamp(min=0)], torch.zeros_like(ar)
        )
    else:
        rs_of = torch.zeros_like(ar)
    # The end of a run is the first break strictly after i: a non-mask
    # position or a document head. Scan-free: break positions are
    # strictly increasing, so the counted gather equals the original
    # flip + cummin + shift (there, rc[i+1] was the first break at or
    # after i+1, that is the first break after i).
    brk = (~mask) | doc_head
    re_of = _first_brk(brk, ar, n, strictly_after=True)
    return head, rid, rs_of, re_of


class _HostMatcherO200k:
    """Sequential leftmost-first matcher over the same class table.

    This is the exact semantics that the sparse fallbacks recompute with:
    the alternatives are tried in pattern order and the first one that
    matches wins. Its predicates read the class table the tensor stages
    use, so the two cannot disagree about what a character is.
    """

    def __init__(
        self,
        table: np.ndarray[Any, Any],
        dmax: int,
        contractions: bool,
    ) -> None:
        # A bytes view of the table: indexing it yields a Python int
        # directly, which is what the per-character loops below want.
        self._cls: bytes = table.astype(np.uint8, copy=False).tobytes()
        self._dmax = dmax
        self._contractions = contractions

    # -- character predicates, all derived from the class table --------

    def _is_ws(self, ch: str) -> bool:
        return self._cls[ord(ch)] == S_

    def _is_num(self, ch: str) -> bool:
        return self._cls[ord(ch)] == N_

    def _is_letter(self, ch: str) -> bool:
        """\\p{L}: the letter classes without the marks."""
        return self._cls[ord(ch)] in (U_, L_, C_)

    def _is_upperish(self, ch: str) -> bool:
        """[Lu Lt Lm Lo M], the first letter class of A1/A2."""
        return self._cls[ord(ch)] in (U_, C_, M_)

    def _is_lowerish(self, ch: str) -> bool:
        """[Ll Lm Lo M], the second letter class of A1/A2."""
        return self._cls[ord(ch)] in (L_, C_, M_)

    def _is_punct4(self, ch: str) -> bool:
        """[^\\s\\p{L}\\p{N}], the A4 punct class; marks belong to it."""
        return self._cls[ord(ch)] in (P_, M_)

    def _prefix_ok(self, ch: str) -> bool:
        """[^\\r\\n\\p{L}\\p{N}], the optional prefix character of A1/A2."""
        return (
            ch not in "\r\n"
            and not self._is_letter(ch)
            and not self._is_num(ch)
        )

    # -- the alternatives ----------------------------------------------

    def _suffix(self, s: str, e: int, n: int) -> int:
        """Greedy (?i:'s|'t|'re|'ve|'m|'ll|'d)? tail."""
        if not self._contractions:
            return e
        if e < n and s[e] == "'" and e + 1 < n:
            f1 = _fold_ch(s[e + 1])
            if f1 in CONTR_ONE:
                return e + 2
            if e + 2 < n and (f1, _fold_ch(s[e + 2])) in CONTR_TWO:
                return e + 3
        return e

    def _letters(self, s: str, i: int, n: int, need_lower: bool) -> int:
        """Letter body of A1 (``need_lower``) or A2: U*L+ or U+L*.

        Includes the optional prefix character and the optional suffix.
        Returns the end of the match, or -1 when it does not match.
        """
        for pre in (1, 0):
            if pre and not (self._prefix_ok(s[i]) and i + 1 <= n - 1):
                continue
            j0 = i + pre
            if j0 >= n:
                continue
            u = j0
            while u < n and self._is_upperish(s[u]):
                u += 1
            if need_lower:
                # Greedy U* backtrack: k is the largest end of U* whose
                # character is lowerish, that is the first letter of L+.
                k = -1
                t = min(u, n - 1)
                while t >= j0:
                    if self._is_lowerish(s[t]):
                        k = t
                        break
                    t -= 1
                if k < 0:
                    continue
                e = k
                while e < n and self._is_lowerish(s[e]):
                    e += 1
            else:
                if u == j0:                     # U+ needs at least one
                    continue
                e = u
                while e < n and self._is_lowerish(s[e]):
                    e += 1
            return self._suffix(s, e, n)
        return -1

    def match(self, s: str, i: int, n: int) -> int:
        """End of the leftmost-first match that starts at ``i``."""
        e = self._letters(s, i, n, need_lower=True)      # A1
        if e >= 0:
            return e
        e = self._letters(s, i, n, need_lower=False)     # A2
        if e >= 0:
            return e
        c = s[i]
        if self._is_num(c):                              # A3
            j = i + 1
            while j < n and j - i < self._dmax and self._is_num(s[j]):
                j += 1
            return j
        p = i                                            # A4
        if c == " " and i + 1 < n and self._is_punct4(s[i + 1]):
            p = i + 1
        if p < n and self._is_punct4(s[p]):
            j = p + 1
            while j < n and self._is_punct4(s[j]):
                j += 1
            while j < n and s[j] in "\r\n/":
                j += 1
            return j
        if self._is_ws(c):                               # A5 / A6 / A7
            j = i + 1
            while j < n and self._is_ws(s[j]):
                j += 1
            run_end = j
            t = run_end - 1
            while t >= i:
                if s[t] in "\r\n":
                    return t + 1            # A5: cut after the last CR/LF
                t -= 1
            if run_end == n:
                return run_end              # A6 at end of input
            if run_end - i >= 2:
                return run_end - 1          # A6 gives the last blank back
            return run_end                  # A7
        return i + 1                        # defensive; not reachable


class GpuPretokO200k:
    """Piece starts for the o200k band.

    The two variant parameters cover the whole band. One family group
    uses the same skeleton without the contraction suffix chain and with
    one piece per digit; a character-by-character comparison against the
    artifacts' own splitters shows exactly those two differences. The
    defaults reproduce the plain o200k behaviour, so families that had it
    are unchanged bit for bit. With ``contractions=False`` the candidate
    mask is always empty, so the whole suffix-chain machinery downstream
    becomes inert and the effective run start falls back to the plain run
    start.

    Args:
        ext: The compiled kernel extension module. This class never loads
            it: one process owns one build, and the loader hands the
            module in.
        class_table: The verified class table for this family.
        digits_max: Maximum number of digits in an A3 piece.
        contractions: Whether the splitter has the contraction
            alternative.
        device: CUDA device for this instance. It is the authoritative
            device here; callers that keep a :class:`GpuOptions` should
            pass ``options.device``.
        options: Tuning options. None of them changes which piece starts
            come out.
    """

    def __init__(
        self,
        ext: Any,
        class_table: np.ndarray[Any, Any],
        *,
        digits_max: int,
        contractions: bool,
        device: str = "cuda:0",
        options: GpuOptions | None = None,
    ) -> None:
        opts = DEFAULT_GPU_OPTIONS if options is None else options
        self.ext = ext
        self.options = opts
        self.dev = torch.device(device)
        self.dmax = digits_max
        self.contractions = contractions
        # Kernel path for the piece-start stage. The pure tensor path is
        # kept as an explicit differential reference and as the fallback
        # for the shapes the kernel hands back; both produce the same
        # starts.
        self.use_cuda = opts.o200k_cuda_starts
        # Sparse windows are resolved on the device by default; the host
        # window path stays available for diagnosis and comparison.
        self._host_win = opts.o200k_host_windows
        # Mean document length (in characters) above which a batch takes
        # the batched kernel path.
        self._batch_cuda_min = opts.o200k_batch_cuda_min
        self.table_np = class_table    # host copy: batched fallback needs it
        self.table = torch.from_numpy(class_table).to(self.dev)
        self._host = _HostMatcherO200k(class_table, digits_max, contractions)

    @classmethod
    def from_family(
        cls,
        ext: Any,
        table: Any,
        *,
        family: Any,
        digits_max: int,
        options: GpuOptions,
    ) -> GpuPretokO200k:
        """Uniform construction hook for the engine's data-driven dispatch.

        ``table`` is a loaded class table and ``family`` a routing-data
        entry; every pretokenizer entry point exposes this signature.
        """
        return cls(
            ext,
            table.array,
            digits_max=digits_max,
            contractions=family.contractions,
            device=options.device,
            options=options,
        )

    def encode_str(self, s: str) -> torch.Tensor:
        """UTF-32 codepoints of ``s`` as an int32 tensor on the device."""
        cp: np.ndarray[Any, Any] = np.frombuffer(
            s.encode("utf-32-le"), dtype=np.uint32)
        return torch.from_numpy(cp.astype(np.int32)).to(self.dev)

    def starts(self, cp: torch.Tensor) -> torch.Tensor:
        """Piece-start mask for one document."""
        if self.use_cuda:
            return self._starts_cuda(cp)
        return self._starts_torch(cp)

    def _starts_cuda(self, cp: torch.Tensor) -> torch.Tensor:
        """Kernel path for one document.

        The sparse window fallback runs on the device in two phases
        (extents, then the host picks the applied set, then apply), so
        window contents never travel to the host and no character is
        re-parsed in Python. The extreme shape where a chain has no safe
        restart point still yields the whole string to the tensor
        reference. The host window functions stay as a reference and as
        the shared implementation of the batched path;
        ``o200k_host_windows`` switches back to them.
        """
        ext = self.ext
        n = cp.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.dev)
        st, pm_trig, chain_trig, chain = ext.pretok_starts_o200k(
            cp, self.table, self.dmax, self.contractions)
        if self._host_win:                     # host windows: diagnosis
            if int(chain.item()):
                qpos = torch.nonzero(chain_trig[:n]).flatten().cpu().tolist()
                st2 = self._chain_windows(cp, st, qpos, n)
                if st2 is None:
                    return self._starts_torch(cp)
                st = st2
            spans_t = torch.nonzero(pm_trig[:n]).flatten()
            if spans_t.numel():
                st = self._pm_windows(cp, st, spans_t.cpu().tolist(), n)
            return st
        if int(chain.item()):
            qpos = torch.nonzero(chain_trig[:n]).flatten().cpu().tolist()
            # Candidates within 64 characters of each other are treated
            # as one chain segment, exactly as the host version does.
            g_sp: list[int] = []
            g_qL: list[int] = []
            for q in qpos:
                if g_qL and q - g_qL[-1] <= 64:
                    g_qL[-1] = q
                else:
                    g_sp.append(q)
                    g_qL.append(q)
            if not self._win_gpu(cp, st, g_sp, g_qL, chain_mode=True):
                # Extreme shape: yield the whole string to the reference.
                return self._starts_torch(cp)
        spans_t = torch.nonzero(pm_trig[:n]).flatten()
        if spans_t.numel():
            spans = spans_t.cpu().tolist()
            self._win_gpu(cp, st, spans, spans, chain_mode=False)
        return st

    def _win_gpu(
        self,
        cp: torch.Tensor,
        st: torch.Tensor,
        sp_list: list[int],
        qL_list: list[int],
        chain_mode: bool,
        ds_list: list[int] | None = None,
        de_list: list[int] | None = None,
        doc_list: list[int] | None = None,
    ) -> bool:
        """Two-phase window resolution on the device.

        Returns False when a chain has no safe restart point, in which
        case the caller yields the whole input to the tensor path.

        The applied set is chosen with the same rule as the host version:
        a window whose trigger is already covered by what has been
        resolved is skipped. The argument that the applied intervals do
        not overlap is in the kernel comments. Batched: ds/de are the
        ``[start, end)`` of the document each window belongs to, and
        ``resolved_until`` resets whenever the document changes, because
        the batched fallback never reuses a resolved region across a
        document boundary.
        """
        ext = self.ext
        n = cp.numel()
        nw = len(sp_list)
        if ds_list is None or de_list is None:
            ds_list, de_list = [0] * nw, [n] * nw
        i32: dict[str, Any] = {"dtype": torch.int32, "device": self.dev}
        sp_t = torch.tensor(sp_list, **i32)
        qL_t = torch.tensor(qL_list, **i32)
        ds_t = torch.tensor(ds_list, **i32)
        de_t = torch.tensor(de_list, **i32)
        lo, hi, _nosafe = ext.o200k_win_extents(
            cp, self.table, sp_t, qL_t, ds_t, de_t, self.dmax,
            self.contractions, 1 if chain_mode else 0)
        # One synchronisation point; "no safe point" arrives as a -1.
        ext_h = torch.stack([lo, hi]).cpu()
        lo_h, hi_h = ext_h[0].tolist(), ext_h[1].tolist()
        if chain_mode and (-1 in lo_h):
            return False
        sel: list[int] = []
        ru = -1
        cur_doc: int | None = None
        for k in range(nw):
            if doc_list is not None and doc_list[k] != cur_doc:
                cur_doc = doc_list[k]
                ru = -1
            # The skip test uses qL (the tail of the chain group) and not
            # sp: when the tail of a group reaches past the resolved
            # region the window has to be applied. A single mismatch in
            # the sibling band taught this. For a point trigger sp == qL,
            # so nothing changes there, and two windows agree on their
            # overlap, so applying one more than strictly needed is safe.
            if qL_list[k] <= ru:
                continue
            sel.append(k)
            ru = hi_h[k]
        if sel:
            ext.o200k_win_apply(
                cp, self.table, st,
                torch.tensor([lo_h[k] for k in sel], **i32),
                torch.tensor([hi_h[k] for k in sel], **i32),
                torch.tensor([qL_list[k] for k in sel], **i32),
                torch.tensor([de_list[k] for k in sel], **i32),
                self.dmax, self.contractions, 1 if chain_mode else 0)
        return True

    def _chain_windows(
        self, cp: torch.Tensor, starts: torch.Tensor, qpos: list[int], n: int
    ) -> torch.Tensor | None:
        r"""Host correction for true contraction chains.

        The kernel treats every contraction candidate as isolated, so the
        subset that a chain drags in is re-resolved here, in order, from a
        safe restart point.

        Safe restart points: the head of an N run is unconditionally safe
        (no rule can reach into a digit run); the head of an S run is safe
        unless its first character is CR/LF and the character before it is
        punct, because the A4 tail only absorbs [\r\n/] and the remaining
        whitespace characters cannot be absorbed. Both kinds of point are
        necessarily piece starts as well: for the S rules the piece start
        is the effective run start, which for a clean head is the run
        start; for the N rule (i - rs) % dmax == 0 holds at rs. Matching
        then proceeds in order until a clean handover point past the end
        of the chain, which must be an S or N **run head** -- a plain
        class change is not enough, as a handover that once landed in the
        middle of a whitespace run showed. If no safe point exists (4096
        characters without an S or N run head, an extreme shape) this
        returns None and the caller yields the whole string.
        """
        seq = self._host
        tab = self.table_np
        # Candidates within 64 characters count as one chain segment: the
        # chain step is at most 3, so this merges generously and saves
        # windows.
        groups: list[list[int]] = []
        for q in sorted(qpos):
            if groups and q - groups[-1][-1] <= 64:
                groups[-1].append(q)
            else:
                groups.append([q])
        f_lo: list[int] = []
        f_hi: list[int] = []
        t_pos: list[int] = []
        resolved_until = -1
        for grp in groups:
            q0, qL = grp[0], grp[-1]
            if q0 <= resolved_until:
                continue
            wlo = max(q0 - 4096, 0)
            text_hi = min(qL + 4096, n)
            win = cp[wlo:text_hi].cpu().numpy()
            # Search backwards for a safe restart point.
            sp = None
            j = q0 - 2
            while j >= wlo:
                c = int(tab[int(win[j - wlo])])
                if c in (S_, N_):
                    pj = int(tab[int(win[j - 1 - wlo])]) if j > 0 else -1
                    if pj != c:                # run head
                        if c == N_:
                            sp = j
                            break
                        v = int(win[j - wlo])
                        if not (v in (10, 13) and pj == P_):
                            sp = j
                            break
                if j == 0:
                    sp = 0
                    break
                j -= 1
            if sp is None:
                return None
            lo = sp
            off_w = lo - wlo
            seg = "".join(map(chr, win[off_w:]))
            hi = lo
            i = 0
            local_starts: list[int] = []
            while True:
                if lo + i >= n:
                    hi = n
                    break
                local_starts.append(lo + i)
                while True:
                    jj = seq.match(seg, i, len(seg))
                    if jj >= len(seg) and text_hi < n:
                        text_hi = min(n, text_hi + 65536)
                        win = cp[wlo:text_hi].cpu().numpy()
                        seg = "".join(map(chr, win[off_w:]))
                        continue
                    break
                i = jj
                end_abs = lo + i
                if end_abs >= n:
                    hi = end_abs
                    break
                if end_abs > qL + 3:     # past the chain: find a handover
                    ce = int(tab[int(win[end_abs - wlo])])
                    cprev = int(tab[int(win[end_abs - 1 - wlo])])
                    if ce in (S_, N_) and cprev != ce:
                        hi = end_abs
                        break
            f_lo.append(lo)
            f_hi.append(hi)
            t_pos.extend(local_starts)
            resolved_until = hi
        if f_lo:
            fr = np.concatenate(
                [np.arange(a, b) for a, b in zip(f_lo, f_hi, strict=True)]
            )
            starts[torch.from_numpy(fr).to(self.dev)] = False
            starts[torch.tensor(t_pos, dtype=torch.long, device=self.dev)] = True
        return starts

    def _starts_torch(self, cp: torch.Tensor) -> torch.Tensor:
        """Pure tensor path for one document."""
        dev = self.dev
        n = cp.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=dev)
        ar = torch.arange(n, device=dev)
        cpl = cp.long()
        cls_ = self.table[cpl].long()
        crlf = (cp == 10) | (cp == 13)
        slash = cp == 47
        asp = cp == 32
        apo = cp == 39
        mark = cls_ == M_
        c_like = (cls_ == C_) | mark
        letter = (cls_ == U_) | (cls_ == L_) | c_like
        lowish = (cls_ == L_) | c_like
        starts = torch.zeros(n, dtype=torch.bool, device=dev)
        big = torch.full((n,), n, dtype=torch.long, device=dev)
        neg = torch.full((n,), -1, dtype=torch.long, device=dev)

        # ---------- absorption (the A4 tail [\r\n/]*) ----------
        # The anchor is a CR/LF directly after a punct character; a
        # leading slash belongs to the punct+ body, so it cannot serve as
        # the run-head criterion. From the anchor on, the whole
        # contiguous {crlf, slash} segment is absorbed.
        cs = crlf | slash
        _, _, cs_rs, _ = _runs(cs)
        prev_is_P = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_is_P[1:] = cls_[:-1] == P_
        anchor = cs & crlf & prev_is_P
        # Scan-free: vals = ar is strictly increasing, so cummax is the
        # last value, and the default -1 matches the original.
        last_anchor = _fill_last(anchor, ar, -1, ar)
        absorbed = cs & (last_anchor >= cs_rs)

        # ---------- S runs (A5/A6/A7; effective start skips absorbed) ----
        s_mask = cls_ == S_
        s_head_all, _, _, s_re = _runs(s_mask)
        first_live = torch.where(s_mask & ~absorbed, ar, big)
        # Minimum within a run: the effective start is the run's first
        # position that was not absorbed. scatter_reduce needs a dense run
        # id, hence the cumsum over the head mask.
        srid = s_head_all.long().cumsum(0) - 1
        Rs = int(srid[-1].item()) + 1 if s_mask.any() else 0
        if Rs > 0:
            fl = torch.full((Rs,), n, dtype=torch.long, device=dev)
            fl.scatter_reduce_(0, srid[s_mask], first_live[s_mask],
                               reduce="amin")
            lc_ = torch.full((Rs,), -1, dtype=torch.long, device=dev)
            lc_.scatter_reduce_(0, srid[s_mask],
                                torch.where(crlf[s_mask] & ~absorbed[s_mask],
                                            ar[s_mask], neg[s_mask]),
                                reduce="amax")
            rs_eff = fl
            re_run = torch.full((Rs,), 0, dtype=torch.long, device=dev)
            re_run.scatter_reduce_(0, srid[s_mask], s_re[s_mask],
                                   reduce="amax")
            nxt_exists = re_run < n
            has_nl = lc_ >= rs_eff
            t0 = torch.where(has_nl, lc_ + 1, rs_eff)
            m = re_run - t0
            idx = rs_eff[has_nl & (rs_eff < n)]
            starts[idx] = True                            # A5 piece
            tail = (m > 0) & (t0 < n)
            starts[t0[tail]] = True                       # start of the tail
            two = (m >= 2) & nxt_exists
            starts[(re_run - 1)[two]] = True              # last blank alone
        # ---------- P runs (effective start; arrival; merge-forward) ----
        p_mask = cls_ == P_
        p_head, _, _, p_re = _runs(p_mask)
        prid = p_head.long().cumsum(0) - 1
        Rp = int(prid[-1].item()) + 1 if p_mask.any() else 0
        merge_fwd_char = torch.zeros(n, dtype=torch.bool, device=dev)
        if Rp > 0:
            p_first_live = torch.full((Rp,), n, dtype=torch.long, device=dev)
            p_first_live.scatter_reduce_(
                0, prid[p_mask],
                torch.where(~absorbed[p_mask], ar[p_mask], big[p_mask]),
                reduce="amin")
            p_re_run = torch.zeros(Rp, dtype=torch.long, device=dev)
            p_re_run.scatter_reduce_(0, prid[p_mask], p_re[p_mask],
                                     reduce="amax")
            p_rs_eff = p_first_live         # may equal re when fully absorbed
            live = p_rs_eff < p_re_run
            pe = p_rs_eff[live]
            # arrival: the character before the effective start is not an
            # ASCII space.
            prev_ok = torch.ones_like(pe, dtype=torch.bool)
            has_prev = pe > 0
            prev_ok[has_prev] = ~asp[pe[has_prev] - 1]
            # merge-forward: effective length 1, next character letterish,
            # arrival holds.
            plen1 = (p_re_run[live] - pe) == 1
            nxt_i = p_re_run[live].clamp(max=n - 1)
            nxt_letter = (p_re_run[live] < n) & letter[nxt_i]
            mf = plen1 & nxt_letter & prev_ok
            # An A4 piece, and a prefix merged into a letter piece alike,
            # begin at the punct character.
            starts[pe[prev_ok]] = True
            merge_fwd_char[pe[mf]] = True   # this punct joins the letter piece

        # ---------- N runs ----------
        n_mask = cls_ == N_
        _, _, n_rs, _ = _runs(n_mask)
        starts |= n_mask & ((ar - n_rs) % self.dmax == 0)

        # ---------- letter runs ----------
        l_head, _, l_rs, l_re = _runs(letter)
        lrid = l_head.long().cumsum(0) - 1
        Rl = int(lrid[-1].item()) + 1 if letter.any() else 0
        if Rl > 0:
            # ---- contraction suffix chains (sparse, resolved on host) --
            # Candidate: an apostrophe q with a letterish character before
            # it and a folded suffix match after it. In the variant
            # without contractions the whole candidate computation is
            # skipped, rather than only zeroing the mask, which used to
            # waste about ten full-length operations.
            if self.contractions:
                q_mask = apo & torch.cat([torch.zeros(
                    1, dtype=torch.bool, device=dev), letter[:-1]])
                f1 = _fold_t(cpl[(ar + 1).clamp(max=n - 1)])
                f2 = _fold_t(cpl[(ar + 2).clamp(max=n - 1)])
                l1ok = torch.zeros(n, dtype=torch.bool, device=dev)
                for v in CONTR1:
                    l1ok |= f1 == v
                lt1 = torch.zeros(n, dtype=torch.bool, device=dev)
                if n > 1:
                    lt1[:-1] = letter[1:]
                l1ok &= lt1
                l2ok = torch.zeros(n, dtype=torch.bool, device=dev)
                for a, b in CONTR2:
                    l2ok |= (f1 == a) & (f2 == b)
                lt2 = torch.zeros(n, dtype=torch.bool, device=dev)
                if n > 2:
                    lt2[:-2] = letter[2:]
                l2ok &= lt2
                cand = q_mask & (l1ok | l2ok)
            else:
                cand = torch.zeros(n, dtype=torch.bool, device=dev)
                l1ok = cand                 # placeholder: only used via cand
            consumed_of_run = torch.zeros(Rl, dtype=torch.long, device=dev)
            if cand.any():
                # ---- vectorised fast path for isolated candidates ----
                # Blocking propagates only along "swallowed whole" links:
                # a fired q' that eats exactly up to the end of its run
                # (q' + 1 + k' == run_end) where that position is itself a
                # candidate. Split the candidates into the ones a chain
                # drags in (they have an outgoing link, or are pointed at)
                # and the isolated ones: nobody can block an isolated
                # candidate, so it always fires -- which is what the
                # original loop decided for them unconditionally -- and it
                # is recorded with vector operations. Only the
                # chain-involved subset takes the original sequential
                # loop; that subset is near zero in real text, and fuzzing
                # covers the alternating "'t't't" shape. Decision for
                # decision this matches the original full loop.
                qs_t = torch.nonzero(cand).flatten()
                k_t = torch.where(l1ok[qs_t],
                                  torch.ones_like(qs_t),
                                  torch.full_like(qs_t, 2))
                run_end_t = l_re[qs_t + 1]
                links = qs_t + 1 + k_t
                is_cand_pos = torch.zeros(n + 3, dtype=torch.bool, device=dev)
                is_cand_pos[qs_t] = True
                tgt = links.clamp(max=n + 2)
                chained_out = (links == run_end_t) & is_cand_pos[tgt]
                tgt_pos = torch.zeros(n + 3, dtype=torch.bool, device=dev)
                tgt_pos[tgt[chained_out]] = True
                involved = chained_out | tgt_pos[qs_t]
                qs_f = qs_t[~involved]
                if qs_f.numel():
                    starts[qs_f] = False        # the ' joins the piece before
                    consumed_of_run.scatter_reduce_(
                        0, lrid[qs_f + 1], k_t[~involved], reduce="amax")
                if bool(involved.any()):
                    qi = qs_t[involved].cpu().tolist()
                    ki = k_t[involved].cpu().tolist()
                    rei = run_end_t[involved].cpu().tolist()
                    lrid_c = lrid.cpu()
                    fired: dict[int, int] = {}
                    ended_with_suffix_at: set[int] = set()
                    for q, k, run_end in zip(qi, ki, rei, strict=True):
                        if q in ended_with_suffix_at:
                            continue    # the piece already used its slot
                        fired[q] = k
                        if q + 1 + k == run_end:
                            ended_with_suffix_at.add(run_end)
                    for q, k in fired.items():
                        starts[q] = False
                        rid_next = int(lrid_c[q + 1])
                        consumed_of_run[rid_next] = max(
                            int(consumed_of_run[rid_next]), k)
                        # The character after the apostrophe is
                        # necessarily that run's head (q + 1 == lrs), and
                        # k of its letters are eaten.
            # ---- per-run aggregation, from the effective start ----
            lrs_run = torch.full((Rl,), n, dtype=torch.long, device=dev)
            lrs_run.scatter_reduce_(0, lrid[letter], l_rs[letter],
                                    reduce="amin")
            lre_run = torch.zeros(Rl, dtype=torch.long, device=dev)
            lre_run.scatter_reduce_(0, lrid[letter], l_re[letter],
                                    reduce="amax")
            eff = lrs_run + consumed_of_run
            lastL = torch.full((Rl,), -1, dtype=torch.long, device=dev)
            lastL.scatter_reduce_(0, lrid[letter & (cls_ == L_)],
                                  ar[letter & (cls_ == L_)], reduce="amax")
            lastC = torch.full((Rl,), -1, dtype=torch.long, device=dev)
            lastC.scatter_reduce_(0, lrid[letter & c_like],
                                  ar[letter & c_like], reduce="amax")
            eff_of = eff[lrid]
            lastL_adj = torch.where(lastL >= eff, lastL, neg[:Rl])
            lastC_adj = torch.where(lastC >= eff, lastC, neg[:Rl])
            # ---- prefix hasL over contiguous {L,C} segments ----
            lc_mask = lowish
            lch, _, _, _ = _runs(lc_mask)
            isL_eff = (cls_ == L_) & (ar >= eff_of) & letter
            cntL = isL_eff.long().cumsum(0)
            # Scan-free: the head value v_k = cntL[h_k] - isL[h_k] is
            # non-decreasing along the head order (between neighbouring
            # heads v_{k+1} - v_k = sum of isL over (h_k, h_{k+1}) plus
            # isL[h_k], which is >= 0) and is itself >= 0, so cummax is
            # the last value and the default 0 matches the original.
            cnt_at_head = _fill_last(lch, cntL - isL_eff.long(), 0, ar)
            hasL_le = lc_mask & ((cntL - cnt_at_head) > 0)
            # lower_role, for C and M characters
            lower_role = c_like & (
                hasL_le | ((ar == lastC_adj[lrid])
                           & (lastL_adj[lrid] < ar)))
            # ---- piece starts ----
            # Prefix-merge test, for run heads with consumed == 0; a run
            # with consumed > 0 starts its piece explicitly at the
            # remainder, which the prefix rule does not touch.
            eff_valid = eff < lre_run
            hp = eff[eff_valid]
            is_orig_head = hp == lrs_run[eff_valid]
            prev_i = (hp - 1).clamp(min=0)
            has_prev = hp > 0
            pc = cls_[prev_i]
            merged = has_prev & is_orig_head & (
                ((pc == S_) & ~crlf[prev_i])
                | merge_fwd_char[prev_i])
            starts_head = hp[~merged]
            starts[starts_head] = True
            # Internal boundary: U-only whose previous character is
            # L-only or a C in lower role.
            prev_low = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_low[1:] = (cls_[:-1] == L_) | lower_role[:-1]
            internal = letter & (cls_ == U_) & (ar > eff_of) & prev_low \
                & (ar > l_rs)                     # inside a run, not its head
            starts |= internal
            # Letters eaten by a suffix never start a piece: the interval
            # [lrs, eff) is covered by eff, and the remainder piece at eff
            # is already part of the head logic above.

        starts[0] = True
        starts &= ~absorbed          # an absorbed character never starts

        # ---------- sparse local fallback: {P union M} super-segments ----
        # The threefold identity of marks produces "single punct stealing
        # chains" (observed against the reference splitter). Those are
        # recomputed locally by the sequential matcher, which is exact:
        # the same philosophy as the serial fallback of the parallel
        # tokenization algorithm, a narrow trigger surface and
        # unconditional correctness.
        pm = p_mask | mark
        if (pm & mark).any() and (pm & p_mask).any():
            pm_head, _, _, _ = _runs(pm)
            pmrid = pm_head.long().cumsum(0) - 1
            Rpm = int(pmrid[-1].item()) + 1
            lastM = torch.full((Rpm,), -1, dtype=torch.long, device=dev)
            lastM.scatter_reduce_(0, pmrid[pm & mark], ar[pm & mark],
                                  reduce="amax")
            # An already absorbed punct belongs to the piece before, so it
            # does not trigger.
            p_live = pm & p_mask & ~absorbed
            firstP = torch.full((Rpm,), n, dtype=torch.long, device=dev)
            firstP.scatter_reduce_(0, pmrid[p_live], ar[p_live],
                                   reduce="amin")
            # Ambiguous shape: inside the span a punct character occurs
            # before some mark, so the mark may be swallowed by punct+ or
            # take part in a stealing chain. When every mark precedes
            # every punct character (leading marks join the letter piece
            # before them) the vectorised rules are already right.
            trig = firstP < lastM
            if trig.any():
                spans = firstP[trig].cpu().tolist()
                starts = self._pm_windows(cp, starts, spans, n)
        return starts

    def _pm_windows(
        self, cp: torch.Tensor, starts: torch.Tensor, spans: list[int], n: int
    ) -> torch.Tensor:
        """{P union M} span fallback, shared by the kernel and tensor paths.

        The IO is batched: every window is pulled to the host in one
        transfer and the result is written back in one pass. Doing it one
        window at a time cost about 240 ms of host time for 166 windows
        over 64 MB, against about 6 ms on the device -- the host round
        trips, not the GPU stage, were the bottleneck. The rare window
        that has to grow past 4096 characters is fetched on its own.
        """
        seq = self._host
        spans = sorted(spans)
        W = 4096
        los = [max(sp - 1, 0) for sp in spans]
        his = [min(lo + W + 1, n) for lo in los]
        flat = np.concatenate(
            [np.arange(a, b) for a, b in zip(los, his, strict=True)]
        )
        vals = cp[torch.from_numpy(flat).to(self.dev)].cpu().numpy()
        wins: list[np.ndarray[Any, Any]] = []
        off = 0
        for a, b in zip(los, his, strict=True):
            wins.append(vals[off:off + (b - a)])
            off += b - a
        tab = self.table_np
        f_lo: list[int] = []
        f_hi: list[int] = []
        t_pos: list[int] = []
        resolved_until = -1
        for wi, sp in enumerate(spans):
            if sp <= resolved_until:
                continue
            wlo = los[wi]
            win = wins[wi]
            text_hi = wlo + len(win)
            lo = sp - 1 if (sp > 0 and win[sp - 1 - wlo] == 32) else sp
            off_w = lo - wlo
            seg = "".join(map(chr, win[off_w:]))
            hi = lo
            i = 0
            local_starts: list[int] = []
            while True:
                if lo + i >= n:
                    hi = n
                    break
                local_starts.append(lo + i)
                while True:
                    j = seq.match(seg, i, len(seg))
                    if j >= len(seg) and text_hi < n:      # grow the window
                        text_hi = min(n, text_hi + 65536)
                        win = cp[wlo:text_hi].cpu().numpy()
                        seg = "".join(map(chr, win[off_w:]))
                        continue
                    break
                i = j
                end_abs = lo + i
                # The handover point must be an S or N **run head**, that
                # is, the character before it belongs to another class.
                if end_abs >= n:
                    hi = end_abs
                    break
                ce = int(tab[int(win[end_abs - wlo])])
                cprev = int(tab[int(win[end_abs - 1 - wlo])])
                if ce in (S_, N_) and cprev != ce:
                    hi = end_abs
                    break
            f_lo.append(lo)
            f_hi.append(hi)
            t_pos.extend(local_starts)
            resolved_until = hi
        if f_lo:
            fr = np.concatenate(
                [np.arange(a, b) for a, b in zip(f_lo, f_hi, strict=True)]
            )
            starts[torch.from_numpy(fr).to(self.dev)] = False
            if t_pos:
                starts[
                    torch.tensor(t_pos, dtype=torch.long, device=self.dev)
                ] = True
        return starts

    def split(self, s: str) -> list[tuple[int, int]]:
        """Piece boundaries of one string, as ``(start, end)`` pairs."""
        if not s:
            return []
        st = self.starts(self.encode_str(s))
        idx = torch.nonzero(st).flatten().cpu().tolist()
        return list(zip(idx, [*idx[1:], len(s)], strict=True))

    # ------------------------------------------------------------------
    # Batched entry. The semantics are "call starts() per document"; the
    # implementation is the same five stages with a document boundary
    # mask, so every cross-character and cross-run propagation stops at a
    # document head. The single-document path above is untouched.
    # ------------------------------------------------------------------

    def _doc_fields(
        self, doc_offsets: torch.Tensor, n: int, ar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``doc_head`` and ``doc_end_of``, reused by every stage.

        A ``start_of`` field is not materialised: in the absorption stage
        the truncated ``cs_rs`` already clamps it from below, which is
        equivalent.
        """
        doc_head = torch.zeros(n, dtype=torch.bool, device=self.dev)
        doc_head[doc_offsets] = True
        seed = torch.where(doc_head, ar, torch.full_like(ar, n))
        rc = torch.flip(torch.cummin(torch.flip(seed, (0,)), 0).values, (0,))
        de_of = torch.cat([rc[1:], torch.full((1,), n, dtype=torch.long,
                                              device=self.dev)])
        return doc_head, de_of

    def _starts_batched_dev(
        self, cp: torch.Tensor, doc_offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure tensor stage: ``(cp, doc_offsets)`` -> ``(starts, span_mask)``.

        ``span_mask`` marks the start of each {P union M} sparse fallback
        segment (its first punct character); the host resolution happens
        in :meth:`_fallback_batched_cpu`. With no trigger, ``starts`` is
        already final.

        The only differences from :meth:`starts` are the document
        boundary masks:

        1. run segmentation: ``_runs_b`` forces a run break at a document
           head;
        2. absorption: ``prev_is_P`` does not cross a document, and the
           anchor fill is clamped from below by the truncated ``cs_rs``,
           so it cannot escape either;
        3. contraction chains: the candidate's previous character and the
           letters it eats must be in the same document, plus a
           conservative guard at the chain handover;
        4. S and P runs: the end-of-input test becomes "end of this
           document" (``de_of`` scattered per run);
        5. {P union M}: trigger segments close at document boundaries, and
           the whole batch is collected in one pass for the host resolver.
        """
        dev = self.dev
        n = cp.numel()
        ar = torch.arange(n, device=dev)
        cpl = cp.long()
        cls_ = self.table[cpl].long()
        doc_head, de_of = self._doc_fields(doc_offsets, n, ar)
        crlf = (cp == 10) | (cp == 13)
        slash = cp == 47
        asp = cp == 32
        apo = cp == 39
        mark = cls_ == M_
        c_like = (cls_ == C_) | mark
        letter = (cls_ == U_) | (cls_ == L_) | c_like
        lowish = (cls_ == L_) | c_like
        starts = torch.zeros(n, dtype=torch.bool, device=dev)
        big = torch.full((n,), n, dtype=torch.long, device=dev)
        neg = torch.full((n,), -1, dtype=torch.long, device=dev)

        # ---------- absorption (the A4 tail [\r\n/]*; point 2) ----------
        cs = crlf | slash
        _, _, cs_rs, _ = _runs_b(cs, doc_head)
        prev_is_P = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_is_P[1:] = cls_[:-1] == P_
        prev_is_P &= ~doc_head             # a document head has no previous
        anchor = cs & crlf & prev_is_P
        last_anchor = _fill_last(anchor, ar, -1, ar)
        # cs_rs is at least this document's head (the _runs_b truncation),
        # so an anchor from an earlier document is automatically below
        # cs_rs: the global last-anchor needs no segmented treatment,
        # because monotone anchors plus that lower clamp are equivalent.
        absorbed = cs & (last_anchor >= cs_rs)

        # ---------- S runs (A5/A6/A7; end test = end of this document) ---
        s_mask = cls_ == S_
        _, srid, _, s_re = _runs_b(s_mask, doc_head)
        first_live = torch.where(s_mask & ~absorbed, ar, big)
        Rs = int(srid[-1].item()) + 1 if bool(s_mask.any()) else 0
        if Rs > 0:
            fl = torch.full((Rs,), n, dtype=torch.long, device=dev)
            fl.scatter_reduce_(0, srid[s_mask], first_live[s_mask],
                               reduce="amin")
            lc_ = torch.full((Rs,), -1, dtype=torch.long, device=dev)
            lc_.scatter_reduce_(0, srid[s_mask],
                                torch.where(crlf[s_mask] & ~absorbed[s_mask],
                                            ar[s_mask], neg[s_mask]),
                                reduce="amax")
            rs_eff = fl
            re_run = torch.zeros(Rs, dtype=torch.long, device=dev)
            re_run.scatter_reduce_(0, srid[s_mask], s_re[s_mask],
                                   reduce="amax")
            de_run = torch.zeros(Rs, dtype=torch.long, device=dev)
            de_run.scatter_reduce_(0, srid[s_mask], de_of[s_mask],
                                   reduce="amax")
            has_nl = lc_ >= rs_eff
            t0 = torch.where(has_nl, lc_ + 1, rs_eff)
            m = re_run - t0
            nxt_exists = re_run < de_run       # end of input = end of doc
            idx = rs_eff[has_nl & (rs_eff < n)]
            starts[idx] = True                            # A5 piece
            tail = (m > 0) & (t0 < n)
            starts[t0[tail]] = True                       # start of the tail
            two = (m >= 2) & nxt_exists
            starts[(re_run - 1)[two]] = True              # last blank alone
        # ---------- P runs (arrival / merge-forward, same document) ------
        p_mask = cls_ == P_
        _, prid, _, p_re = _runs_b(p_mask, doc_head)
        Rp = int(prid[-1].item()) + 1 if bool(p_mask.any()) else 0
        merge_fwd_char = torch.zeros(n, dtype=torch.bool, device=dev)
        if Rp > 0:
            p_first_live = torch.full((Rp,), n, dtype=torch.long, device=dev)
            p_first_live.scatter_reduce_(
                0, prid[p_mask],
                torch.where(~absorbed[p_mask], ar[p_mask], big[p_mask]),
                reduce="amin")
            p_re_run = torch.zeros(Rp, dtype=torch.long, device=dev)
            p_re_run.scatter_reduce_(0, prid[p_mask], p_re[p_mask],
                                     reduce="amax")
            p_de_run = torch.zeros(Rp, dtype=torch.long, device=dev)
            p_de_run.scatter_reduce_(0, prid[p_mask], de_of[p_mask],
                                     reduce="amax")
            p_rs_eff = p_first_live
            live = p_rs_eff < p_re_run
            pe = p_rs_eff[live]
            prev_ok = torch.ones_like(pe, dtype=torch.bool)
            has_prev = ~doc_head[pe]       # no previous char at a doc head
            prev_ok[has_prev] = ~asp[pe[has_prev] - 1]
            plen1 = (p_re_run[live] - pe) == 1
            nxt_i = p_re_run[live].clamp(max=n - 1)
            nxt_letter = (p_re_run[live] < p_de_run[live]) & letter[nxt_i]
            mf = plen1 & nxt_letter & prev_ok    # merge target, same document
            starts[pe[prev_ok]] = True
            merge_fwd_char[pe[mf]] = True

        # ---------- N runs (the digit phase restarts per document) -------
        n_mask = cls_ == N_
        _, _, n_rs, _ = _runs_b(n_mask, doc_head)
        starts |= n_mask & ((ar - n_rs) % self.dmax == 0)

        # ---------- letter runs ----------
        _, lrid, l_rs, l_re = _runs_b(letter, doc_head)
        Rl = int(lrid[-1].item()) + 1 if bool(letter.any()) else 0
        if Rl > 0:
            # ---- contraction chains (point 3: candidate and eaten
            # letters must lie in the same document) ----
            prev_letter = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_letter[1:] = letter[:-1]
            q_mask = apo & prev_letter & ~doc_head   # previous char, same doc
            if not self.contractions:      # variant: no suffix, no chain
                q_mask &= False
            f1 = _fold_t(cpl[(ar + 1).clamp(max=n - 1)])
            f2 = _fold_t(cpl[(ar + 2).clamp(max=n - 1)])
            l1ok = torch.zeros(n, dtype=torch.bool, device=dev)
            for v in CONTR1:
                l1ok |= f1 == v
            lt1 = torch.zeros(n, dtype=torch.bool, device=dev)
            if n > 1:
                lt1[:-1] = letter[1:]
            l1ok &= lt1 & (ar + 1 < de_of)           # q + 1 in the same doc
            l2ok = torch.zeros(n, dtype=torch.bool, device=dev)
            for a, b in CONTR2:
                l2ok |= (f1 == a) & (f2 == b)
            lt2 = torch.zeros(n, dtype=torch.bool, device=dev)
            if n > 2:
                lt2[:-2] = letter[2:]
            l2ok &= lt2 & (ar + 2 < de_of)           # q + 2 in the same doc
            cand = q_mask & (l1ok | l2ok)
            consumed_of_run = torch.zeros(Rl, dtype=torch.long, device=dev)
            q_t = torch.nonzero(cand).flatten()
            if q_t.numel():
                # One gather and one device-to-host copy for the candidate
                # fields; the original read them element by element from
                # the device, which is a throughput disaster in a batch.
                # The chain loop is pure host code and decides exactly as
                # the original did.
                pack = torch.stack(
                    [q_t, l_re[q_t + 1], lrid[q_t + 1],
                     l1ok[q_t].long(), de_of[q_t]]).cpu().numpy()
                qs_, lre1_, rid1_, one_, de1_ = pack
                fired_q: list[int] = []
                fired_k: list[int] = []
                fired_rid: list[int] = []
                ended: set[int] = set()
                for t in range(qs_.shape[0]):
                    q = int(qs_[t])
                    if q in ended:
                        continue    # the piece already used its suffix slot
                    k = 1 if one_[t] else 2
                    fired_q.append(q)
                    fired_k.append(k)
                    fired_rid.append(int(rid1_[t]))
                    run_end = int(lre1_[t])
                    # Chain handover guard, deliberately conservative: the
                    # next apostrophe must still be inside this document.
                    # q_mask already blocks cross-document candidates;
                    # this is the second lock.
                    if q + 1 + k == run_end and run_end < int(de1_[t]):
                        ended.add(run_end)
                if fired_q:
                    cons: np.ndarray[Any, Any] = np.zeros(
                        Rl, dtype=np.int64)
                    np.maximum.at(cons, fired_rid, fired_k)
                    consumed_of_run = torch.from_numpy(cons).to(dev)
                    starts[torch.tensor(fired_q, dtype=torch.long,
                                        device=dev)] = False  # ' joins before
            # ---- per-run aggregation (run ids already close per doc) ----
            lrs_run = torch.full((Rl,), n, dtype=torch.long, device=dev)
            lrs_run.scatter_reduce_(0, lrid[letter], l_rs[letter],
                                    reduce="amin")
            lre_run = torch.zeros(Rl, dtype=torch.long, device=dev)
            lre_run.scatter_reduce_(0, lrid[letter], l_re[letter],
                                    reduce="amax")
            eff = lrs_run + consumed_of_run
            lastL = torch.full((Rl,), -1, dtype=torch.long, device=dev)
            lastL.scatter_reduce_(0, lrid[letter & (cls_ == L_)],
                                  ar[letter & (cls_ == L_)], reduce="amax")
            lastC = torch.full((Rl,), -1, dtype=torch.long, device=dev)
            lastC.scatter_reduce_(0, lrid[letter & c_like],
                                  ar[letter & c_like], reduce="amax")
            eff_of = eff[lrid]
            lastL_adj = torch.where(lastL >= eff, lastL, neg[:Rl])
            lastC_adj = torch.where(lastC >= eff, lastC, neg[:Rl])
            # ---- prefix hasL over {L,C} segments (segments cut per doc) --
            lc_mask = lowish
            lch, _, _, _ = _runs_b(lc_mask, doc_head)
            isL_eff = (cls_ == L_) & (ar >= eff_of) & letter
            cntL = isL_eff.long().cumsum(0)
            # Scan-free, by the same argument as in the single-document
            # path: the head value is non-decreasing along the head order
            # and non-negative, so cummax is the last value and the
            # default 0 matches.
            cnt_at_head = _fill_last(lch, cntL - isL_eff.long(), 0, ar)
            hasL_le = lc_mask & ((cntL - cnt_at_head) > 0)
            lower_role = c_like & (
                hasL_le | ((ar == lastC_adj[lrid])
                           & (lastL_adj[lrid] < ar)))
            # ---- piece starts ----
            eff_valid = eff < lre_run
            hp = eff[eff_valid]
            is_orig_head = hp == lrs_run[eff_valid]
            prev_i = (hp - 1).clamp(min=0)
            has_prev = ~doc_head[hp]     # a doc's first run head always starts
            pc = cls_[prev_i]
            merged = has_prev & is_orig_head & (
                ((pc == S_) & ~crlf[prev_i])
                | merge_fwd_char[prev_i])
            starts[hp[~merged]] = True
            prev_low = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_low[1:] = (cls_[:-1] == L_) | lower_role[:-1]
            prev_low &= ~doc_head        # defence in depth (ar > l_rs blocks)
            internal = letter & (cls_ == U_) & (ar > eff_of) & prev_low \
                & (ar > l_rs)
            starts |= internal

        starts[doc_offsets] = True   # every document head starts a piece
        # A document head is never absorbed: the anchor does not cross.
        starts &= ~absorbed

        # ---------- {P union M} trigger marks (point 5) ----------
        pm = p_mask | mark
        _, pmrid, _, _ = _runs_b(pm, doc_head)
        span_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        Rpm = int(pmrid[-1].item()) + 1 if bool(pm.any()) else 0
        if Rpm > 0:
            lastM = torch.full((Rpm,), -1, dtype=torch.long, device=dev)
            lastM.scatter_reduce_(0, pmrid[pm & mark], ar[pm & mark],
                                  reduce="amax")
            p_live = pm & p_mask & ~absorbed
            firstP = torch.full((Rpm,), n, dtype=torch.long, device=dev)
            firstP.scatter_reduce_(0, pmrid[p_live], ar[p_live],
                                   reduce="amin")
            trig = firstP < lastM
            span_mask = p_live & (ar == firstP[pmrid]) & trig[pmrid]
        return starts, span_mask

    def _fallback_batched_cpu(
        self,
        st_np: np.ndarray[Any, Any],
        cp_np: np.ndarray[Any, Any],
        cls_np: np.ndarray[Any, Any],
        spans: np.ndarray[Any, Any],
        doc_offsets_np: np.ndarray[Any, Any],
        n: int,
        text: str | None = None,
    ) -> None:
        """{P union M} sparse fallback for a whole batch, on the host.

        The trigger segments collected for the batch are recomputed
        sequentially and patched into ``st_np`` in place. Equivalent to
        the per-document fallback inside :meth:`starts`: the match window
        and the handover test are bounded by this document
        ``[dstart, dend)``, the S/N **run head** handover criterion is
        kept exactly as it is, and ``resolved_until`` resets per document.
        """
        seq = self._host
        ends_np: np.ndarray[Any, Any] = np.append(doc_offsets_np[1:], n)
        di_of: np.ndarray[Any, Any] = np.asarray(
            np.searchsorted(doc_offsets_np, spans, side="right")) - 1
        resolved_until = -1
        cur_doc = -1
        for sp, d in zip(spans.tolist(), di_of.tolist(), strict=True):
            dstart = int(doc_offsets_np[d])
            dend = int(ends_np[d])
            if d != cur_doc:
                cur_doc = d
                resolved_until = -1    # never reuse across a document
            if sp <= resolved_until:
                continue
            lo = sp - 1 if sp > dstart and cp_np[sp - 1] == 32 else sp
            hi = lo
            text_hi = min(dend, lo + 4096)
            seg = (text[lo:text_hi] if text is not None
                   else "".join(map(chr, cp_np[lo:text_hi])))
            i = 0
            local_starts: list[int] = []
            while True:
                if lo + i >= dend:
                    hi = dend
                    break
                local_starts.append(lo + i)
                while True:
                    j = seq.match(seg, i, len(seg))
                    if j >= len(seg) and text_hi < dend:
                        text_hi = min(dend, text_hi + 65536)
                        seg = (text[lo:text_hi] if text is not None
                               else "".join(map(chr, cp_np[lo:text_hi])))
                        continue
                    break
                i = j
                end_abs = lo + i
                # The handover must be an S or N run head inside this
                # document, or the end of the document; the run-head
                # criterion is not weakened, and a document boundary
                # counts as end of input.
                if end_abs >= dend or (cls_np[end_abs] in (S_, N_)
                                       and cls_np[end_abs - 1]
                                       != cls_np[end_abs]):
                    hi = end_abs
                    break
            st_np[lo:hi] = False
            for pos in local_starts:
                st_np[pos] = True
            resolved_until = hi

    def _starts_batched_cuda(
        self, cp: torch.Tensor, off: torch.Tensor
    ) -> torch.Tensor | None:
        """Batched kernel path (document-start channel; runs stay inside).

        Sparse windows are resolved on the device within document bounds,
        with ``resolved_until`` reset across documents. A chain with no
        safe restart point (extreme) returns None, and the caller yields
        the whole batch to the tensor path.
        """
        ext = self.ext
        n = cp.numel()
        dstart = torch.zeros(n, dtype=torch.uint8, device=self.dev)
        dstart[off] = 1
        st, pm_trig, chain_trig, chain = ext.pretok_starts_batched_o200k(
            cp, dstart, self.table, self.dmax, self.contractions)
        off_np = off.cpu().numpy()
        ends_np: np.ndarray[Any, Any] = np.append(off_np[1:], n)

        def doc_bounds(
            pos_list: np.ndarray[Any, Any],
        ) -> tuple[list[int], list[int], list[int]]:
            di: np.ndarray[Any, Any] = np.asarray(
                np.searchsorted(off_np, pos_list, side="right")) - 1
            return (di.tolist(), off_np[di].tolist(), ends_np[di].tolist())

        if int(chain.item()):
            qpos = torch.nonzero(chain_trig[:n]).flatten().cpu().tolist()
            qdoc, _, _ = doc_bounds(np.asarray(qpos))
            g_sp: list[int] = []
            g_qL: list[int] = []
            g_doc: list[int] = []
            for q, d in zip(qpos, qdoc, strict=True):
                if g_qL and d == g_doc[-1] and q - g_qL[-1] <= 64:
                    g_qL[-1] = q       # same document and within 64: one group
                else:
                    g_sp.append(q)
                    g_qL.append(q)
                    g_doc.append(d)
            _, g_ds, g_de = doc_bounds(np.asarray(g_sp))
            if not self._win_gpu(cp, st, g_sp, g_qL, chain_mode=True,
                                 ds_list=g_ds, de_list=g_de,
                                 doc_list=g_doc):
                return None
        spans_t = torch.nonzero(pm_trig[:n]).flatten()
        if spans_t.numel():
            spans = spans_t.cpu().tolist()
            sdoc, sds, sde = doc_bounds(np.asarray(spans))
            self._win_gpu(cp, st, spans, spans, chain_mode=False,
                          ds_list=sds, de_list=sde, doc_list=sdoc)
        return st

    def starts_batched(
        self,
        cp: torch.Tensor,
        doc_offsets: torch.Tensor,
        text: str | None = None,
    ) -> torch.Tensor:
        """Piece starts for several documents in one batch.

        ``doc_offsets`` holds the start of every non-empty document in
        ``cp``, ascending and including 0. The returned mask is in
        concatenated coordinates.
        """
        n = cp.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.dev)
        off = doc_offsets.to(device=self.dev).long()
        # Defensive: sort, de-duplicate and drop out-of-range offsets.
        off = torch.unique(off[off < n])
        if off.numel() == 0 or int(off[0].item()) != 0:
            off = torch.unique(torch.cat(
                [torch.zeros(1, dtype=torch.long, device=self.dev), off]))
        # The kernel path is chosen per batch profile. Measured on the
        # reference device, the starts kernel alone is about 42 times the
        # tensor version (0.8 against 34.2 ms per 8 MB), but it needs
        # about four host synchronisations per batch, which breaks the
        # asynchronous overlap a batched pipeline lives on (about -12% on
        # a small-document profile). So small-document batches, where the
        # overlap dominates, stay on the tensor path, and large-document
        # batches, where the starts stage dominates, take the kernel. The
        # threshold is a mean document length and is a tuning option.
        avg_doc = n // max(int(off.numel()), 1)
        if (self.use_cuda and not self._host_win
                and avg_doc >= self._batch_cuda_min):
            st_cuda = self._starts_batched_cuda(cp, off)
            if st_cuda is not None:
                return st_cuda            # None means: yield the whole batch
        st, span_mask = self._starts_batched_dev(cp, off)
        if bool(span_mask.any()):
            st_np = st.cpu().numpy()
            cp_np = cp.cpu().numpy()
            spans = np.flatnonzero(span_mask.cpu().numpy())
            self._fallback_batched_cpu(st_np, cp_np, self.table_np[cp_np],
                                       spans, off.cpu().numpy(), n, text)
            st = torch.from_numpy(st_np).to(self.dev)
        return st

    def split_docs(self, docs: list[str]) -> list[list[tuple[int, int]]]:
        """Piece list per document, in document-local coordinates."""
        lens = [len(d) for d in docs]
        joined = "".join(docs)
        if not joined:
            return [[] for _ in docs]
        cp = self.encode_str(joined)
        offs: list[int] = []
        acc = 0
        for ln in lens:
            offs.append(acc)
            acc += ln
        doc_offsets = torch.tensor(
            sorted({o for o, ln in zip(offs, lens, strict=True) if ln > 0}
                   | {0}),
            dtype=torch.long, device=self.dev)
        st = self.starts_batched(cp, doc_offsets, text=joined)
        idx = torch.nonzero(st).flatten().cpu().tolist()
        bounds = list(zip(idx, [*idx[1:], len(joined)], strict=True))
        # Single forward pass, O(bounds + docs): every document head has a
        # start, so no piece can cross a document boundary.
        out: list[list[tuple[int, int]]] = [[] for _ in docs]
        ends = [o + ln for o, ln in zip(offs, lens, strict=True)]
        di = 0
        for a, b in bounds:
            while di < len(docs) and a >= ends[di]:
                di += 1
            if di < len(docs) and a >= offs[di] and b <= ends[di]:
                out[di].append((a - offs[di], b - offs[di]))
        return out


def doc_start_arrays(
    st_np: np.ndarray[Any, Any], offs: list[int], docs: list[str]
) -> list[np.ndarray[Any, Any]]:
    """Concatenated starts -> one local piece-start array per document.

    A searchsorted form, so that callers that only need the offsets do
    not pay for materialising tuples.
    """
    idx = np.flatnonzero(st_np)
    ends = [o + len(d) for o, d in zip(offs, docs, strict=True)]
    los: np.ndarray[Any, Any] = np.asarray(np.searchsorted(idx, offs))
    his: np.ndarray[Any, Any] = np.asarray(np.searchsorted(idx, ends))
    return [idx[lo:hi] - o
            for lo, hi, o in zip(los, his, offs, strict=True)]
