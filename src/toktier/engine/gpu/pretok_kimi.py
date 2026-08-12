r"""Piece starts for the kimi pre-tokenization band, computed on the GPU.

The band
--------
"kimi" is the splitter group whose pattern has eight leftmost-first
alternatives: the seven of the o200k band, plus a leading Han branch.

    b0  [\p{Han}]+                       (new: a Han run is one piece)
    b1  [^\r\n\p{L}\p{N}]? [U L C M && ^Han]* [L C M && ^Han]+ (?i:'s|...)?
    b2  [^\r\n\p{L}\p{N}]? [U L C M && ^Han]+ [L C M && ^Han]* (?i:'s|...)?
    b3  \p{N}{1,3}
    b4  ' ?[^\s\p{L}\p{N}]+[\r\n]*       (the tail has no slash)
    b5  \s*[\r\n]+      b6  \s+(?!\S)      b7  \s+

Because b0 is leftmost, a match that starts on a Han character always
takes b0 and consumes the maximal Han run, mixed subclasses included; a
Han character can only ever be reached by another branch in the middle of
a piece.

What this module computes
-------------------------
:class:`GpuPretokKimi` is the split layer: it produces the boolean
piece-start mask and nothing else. The band's end-to-end encoder is that
mask followed by the shared byte-level BPE layer, which is where the
rest of the chain lives.

The implementation is the five propagation stages of the o200k band
instantiated for this family (the shared run primitives are imported from
that module), with four structural changes:

1. **Ten-class table.** The ``&&`` class differences are resolved when
   the table is generated: the letter classes U/L/C/M hold no Han, and
   three Han subclasses are added -- HL (Han that is Lm or Lo), HN (Han
   that is a number) and HP (Han that is neither letter nor number nor
   whitespace, so it belongs to the punct class).
2. **The absorbed tail is [\r\n]*, with no slash.** The cross-run slash
   absorption of the o200k band does not exist here, so the shape
   degenerates to the older band's: a CR/LF streak is a prefix of a
   single whitespace run, and the anchor mechanism is reused unchanged.
3. **b0, vectorised: the head of a Han run always starts a piece.** For a
   run of purely letter-like Han, the boundaries at both ends are already
   produced correctly by the existing per-run rules against all six
   neighbouring classes (the argument is pair by pair), because HL
   appears in no other stage's mask and is therefore isolated by
   construction.
4. **HN and HP are resolved exactly by a sparse fallback.** HN takes part
   in the digit phase (the numeric mask includes it) and HP in the punct
   arm, and both interleave with b0 in a way that is inherently
   sequential. So any occurrence of HN or HP triggers a local sequential
   recomputation with the matcher below. Its anchor is the nearest clean
   whitespace run head or document head, and its handover is a clean
   whitespace, digit or Han-letter run head. The trigger surface is 345
   rare codepoints, and correctness is unconditional -- the same
   discipline as the serial fallback of the parallel tokenization
   algorithm. The {P union M} super-segment fallback is inherited from
   the o200k band unchanged (the threefold identity of marks and the
   single-punct stealing chains it produces).

Case backtracking and the contraction suffix chain are structurally
identical to the o200k band; only class membership changes (Han removed).

Semantics are pinned to the reference tokenizer's own splitter: the rules
were derived from, and differentially checked against, the reference
engine's regex splitter on this artifact. The class masks behind the
table must come from that engine itself and never from another Unicode
table -- a table built from a different Unicode version skews the letter
and unassigned sets and silently changes where pieces begin. The Han
subclass ranges are likewise machine-extracted and cross-checked against
an independent probe of the engine, never transcribed by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .options import DEFAULT_GPU_OPTIONS, GpuOptions
from .pretok_o200k import (
    C_,
    CONTR1,
    CONTR2,
    CONTR_ONE,
    CONTR_TWO,
    L_,
    M_,
    N_,
    P_,
    S_,
    U_,
    _fold_ch,
    _fold_t,
    _runs_b,
)

__all__ = ["GpuPretokKimi"]

#: Han subclass values. Values 0 to 6 are the o200k class values, which
#: this table reuses by construction (the generator emits one enum for
#: both); 7 to 9 are added here. Everything Han has a value >= HL_.
HL_, HN_, HP_ = 7, 8, 9

#: The digit alternative of this band is \p{N}{1,3}. The value is fixed
#: by the artifact's own pattern and is compiled into the kernel, so it
#: is not a free parameter of this class.
DIGITS_MAX = 3


class _HostMatcherKimi:
    """Sequential leftmost-first matcher over the kimi class table.

    This is the exact semantics the sparse fallback recomputes with: the
    alternatives are tried in pattern order and the first one that
    matches wins. Its predicates read the same class table as the tensor
    stages, so the two cannot disagree about what a character is.
    """

    def __init__(self, table: np.ndarray[Any, Any], dmax: int) -> None:
        # A bytes view of the table: indexing it yields a Python int
        # directly, which is what the per-character loops below want.
        self._cls: bytes = table.astype(np.uint8, copy=False).tobytes()
        self._dmax = dmax

    # -- character predicates, all derived from the class table --------

    def _is_han(self, ch: str) -> bool:
        """\\p{Han}: the union of the three subclasses."""
        return self._cls[ord(ch)] >= HL_

    def _is_ws(self, ch: str) -> bool:
        return self._cls[ord(ch)] == S_

    def _is_num(self, ch: str) -> bool:
        """\\p{N}, which includes the numeric Han subclass."""
        return self._cls[ord(ch)] in (N_, HN_)

    def _is_letter(self, ch: str) -> bool:
        """\\p{L}, which includes the letter-like Han subclass."""
        return self._cls[ord(ch)] in (U_, L_, C_, HL_)

    def _is_upperish(self, ch: str) -> bool:
        """[Lu Lt Lm Lo M && ^Han], the first letter class of b1/b2."""
        return self._cls[ord(ch)] in (U_, C_, M_)

    def _is_lowerish(self, ch: str) -> bool:
        """[Ll Lm Lo M && ^Han], the second letter class of b1/b2."""
        return self._cls[ord(ch)] in (L_, C_, M_)

    def _is_punct4(self, ch: str) -> bool:
        """[^\\s\\p{L}\\p{N}]: non-Han marks and the punct Han subclass."""
        return self._cls[ord(ch)] in (P_, M_, HP_)

    def _prefix_ok(self, ch: str) -> bool:
        """[^\\r\\n\\p{L}\\p{N}], the optional prefix character of b1/b2."""
        return (
            ch not in "\r\n"
            and not self._is_letter(ch)
            and not self._is_num(ch)
        )

    # -- the alternatives ----------------------------------------------

    def _suffix(self, s: str, e: int, n: int) -> int:
        """Greedy (?i:'s|'t|'re|'ve|'m|'ll|'d)? tail, folded as in o200k."""
        if e < n and s[e] == "'" and e + 1 < n:
            f1 = _fold_ch(s[e + 1])
            if f1 in CONTR_ONE:
                return e + 2
            if e + 2 < n and (f1, _fold_ch(s[e + 2])) in CONTR_TWO:
                return e + 3
        return e

    def _letters(self, s: str, i: int, n: int, need_lower: bool) -> int:
        """Letter body of b1 (``need_lower``) or b2: U*L+ or U+L*.

        Line for line the o200k form, with the Han-free class members.
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
        c = s[i]
        # b0: leftmost branch. A match that starts on Han always lands
        # here and swallows the whole run, mixed subclasses included.
        if self._is_han(c):
            j = i + 1
            while j < n and self._is_han(s[j]):
                j += 1
            return j
        e = self._letters(s, i, n, need_lower=True)      # b1
        if e >= 0:
            return e
        e = self._letters(s, i, n, need_lower=False)     # b2
        if e >= 0:
            return e
        if self._is_num(c):             # b3, may absorb HN in mid-piece
            j = i + 1
            while j < n and j - i < self._dmax and self._is_num(s[j]):
                j += 1
            return j
        # b4 ' '? punct+ [\r\n]*: may absorb HP in mid-piece, a space
        # followed by HP opens the arm, and there is no slash in the tail.
        p = i
        if c == " " and i + 1 < n and self._is_punct4(s[i + 1]):
            p = i + 1
        if p < n and self._is_punct4(s[p]):
            j = p + 1
            while j < n and self._is_punct4(s[j]):
                j += 1
            while j < n and s[j] in "\r\n":
                j += 1
            return j
        if self._is_ws(c):                               # b5 / b6 / b7
            j = i + 1
            while j < n and self._is_ws(s[j]):
                j += 1
            run_end = j
            t = run_end - 1
            while t >= i:
                if s[t] in "\r\n":
                    return t + 1            # b5: cut after the last CR/LF
                t -= 1
            if run_end == n:
                return run_end              # b6 at end of input
            if run_end - i >= 2:
                return run_end - 1          # b6 gives the last blank back
            return run_end                  # b7
        return i + 1        # defensive: the pattern covers everything


class GpuPretokKimi:
    """Piece starts for the kimi band (split layer only).

    Args:
        ext: The compiled kernel extension module. This class never loads
            it: one process owns one build, and the loader hands the
            module in.
        class_table: The verified ten-class table for this family.
        digits_max: Maximum number of digits in a b3 piece. Fixed at 3 by
            the artifact's own pattern and compiled into the kernel, so
            the only accepted value is 3; it is an argument so that a
            caller wiring it from registry data finds out loudly rather
            than silently splitting digits differently from the kernel.
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
        digits_max: int = DIGITS_MAX,
        device: str = "cuda:0",
        options: GpuOptions | None = None,
    ) -> None:
        if digits_max != DIGITS_MAX:
            raise ValueError(
                "the kimi splitter's digit alternative is fixed at "
                f"{DIGITS_MAX} digits by its own pattern and by the "
                f"kernel; digits_max={digits_max} cannot be honoured"
            )
        opts = DEFAULT_GPU_OPTIONS if options is None else options
        self.ext = ext
        self.options = opts
        self.dev = torch.device(device)
        self.dmax = digits_max
        self.table_np = class_table    # host copy: the fallback needs it
        self.table = torch.from_numpy(class_table).to(self.dev)
        self._host = _HostMatcherKimi(class_table, digits_max)
        # Kernel path for the piece-start stage. The pure tensor path is
        # kept as an explicit differential reference and as the fallback
        # for the shapes the kernel hands back; both produce the same
        # starts.
        self.use_cuda = opts.kimi_cuda_starts

    @classmethod
    def from_family(
        cls,
        ext: Any,
        table: Any,
        *,
        family: Any,
        digits_max: int,
        options: GpuOptions,
    ) -> GpuPretokKimi:
        """Uniform construction hook for the engine's data-driven dispatch.

        ``family`` is accepted for signature uniformity; this band's
        splitter takes no per-family variant parameters beyond
        ``digits_max``, which the constructor pins against the value the
        kernel compiled in.
        """
        del family  # signature uniformity; no variant parameters here
        return cls(
            ext,
            table.array,
            digits_max=digits_max,
            device=options.device,
            options=options,
        )

    def encode_str(self, s: str) -> torch.Tensor:
        """UTF-32 codepoints of ``s`` as an int32 tensor on the device."""
        cp: np.ndarray[Any, Any] = np.frombuffer(
            s.encode("utf-32-le"), dtype=np.uint32)
        return torch.from_numpy(cp.astype(np.int32)).to(self.dev)

    # ------------------------------------------------------------------
    # Pure tensor stage: the o200k batched stage instantiated for this
    # family. The document boundary masks are structurally identical
    # point by point. Returns (starts, trig_sp, trig_lo): trig_lo == -1
    # means a {P union M} trigger, where the resolver picks the anchor
    # with the o200k leading-space rule; otherwise trig_lo is the anchor
    # the device already computed for an HN or HP trigger.
    # ------------------------------------------------------------------

    def _starts_batched_dev(
        self, cp: torch.Tensor, doc_offsets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.dev
        n = cp.numel()
        ar = torch.arange(n, device=dev)
        cpl = cp.long()
        cls_ = self.table[cpl].long()
        doc_head = torch.zeros(n, dtype=torch.bool, device=dev)
        doc_head[doc_offsets] = True
        seed = torch.where(doc_head, ar, torch.full_like(ar, n))
        rc = torch.flip(torch.cummin(torch.flip(seed, (0,)), 0).values, (0,))
        de_of = torch.cat([rc[1:], torch.full((1,), n, dtype=torch.long,
                                              device=dev)])
        crlf = (cp == 10) | (cp == 13)
        asp = cp == 32
        apo = cp == 39
        mark = cls_ == M_          # non-Han marks; the table removed Han
        c_like = (cls_ == C_) | mark
        letter = (cls_ == U_) | (cls_ == L_) | c_like     # no Han in here
        lowish = (cls_ == L_) | c_like
        han = cls_ >= HL_                     # union of the three subclasses
        num = (cls_ == N_) | (cls_ == HN_)    # the b3 digit phase includes HN
        starts = torch.zeros(n, dtype=torch.bool, device=dev)
        big = torch.full((n,), n, dtype=torch.long, device=dev)
        neg = torch.full((n,), -1, dtype=torch.long, device=dev)

        # ---------- absorption (the b4 tail [\r\n]*; no slash) ----------
        cs = crlf
        _, _, cs_rs, _ = _runs_b(cs, doc_head)
        prev_is_P = torch.zeros(n, dtype=torch.bool, device=dev)
        prev_is_P[1:] = cls_[:-1] == P_
        prev_is_P &= ~doc_head
        anchor = cs & crlf & prev_is_P
        last_anchor = torch.cummax(torch.where(anchor, ar, neg), 0).values
        absorbed = cs & (last_anchor >= cs_rs)

        # ---------- S runs (b5/b6/b7; end test = end of this document) ---
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
            nxt_exists = re_run < de_run
            idx = rs_eff[has_nl & (rs_eff < n)]
            starts[idx] = True                            # b5 piece
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
            has_prev = ~doc_head[pe]
            prev_ok[has_prev] = ~asp[pe[has_prev] - 1]
            plen1 = (p_re_run[live] - pe) == 1
            nxt_i = p_re_run[live].clamp(max=n - 1)
            nxt_letter = (p_re_run[live] < p_de_run[live]) & letter[nxt_i]
            mf = plen1 & nxt_letter & prev_ok
            starts[pe[prev_ok]] = True
            merge_fwd_char[pe[mf]] = True

        # ---------- N runs (b3 phase restarts per document; HN included) --
        _, _, n_rs, _ = _runs_b(num, doc_head)
        starts |= num & ((ar - n_rs) % self.dmax == 0)

        # ---------- letter runs (o200k mechanism; class members lack Han) -
        _, lrid, l_rs, l_re = _runs_b(letter, doc_head)
        Rl = int(lrid[-1].item()) + 1 if bool(letter.any()) else 0
        if Rl > 0:
            prev_letter = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_letter[1:] = letter[:-1]
            q_mask = apo & prev_letter & ~doc_head
            f1 = _fold_t(cpl[(ar + 1).clamp(max=n - 1)])
            f2 = _fold_t(cpl[(ar + 2).clamp(max=n - 1)])
            l1ok = torch.zeros(n, dtype=torch.bool, device=dev)
            for v in CONTR1:
                l1ok |= f1 == v
            lt1 = torch.zeros(n, dtype=torch.bool, device=dev)
            if n > 1:
                lt1[:-1] = letter[1:]
            l1ok &= lt1 & (ar + 1 < de_of)
            l2ok = torch.zeros(n, dtype=torch.bool, device=dev)
            for a, b in CONTR2:
                l2ok |= (f1 == a) & (f2 == b)
            lt2 = torch.zeros(n, dtype=torch.bool, device=dev)
            if n > 2:
                lt2[:-2] = letter[2:]
            l2ok &= lt2 & (ar + 2 < de_of)
            cand = q_mask & (l1ok | l2ok)
            consumed_of_run = torch.zeros(Rl, dtype=torch.long, device=dev)
            q_t = torch.nonzero(cand).flatten()
            if q_t.numel():
                # One gather and one device-to-host copy for the candidate
                # fields; the chain loop is pure host code and decides
                # exactly as the sequential matcher would.
                pack = torch.stack(
                    [q_t, l_re[(q_t + 1).clamp(max=n - 1)],
                     lrid[(q_t + 1).clamp(max=n - 1)],
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
                    # Chain handover guard: the next apostrophe must still
                    # be inside this document.
                    if q + 1 + k == run_end and run_end < int(de1_[t]):
                        ended.add(run_end)
                if fired_q:
                    cons: np.ndarray[Any, Any] = np.zeros(
                        Rl, dtype=np.int64)
                    np.maximum.at(cons, fired_rid, fired_k)
                    consumed_of_run = torch.from_numpy(cons).to(dev)
                    starts[torch.tensor(fired_q, dtype=torch.long,
                                        device=dev)] = False  # ' joins before
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
            lc_mask = lowish
            lch, _, _, _ = _runs_b(lc_mask, doc_head)
            isL_eff = (cls_ == L_) & (ar >= eff_of) & letter
            cntL = isL_eff.long().cumsum(0)
            cnt_at_head = torch.cummax(torch.where(
                lch, cntL - isL_eff.long(), torch.zeros_like(ar)), 0).values
            hasL_le = lc_mask & ((cntL - cnt_at_head) > 0)
            lower_role = c_like & (
                hasL_le | ((ar == lastC_adj[lrid])
                           & (lastL_adj[lrid] < ar)))
            eff_valid = eff < lre_run
            hp = eff[eff_valid]
            is_orig_head = hp == lrs_run[eff_valid]
            prev_i = (hp - 1).clamp(min=0)
            has_prev = ~doc_head[hp]
            pc = cls_[prev_i]
            merged = has_prev & is_orig_head & (
                ((pc == S_) & ~crlf[prev_i])
                | merge_fwd_char[prev_i])
            starts[hp[~merged]] = True
            prev_low = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_low[1:] = (cls_[:-1] == L_) | lower_role[:-1]
            prev_low &= ~doc_head
            internal = letter & (cls_ == U_) & (ar > eff_of) & prev_low \
                & (ar > l_rs)
            starts |= internal

        # ---------- b0: the head of a Han run always starts a piece ------
        # A run of purely letter-like Han is closed correctly at both ends
        # by the existing rules, since HL is in no other mask. The head of
        # an HN or HP run can be swallowed by the piece before it (the
        # digit phase or the punct arm), but every occurrence of HN or HP
        # triggers the fallback below, which rewrites the whole segment,
        # so a vectorised misjudgement there is always overwritten.
        h_head, _, _, _ = _runs_b(han, doc_head)
        starts |= h_head

        starts[doc_offsets] = True   # every document head starts a piece
        starts &= ~absorbed

        # ---------- collect the fallback triggers ----------
        # (a) {P union M} super-segments, exactly as in the o200k band:
        # the threefold identity of marks and its stealing chains.
        pm = p_mask | mark
        _, pmrid, _, _ = _runs_b(pm, doc_head)
        Rpm = int(pmrid[-1].item()) + 1 if bool(pm.any()) else 0
        span_a = torch.zeros(n, dtype=torch.bool, device=dev)
        if Rpm > 0:
            lastM = torch.full((Rpm,), -1, dtype=torch.long, device=dev)
            lastM.scatter_reduce_(0, pmrid[pm & mark], ar[pm & mark],
                                  reduce="amax")
            p_live = pm & p_mask & ~absorbed
            firstP = torch.full((Rpm,), n, dtype=torch.long, device=dev)
            firstP.scatter_reduce_(0, pmrid[p_live], ar[p_live],
                                   reduce="amin")
            trig = firstP < lastM
            span_a = p_live & (ar == firstP[pmrid]) & trig[pmrid]
        # (b) HN and HP positions: the digit phase and the punct arm
        # interleave with b0, so the decision is sequential. The anchor is
        # the nearest document head or clean whitespace run head. Clean
        # means: not (a CR/LF whose previous character is in the punct
        # class {P, M, HP}), because the b4 tail is the only channel that
        # can swallow the head of a whitespace run; every other shape of
        # whitespace run head is always a true piece start.
        hanx = (cls_ == HN_) | (cls_ == HP_)
        trig_sp = torch.zeros(0, dtype=torch.long, device=dev)
        trig_lo = torch.zeros(0, dtype=torch.long, device=dev)
        sp_a = torch.nonzero(span_a).flatten()
        if bool(hanx.any()):
            prev_s = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_s[1:] = s_mask[:-1]
            prev_p4 = torch.zeros(n, dtype=torch.bool, device=dev)
            prev_p4[1:] = (cls_[:-1] == P_) | (cls_[:-1] == M_) \
                | (cls_[:-1] == HP_)
            clean = doc_head | (s_mask & (~prev_s | doc_head)
                                & ~(crlf & ~doc_head & prev_p4))
            last_clean = torch.cummax(
                torch.where(clean, ar, neg), 0).values
            sp_b = torch.nonzero(hanx).flatten()
            lo_b = last_clean[sp_b]
            trig_sp = torch.cat([sp_a, sp_b])
            trig_lo = torch.cat(
                [torch.full_like(sp_a, -1), lo_b])
        elif sp_a.numel():
            trig_sp = sp_a
            trig_lo = torch.full_like(sp_a, -1)
        return starts, trig_sp, trig_lo

    def _fallback_cpu(
        self,
        st_np: np.ndarray[Any, Any],
        cp_np: np.ndarray[Any, Any],
        cls_np: np.ndarray[Any, Any],
        trigs: np.ndarray[Any, Any],
        los: np.ndarray[Any, Any],
        doc_offsets_np: np.ndarray[Any, Any],
        n: int,
        text: str | None = None,
    ) -> None:
        """Sparse exact fallback on the host.

        The kimi extension of the o200k batched fallback. ``lo == -1``
        means a {P union M} trigger, where the anchor follows the o200k
        leading-space rule; otherwise ``lo`` is the anchor the device
        already computed. The handover must be a clean run head -- a
        whitespace run head (previous character not whitespace), a digit
        run head (previous character neither a digit nor a numeric Han),
        or a Han-letter run head (previous character not Han) -- and it
        must lie past the trigger position.
        """
        seq = self._host
        ends_np: np.ndarray[Any, Any] = np.append(doc_offsets_np[1:], n)
        order = np.argsort(trigs, kind="stable")
        trigs, los = trigs[order], los[order]
        di_of: np.ndarray[Any, Any] = np.asarray(
            np.searchsorted(doc_offsets_np, trigs, side="right")) - 1
        resolved_until = -1
        cur_doc = -1

        def handoff(e: int) -> bool:
            c = cls_np[e]
            p = cls_np[e - 1]
            if c == S_:
                return bool(p != S_)
            if c == N_:
                return bool(p != N_ and p != HN_)
            if c == HL_:
                return bool(p < HL_)
            return False

        for sp, lo0, d in zip(trigs.tolist(), los.tolist(), di_of.tolist(),
                              strict=True):
            dstart = int(doc_offsets_np[d])
            dend = int(ends_np[d])
            if d != cur_doc:
                cur_doc = d
                resolved_until = -1
            if sp <= resolved_until:
                continue
            if lo0 < 0:      # {P union M}: the o200k anchor rule
                lo = sp - 1 if sp > dstart and cp_np[sp - 1] == 32 else sp
            else:            # HN or HP: the precomputed clean anchor,
                lo = max(int(lo0), dstart)   # never before this document
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
                if end_abs >= dend or (end_abs > sp and handoff(end_abs)):
                    hi = end_abs
                    break
            st_np[lo:hi] = False
            for pos in local_starts:
                st_np[pos] = True
            resolved_until = hi

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
        off = torch.unique(off[off < n])
        if off.numel() == 0 or int(off[0].item()) != 0:
            off = torch.unique(torch.cat(
                [torch.zeros(1, dtype=torch.long, device=self.dev), off]))
        st, trig_sp, trig_lo = self._starts_batched_dev(cp, off)
        if trig_sp.numel():
            st_np = st.cpu().numpy()
            cp_np = cp.cpu().numpy()
            self._fallback_cpu(st_np, cp_np, self.table_np[cp_np],
                               trig_sp.cpu().numpy(), trig_lo.cpu().numpy(),
                               off.cpu().numpy(), n, text)
            st = torch.from_numpy(st_np).to(self.dev)
        return st

    def starts(self, cp: torch.Tensor) -> torch.Tensor:
        """Piece starts for one document.

        There is no separate single-document implementation: this is the
        batched path with one document, so the two cannot drift apart.
        With ``kimi_cuda_starts`` on, the kernel resolves the sparse
        surface in device windows and extreme shapes yield to this
        reference path.
        """
        if self.use_cuda:
            st = self._starts_cuda(cp)
            if st is not None:
                return st
        return self.starts_batched(
            cp, torch.zeros(1, dtype=torch.long, device=self.dev))

    def _starts_cuda(self, cp: torch.Tensor) -> torch.Tensor | None:
        """Kernel path for one document.

        The three sparse surfaces share one window protocol: {P union M}
        uses mode 2, whose anchor steps back over a leading space, while
        HN/HP triggers and contraction chains use mode 3, which searches
        backwards for a clean anchor. The windows are merged, sorted by
        trigger position and passed through a single ``resolved_until``
        in the same order the host fallback uses. Phase one computes the
        extents per mode, phase two applies them per mode; the clear and
        mark steps are two separate passes, and two windows agree on
        their overlap, which is what makes applying them safe. If the
        anchor search fails (an extreme shape) this returns None and the
        caller yields to the reference path.
        """
        ext = self.ext
        n = cp.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.dev)
        st, pm_trig, chain_trig, hanx_trig, chain = ext.pretok_starts_kimi(
            cp, self.table)
        wins: list[tuple[int, int, int]] = []      # (sp, qL, mode)
        if int(chain.item()):
            qpos = torch.nonzero(chain_trig[:n]).flatten().cpu().tolist()
            g_sp: list[int] = []
            g_qL: list[int] = []
            for q in qpos:
                if g_qL and q - g_qL[-1] <= 64:
                    g_qL[-1] = q
                else:
                    g_sp.append(q)
                    g_qL.append(q)
            wins += [(a, b, 3) for a, b in zip(g_sp, g_qL, strict=True)]
        hx = torch.nonzero(hanx_trig[:n]).flatten()
        if hx.numel():
            wins += [(h, h, 3) for h in hx.cpu().tolist()]
        pml = torch.nonzero(pm_trig[:n]).flatten()
        if pml.numel():
            wins += [(s, s, 2) for s in pml.cpu().tolist()]
        if not wins:
            return st
        wins.sort()
        i32: dict[str, Any] = {"dtype": torch.int32, "device": self.dev}
        lo_h = [0] * len(wins)
        hi_h = [0] * len(wins)
        for mode in (2, 3):
            idx = [k for k, w in enumerate(wins) if w[2] == mode]
            if not idx:
                continue
            sp_t = torch.tensor([wins[k][0] for k in idx], **i32)
            qL_t = torch.tensor([wins[k][1] for k in idx], **i32)
            zz = torch.zeros(len(idx), **i32)
            de = torch.full((len(idx),), n, **i32)
            lo, hi, _nosafe = ext.o200k_win_extents(
                cp, self.table, sp_t, qL_t, zz, de, self.dmax, True, mode)
            eh = torch.stack([lo, hi]).cpu()
            el, ei = eh[0].tolist(), eh[1].tolist()
            for j, k in enumerate(idx):
                lo_h[k], hi_h[k] = el[j], ei[j]
        if any(v < 0 for v in lo_h):
            return None                   # no clean anchor: yield to torch
        sel: list[int] = []
        ru = -1
        for k, (_sp, qL, _m) in enumerate(wins):
            # Skip test: the whole trigger surface is already covered by
            # the resolved region (qL <= ru). For the single-point pm and
            # hanx triggers (sp == qL) this is word for word the ``sp``
            # rule of the host fallback; for a chain group (sp < qL) it is
            # the necessary generalisation, because a chain tail reaching
            # past the resolved region must still be applied. Two windows
            # agree on their overlap, so applying is safe.
            if qL <= ru:
                continue
            sel.append(k)
            ru = hi_h[k]
        for mode in (2, 3):
            kk = [k for k in sel if wins[k][2] == mode]
            if not kk:
                continue
            ext.o200k_win_apply(
                cp, self.table, st,
                torch.tensor([lo_h[k] for k in kk], **i32),
                torch.tensor([hi_h[k] for k in kk], **i32),
                torch.tensor([wins[k][1] for k in kk], **i32),
                torch.full((len(kk),), n, **i32), self.dmax, True, mode)
        return st

    def split(self, s: str) -> list[tuple[int, int]]:
        """Piece boundaries of one string, as ``(start, end)`` pairs."""
        if not s:
            return []
        cp = self.encode_str(s)
        st = self.starts_batched(
            cp, torch.zeros(1, dtype=torch.long, device=self.dev), text=s)
        idx = torch.nonzero(st).flatten().cpu().tolist()
        return list(zip(idx, [*idx[1:], len(s)], strict=True))

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
        out: list[list[tuple[int, int]]] = [[] for _ in docs]
        ends = [o + ln for o, ln in zip(offs, lens, strict=True)]
        di = 0
        for a, b in bounds:
            while di < len(docs) and a >= ends[di]:
                di += 1
            if di < len(docs) and a >= offs[di] and b <= ends[di]:
                out[di].append((a - offs[di], b - offs[di]))
        return out
