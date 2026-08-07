# 0002 — Mainline ratifications at the S1 freeze (2026-08-05)

Rulings on the four items the contract author left open. With these, the
contract set 0001 + 0002 is frozen as v1.

1. **certified_source is eligible under the default CERTIFIED policy** —
   ratified. Rationale: the wave-1 GPU path is JIT-built, so a stricter
   reading would leave the default policy with no accelerated route at
   all; eligibility is conditional on every bound constraint verifying
   (kernel source digest, build flags, toolchain constraints, class-table
   digest, oracle semantic id), and the status is labeled
   `certified_source`, never `certified`, in every user-facing surface.
2. **Store record semantics: full core stream per record** — ratified as
   specified. The store lane must confirm during its equivalence pass
   that this matches the ported v1 behavior; if it needs sealed-ids-only
   records instead, the field-semantics text in store-format-v1.md §2.1
   is the single place to amend, before any store file leaves this repo.
3. **Bounds constants** (text <= 2^40 bytes, tail <= 2^31 bytes,
   tokens <= 2^31) — ratified unchanged.
4. **`family` accepts registry ids only** — ratified. Local artifact
   paths, if ever supported, arrive as a separate constructor parameter
   in a later minor version, not as an overload of `family`.

Process note: decisions in this repository are made by the project
maintainers only; reviews from any assistant lane are advisory input.
