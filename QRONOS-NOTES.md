# Qronos → GPTQ-Pro port — math mapping and deliberate deviations

Reference: `reference/qronos.py` (+ `gpfq.py`, `gpxq.py`) from
`i-colbert/brevitas@qronos` (paper arXiv:2505.11695). Port target:
**GPTQ-Pro latest main (`reference/gptq-pro/`, groxaxo fork @ cd57c06,
version string 6.1.0-dev)** — NOT the upstream 7.3.2 tree also vendored
under reference/ (kept only for lineage comparison; review m7). The port is
structured exactly like the fork's FOEM solver (`quantization/foem.py`),
which is byte-identical between the two trees.
This file is the review brief: every reviewer verifies the code against THIS
mapping and this mapping against the reference.

## Statistics (what add_batch/process_batch accumulates)

Reference (qronos.py update_batch — note it OVERRIDES gpfq.py, and the
override flips G's orientation on purpose):
- H = E[x̂ x̂ᵀ]   (quantized-path input covariance; nsamples counted here)
- G = E[x x̂ᵀ]   (FLOAT input × QUANT inputᵀ — `inp_processed.bmm(quant_input.T)`
  where inp_processed is the float pass; gpfq.py's own G is the TRANSPOSED
  convention — do not copy gpfq)

Port (mirrors FOEM.process_batch):
- `inp` given to add_batch = X̂ (the GPTQ processor forward runs the
  progressively-quantized model)
- `native_inp` popped from NATIVE_INPUTS_STATE_KEY = X (NativeProcessor runs
  the unquantized model first) — same source FOEM/GPTAQ use
- H += (√(2/n) x̂)(√(2/n) x̂)ᵀ with the same EMA renormalization FOEM uses
- G += (√(2/n) x)(√(2/n) x̂)ᵀ  ← orientation: rows=float dims, cols=quant dims
- Scale convention: GPTQModel's 2/n vs reference's 1/n — uniform across H
  and G, so every update formula below is invariant; damp is derived from
  the same-scaled H. DEVIATION-1 (benign, documented).

## The update (single_layer_update → quantize())

Let W = current weight [OC, IC] (float32 working copy), W_orig = original
weights, all in the reference's group-index-0 case (LLM Linear ⇒ groups=1;
port RAISES on conv — DEVIATION-2, out of scope for LLMs).

0. dead = diag(H)==0 → W[:, dead]=0 and NOTHING ELSE (rev 3, faithful to
   the reference; the rev-2 extra zeroing was proven non-equivalent —
   math C1 / adv M-1, dispositions below. G's dead rows carry the float
   path's real signal and must survive; step 2 restores W_orig for
   both-paths-dead columns via iH[d,d] ≈ 1/damp).
1. damp = percdamp × λ_max(H) via 30-iteration power iteration
   (reference `_power_iteration`; percdamp default 1e-5). iH =
   cholesky_inverse(cholesky(H + damp·I)). LinAlgError → warn + return
   weights unmodified (null correction), exactly like reference.
2. Dhi = 1/diag(H) (0 where dead). Uh = triu(H, 1).
3. STEP 1 (column 0, closed-form): with v = W (current), w = W_orig:
   q_arg = (w · G[:,0]) · Dhi[0] − (v · Uh[0,:]) · Dhi[0]
   W[:, 0] = q_arg.
   (Reference fetches q via get_quant_weights(0,0,hist=True) = EMPTY and
   never uses it — dead code, not ported.)
4. SMW downdate: iH ← iH[1:,1:] − iH[1:,0:1]·iH[0:1,1:]/iH[0,0]
   (per reference: A −= b bᵀ / c). After this, iH indexes columns 1..IC-1.
5. STEP 2 (columns 1..IC-1, closed-form interim): q0 = quantize(W[:, 0])
   on a PROVISIONAL group-0 grid; after the solve the grid is REFIT on the
   actual values and W is delta-updated (rev 3 two-pass — DEVIATION-6b,
   see GRID below). Gh = G + damp·I.
   W[:, 1:] = W_orig · (Gh[:, 1:] @ iH) − q0 · (H[0:1, 1:] @ iH)
   (Reference: get_quant_weights(0, 1, hist=True) = the quantized column 0
   of the CURRENT W — i.e. quantize(step-1's q_arg). H row 0 is UNDAMPED;
   Gh is DAMPED. Match exactly.)
6. L = chol_upper(iH·c)/√c with c=1e4 (numerical conditioning, reference
   default). LinAlgError → warn + return (weights keep the step-1/2 values —
   same as reference behavior).
7. BLOCK LOOP over columns 1..IC-1 (rev 3.2): edges = group boundaries,
   plus blocksize multiples ONLY when blocksize < group_size (F4 — at
   g128/bs128 the offset grids would interleave 127/1 slivers; at
   group_size=-1 the blocksize cap is what prevents one unblocked rank-1
   sweep, math O4). Every group boundary is an edge (all prior diffusion
   visible at find_params, the M2 fix). Historical
   record on the retired DEVIATION-3 claim: partition choice DOES change Q
   through grid timing (rev-1 measurement: 2/3 of columns flipped, max
   |dQ| 15% of range) but the objective moved only −0.053% — the alignment
   is for determinism and GPTQ-parity, not because the old partition was
   measurably worse (math O7).
   For each column j in block: q = quantize(W[:, j]) on its group grid;
   err = (W[:, j] − q)/L[j−1, j−1]  ← NOTE the −1 shift: L indexes the
   downdated (size IC−1) system. In-block update:
   W[:, j:] −= err ⊗ L[j−1, j−1:i2−1]; after block: W[:, i2:] −=
   Err_block · L[i1−1:i2−1, i2−1:].
   Loss[:, j] = (w−q)²/d² accumulated for avg_loss reporting (GPTQ
   convention; reference does not track loss — ADDITION-1).
8. Q assembled column-by-column from the quantize() outputs (the reference
   materializes Q implicitly through the layer quantizer; the port stores q
   as produced — identical grid).

## GRID (scales/zeros) — GPTQModel conventions

- quantizer.find_params(W[:, k:k+g]) at each group boundary k on the
  THEN-CURRENT W, scales appended in group order; g_idx = i//g; return
  contract identical to FOEM: (Q, scale, zero, g_idx, duration, avg_loss,
  damp, nsamples).
- Group 0's params are found via the two-pass fixed point around step 5
  (provisional + ≤3 refit rounds, rev 3.1) and the FINAL grid is the one
  stored and reused when the block loop covers columns 1..g−1 — the loop
  never sees j=0 and must not re-find at a j=0 boundary. Review point R1:
  off-by-one on group boundary bookkeeping.
- desc_act (act_order): NOT SUPPORTED in v1 — explicit raise
  (DEVIATION-4). Our recipe uses desc_act=False. The reference act_order
  path permutes H, G, and the weight view; adding it later needs its own
  differential test.
- static_groups: NOT SUPPORTED in v1 — explicit raise (DEVIATION-5).
- sym int4 g128 is the target configuration; the code must not assume sym
  (grid comes from Quantizer), but tests pin sym.
- DEVIATION-6/6b FINAL (rev 3; grid timing): the reference's grid is
  frozen from the ORIGINAL weights (brevitas
  scaling_impl_type='parameter_from_stats' — resolved, see round-2 m7).
  The port instead uses: **group 0 = two-pass fixed point** (provisional
  grid → provisional q0 → step-2 solve → refit on the actual
  [q_arg | step-2] values → re-quantize q0 → exact delta-update of
  W[:,1:]; packer-consistent by construction), **groups ≥1 = GPTQ-lazy**
  (boundary, then-current diffused W, group-aligned blocks). Both deviate
  from the frozen-originals reference in a MEASURED quality-positive
  direction (adv H-1: frozen/stale grids cost 1-2 bits of group-0
  resolution at high float/quant path drift). KNOWN CORRELATED BLIND SPOT:
  the test transcription mirrors these conventions deliberately (choices,
  not algebra) — the differential covers the algebra only.

## Failure containment (all kept or strengthened vs reference)

- NaN in H/G → raise ValueError (rev 2: loud raises, not -O-strippable
  asserts; reference uses asserts)
- q_arg non-finite OR outside finfo(weight.dtype) → raise (matches the
  reference's dtype-bound assert; strengthened with a weak-H[0,0]
  fingerprint warn pre-grid and a >20× group-0-scale warn post-run)
- Total inversion failure or zero-batch module → the STOCK
  `_fallback_quantize` (RTN + string loss "fallback(rtn): …" — rev 2,
  review C3/M3; replaces rev 1's hand-rolled _rtn_fallback). Mid-path L
  failure → per-group RTN on the step-1/2 reconstruction, all group scales
  emitted, loud warn.
- avg_loss NaN → raise (GPTQModel convention).

## Review points for the agents

R1 group-boundary bookkeeping with the loop starting at column 1.
R2 the −1 index shift everywhere L/iH touch columns (SMW downdate shrinks
   the system by one).
R3 dead-column handling: FAITHFUL as of rev 3 (only W[:, dead]=0); any
   reintroduction of G/W_orig/H zeroing must re-clear math C1.
R4 G orientation (XX̂ᵀ, not X̂Xᵀ) — verify against qronos.py:88, NOT
   gpfq.py:109.
R5 damped-vs-undamped usage: Gh damped in step 5; H row 0 undamped in
   step 5; Dhi undamped in step 3; iH from damped H.
R6 EMA/2-n normalization consistency between H, G, and damp.
R7 memory: buffers freed in the same order as reference (B n/a, G, H, iH,
   L); no fp64 anywhere; fp32 accumulation throughout.
R8 the null-correction fallback emits valid scales for ALL groups.

## Review round 1 dispositions (reviews/review-integration.md, rev 2 of the solver)

- C1 GAR: Qronos now RAISES on act_group_aware=True; the driver sets it
  False on Qronos arms; postsave records False. GAR-vs-solver confound in
  cross-arm reads: arms compare TOTAL recipes (S-arms carry GAR, S4 does
  not) — stated, not hidden. Note the reviewer's finding that FOEM/GPTAQ
  (and therefore armD) share the same GAR-skip hole.
- C2 batch-order: multi-device parallel forward can desync the float/quant
  pairing. Qronos __init__ RAISES if device_count>1 without
  auto_forward_data_parallel=False. Rentals are single-GPU. The deep fix
  (batch_index-keyed caches) is deferred and documented.
- C3 zero-batch: quantize() now routes H-is-None / under-threshold modules
  to the STOCK _fallback_quantize (string loss, telemetry-correct).
- M1/M2 (grid + block conventions): DEVIATION-6b — group-0 grid from
  W_orig (the reference's own pre-fixed-grid convention; the packer
  requires one grid per group and q0 precedes step 2); blocks are now
  GROUP-ALIGNED (edges at k*g) so boundary find_params sees all prior
  diffusion. The test transcription mirrors both conventions (they are
  choices, not algebra — the differential test covers the algebra).
- M3 fallback telemetry: stock fallback path everywhere; string loss.
- M4 loss: column 0 excluded from the loss; /2 GPTQ convention applied.
- M5 memory: G freed before the L allocation; in-place iH.mul_(c);
  excepts widened to (LinAlgError, RuntimeError); CPU-OOM ladder NOT
  ported (96GB rental; peak ~5xIC^2 ~ 6GB documented).
- M6: to_device() rehome hook moves H, G, module_copy.
- M7: Q returned on the weight's home device.
- M8: raise on _tp_pad_cols.
- M9: activation_weighted_mse honoured (importance= into every
  find_params); hessian-chunking NOT honoured (process_batch overridden)
  — postsave normalises the artifact metadata for it and for fallback.
- m1: damp slot reports percdamp (relative), lambda_max logged separately.
- m2 free() clears native_inps; m3 folded into M5; m4 asserts->raises;
  m5 construction-time type check; m6 driver must import qronos_gptqmodel
  BEFORE model.quantize (fail-fast on PYTHONPATH mistakes); m7 this header
  fixed; n1 patch comment fixed; n2/n3 dead code removed; n4 one matvec
  per power iteration; n5 columns==1 -> stock fallback.
- m8 (end-to-end integration test through model.quantize): OPEN — planned
  as the on-rental 2-layer smoke gate rather than a local test (no GPU
  locally; the smoke is mandatory in the runbook before any full arm).

## Review round 2 dispositions (review-math.md + review-adversarial.md → rev 3)

- math C1 / adv M-1 (dead columns): rev 2's three extra zeroing lines
  DELETED — handling is now faithful (only W[:, dead] = 0). G's dead ROWS
  survive (float-path compensation, the point of Qronos); step 2 restores
  W_orig for both-paths-dead columns exactly like the reference; the damped
  inverse needs no H[dead,dead]=1 (damp > 0 always). Two differential
  dead-column tests added (quant-only-dead and both-dead).
- adv H-1 / math m2 / int M1 (group-0 grid): resolved with the TWO-PASS
  fixed point — provisional grid → provisional q0 → step-2 solve → refit
  grid on the actual values ([q_arg | step-2 cols]) → re-quantize q0 →
  exact delta-update W[:,1:] += outer(q0_prev − q0_final, h0iH). Stored
  grid and Q stay packer-consistent (the packer's integer re-derivation is
  UNCLAMPED — adversarial measured 20-21 on a 0..15 grid for the naive
  refit, which is why the delta pass exists).
- math m7 RESOLVED (2026-08-17): brevitas's LLM flow uses
  scaling_impl_type='parameter_from_stats' (common/generative/quantize.py:
  380-386 on the qronos branch) — the reference's grid is materialized ONCE
  from the ORIGINAL weights and frozen. DEVIATION-6 final form: the port's
  grids (two-pass group 0, lazy diffused-W groups ≥1) deviate from the
  reference's frozen-originals grid in a measured quality-positive
  direction (adv H-1 table: frozen/stale grids cost 1-2 bits of group-0
  resolution at high path drift).
- math M1 (step-1 containment): bound restored against the LAYER dtype's
  finfo (not fp32); avg_loss now rejects inf too; >20x group-0-scale
  fingerprint warning added.
- math M3 / adv L-2 (degraded-path loss): chol-failure path reports
  string loss "fallback(chol): <mean_abs_err>"; total-failure and
  zero-batch paths use the stock fallback's string loss.
- math m1 / adv L-2 (loss basis): column-0 loss on the pre-downdate
  iH[0,0] basis; /2 convention applied.
- math m4 (reproducibility): power iteration now torch.rand start from a
  dedicated generator seeded from (columns, name-length) — damp is
  run-to-run deterministic. CAVEAT (adv H-2b): bit-reproducibility holds
  per BLAS backend/device only; round-half ties flip int4 levels under
  1e-7 perturbations, so byte-identical CPU-vs-CUDA artifacts are NOT
  expected — announce-anchor-style byte comparisons must fix the backend.
- math m5: degenerate-damp floor made relative to H's scale.
- math m6 DEVIATION-7: the port computes fp32 throughout and casts once at
  the end; the reference rounds every write to the layer dtype (bf16/fp16).
  The port's way is numerically better and matches GPTQModel convention —
  declared, since a brevitas output diff will show it.
- math M2 / adv H-2 (oracle): transcription now mirrors rev-3 conventions;
  damp passed deterministically (solver_damp helper); dead-column
  differentials added; a 30-config randomized sweep asserts well-formedness
  + objective-within-2%-of-transcription (stable under tie-flips) instead
  of pretending exact bit equality generalizes.
- adv M-2 / math n4 (memory): H deleted right after its last use (h0iH);
  W_orig deleted after step 2; G before the L allocation; in-place mul_;
  peak square-buffer count reduced from ~5 to ~3 x IC^2.
- adv L-3: empty native-cache pop raises a NAMED ValueError (test added).
- adv L-4: the dead full-matrix find_params (an 80-iter MSE search per
  layer for nothing) removed; groups_found dropped for a REAL emitted-
  groups invariant raise; dead placeholder/nan lines gone (rev 2).
- int m1/adv L-1: damp slot returns percdamp (fraction) — column-compatible.
- Attacks that held (adversarial): SMW exactness, -1 shifts, block-partition
  cleanliness at fixed grid, Gh construction (bit-exact), group counts,
  hostile statistics, degenerate shapes, packer round-trip on bf16/fp16,
  no dtype leakage, inference_mode parity, del ordering.

## Review round 3 dispositions (math lane's rev-3 re-review → rev 3.1)

Math lane re-verified all its rev-2 findings as FIXED in rev 3 and left
5 MINOR + 2 NIT open; all closed in rev 3.1:
- O1: weak-column-0 fingerprint added BEFORE grid fitting (warn when
  H[0,0] < 1e-3 × median diag) — catches the 3-20× band the dtype bound
  and the 20× scale warn both miss.
- O2: group-0 refit iterated to a fixed point (≤3 rounds; exact rank-1
  delta each round); oracle mirrors it.
- O3: the oracle now has its OWN power-iteration transcription;
  test_power_iteration_oracle_agrees pins port/oracle agreement.
- O4: block edges = union of group boundaries and blocksize multiples —
  blocksize is meaningful again and group_size=-1 no longer degenerates
  to one unblocked sweep.
- O5/O7: NOTES text synced to the code (failure-containment bullet;
  block-loop item rewritten with the preserved DEVIATION-3 measurement).
- O6: dead c00 clamp removed (c00_pre strictly positive; commented).
Suite: 23/23 after the round.

## Review round 4 dispositions (review-final.md fresh-eyes pass → rev 3.2)

Verdict was: no CRITICAL; 1 MAJOR + 3 MINOR + 3 NIT; all prior-round
dispositions VERIFIED in code. Closed as follows:
- F1 (MAJOR): run_qronos_postsave.py now rewrites BOTH config.json and
  quantize_config.json (the loader checks quantize_config.json FIRST) with
  one shared _rewrite_meta, and ends with a verification pass that exits
  non-zero if meta.gptaq survives in either file.
- F2: the driver's --qronos branch imports qronos_gptqmodel at config-build
  time — a bad PYTHONPATH now dies in second 1, not at layer 0 (patch v5).
- F3: an identically-zero H (hooks fired, all inputs zero) routes to the
  stock fallback with a loud warn instead of reporting avg_loss~0.
- F4: blocksize edges are only added when blocksize < group_size — the
  127/1 alternation at the production g128/bs128 config is gone; the
  group_size=-1 capping case (O4's real target) is preserved.
- F5: memory statement corrected here: square-buffer peak ~3.3 x IC^2
  measured (round-1's "~5x/6GB" line is retired; the ~5x figure described
  rev 2 before the round-2 frees).
- F6: partition-invariance statement, precise form: the OBJECTIVE is
  blocksize-invariant (6 decimals); Q is invariant only up to round-half
  tie flips (whole-level, isolated); determinism holds at fixed blocksize,
  which production has (default 128).
- F7: weak-column-0 detection threshold raised to 1.5e-2 x median diag
  (catches the 3-5x band the 1e-3 version measurably missed) PLUS the
  dtype-independent amplitude fingerprint (|q_arg| > 4x |W_orig[:,0]|).
  Corrected claim: covered band is eps <= ~0.1 (>=3x widening); eps=0.3
  (~1.4x widening) intentionally stays silent.
Suite: 23/23 after the round. The reviewer's §"Residual risk" +
8-point checklist = the ON-RENTAL smoke gate (referenced from
launch-kit/ARMS-RUNBOOK.md).
