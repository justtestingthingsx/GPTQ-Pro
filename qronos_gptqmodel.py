# SPDX-License-Identifier: Apache-2.0
# Qronos solver for GPTQ-Pro (groxaxo fork, latest main cd57c06, version
# string 6.1.0-dev) — house port, 2026-08-17, revision 3
# (post integration + math + adversarial reviews; see reviews/*.md and the
# "Review round" sections of NOTES.md for every finding's disposition).
#
# Implements the Qronos rounding/update algorithm (arXiv:2505.11695) as a
# GPTQ-Pro solver class, structured like gptqmodel/quantization/foem.py
# (the FOEM/GPTAQ sibling). Reimplemented from the paper's equations and the
# reference structure (i-colbert/brevitas@qronos, src/brevitas/graph/qronos.py);
# no reference code is copied verbatim (the fork's license is unasserted).
#
# v1 scope (enforced by explicit raises):
#   - nn.Linear / transformers.Conv1D only (LLM path; groups==1)
#   - desc_act (act_order) NOT supported
#   - static_groups NOT supported
#   - act_group_aware (GAR) NOT supported — must be explicitly False (int C1)
#   - tensor-parallel column padding NOT supported (int M8 / math m8)
#   - single-device forward only (int C2)

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import transformers

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.fallback import resolve_fallback_strategy, should_use_fallback
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.torch import torch_sync

log = setup_logger()


def _power_iteration(mat: torch.Tensor, iters: int = 30,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Estimate the dominant eigenvalue of a symmetric PSD matrix.

    Mirrors the reference damping basis (brevitas magr._power_iteration):
    damp = percdamp * lambda_max(H). Uniform-positive start like the
    reference's torch.rand (covariance-like H has a near-positive dominant
    eigenvector, so this converges faster than a randn start — math m4), one
    mat-vec per iteration (int n4). A seeded generator makes damp — and
    therefore the artifact — reproducible run-to-run (math m4; NOTE: bit
    reproducibility holds per BLAS backend/device, not across them —
    adversarial H-2b).
    """
    n = mat.shape[-1]
    if generator is not None:
        v = torch.rand(n, dtype=torch.float32, device=mat.device, generator=generator)
    else:
        v = torch.rand(n, dtype=torch.float32, device=mat.device)
    v_norm = torch.linalg.vector_norm(v)
    if v_norm == 0 or not torch.isfinite(v_norm):
        v = torch.ones(n, dtype=torch.float32, device=mat.device)
        v_norm = torch.linalg.vector_norm(v)
    v = v / v_norm
    lam = torch.zeros((), dtype=torch.float32, device=mat.device)
    for _ in range(iters):
        w = mat @ v
        w_norm = torch.linalg.vector_norm(w)
        if w_norm == 0 or not torch.isfinite(w_norm):
            # Degenerate (H ~ 0): return 0; caller floors the damp.
            return torch.zeros((), dtype=torch.float32, device=mat.device)
        lam = torch.dot(v, w)  # Rayleigh quotient with the pre-normalized v
        v = w / w_norm
    return lam


class Qronos(GPTQ):
    """Qronos solver: GPTQ-shaped error correction that additionally corrects
    the error inherited from quantizing PREVIOUS layers, using the
    cross-covariance between the float-path and quantized-path activations.

    Statistics (NOTES.md 'Statistics'):
        H = c_n * sum x_hat x_hat^T   (quantized-path inputs; GPTQ convention)
        G = c_n * sum x x_hat^T       (float inputs x quantized inputs^T)
    with the same running normalization FOEM uses, so H and G share scale.
    """

    def __init__(self, module: NamedModule, qcfg: Optional[QuantizeConfig] = None):
        from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY  # avoid import loop

        super().__init__(module, qcfg)

        # v1 scope checks at construction time, not mid-calibration (int m5).
        if not isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            raise NotImplementedError(
                "Qronos v1 supports nn.Linear / transformers.Conv1D only "
                f"(got {type(self.module)})."
            )
        if getattr(self, "_tp_pad_cols", 0):
            raise NotImplementedError(
                "Qronos v1 does not support tensor-parallel column padding "
                "(int M8 / math m8).")
        # int C2: multi-GPU parallel forward interleaves hook firing order
        # across devices, silently desynchronizing the float/quant batch
        # pairing. Refuse unless data-parallel forward is explicitly disabled
        # or there is only one visible device.
        if torch.cuda.device_count() > 1 and getattr(
                self.qcfg, "auto_forward_data_parallel", True):
            raise RuntimeError(
                "Qronos requires deterministic batch order: run with a single "
                "visible CUDA device or set "
                "quantize_config.auto_forward_data_parallel=False."
            )

        self.H = None
        self.G = None

        # percdamp basis for the Qronos-side damping (reference default 1e-5,
        # applied to lambda_max(H) — NOT the GPTQ damp_percent*mean(diag)).
        self.qronos_percdamp = getattr(self.qcfg, "qronos_percdamp", 1e-5)
        # Cholesky conditioning constant from the reference (c=1e4).
        self.qronos_chol_c = 1e4

        # The float-path inputs collected by NativeProcessor (same source as
        # GPTAQ/FOEM). Popped per batch in process_batch.
        self.native_inps = module.state.pop(NATIVE_INPUTS_STATE_KEY)

    # ------------------------------------------------------------------ stats

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        with self.lock:
            self.fwd_counter += 1
            self.process_batch(inp)

    def process_batch(self, inp):
        # `inp` is the QUANTIZED-path activation (the looper forwards the
        # progressively quantized model); `native_inp` is the FLOAT-path
        # activation cached by NativeProcessor. R4: G = x x_hat^T.
        inp = inp.to(dtype=torch.float32)
        if not self.native_inps:
            # adversarial L-3: name the desync instead of a bare IndexError.
            raise ValueError(
                f"Qronos `{self.name}`: NativeProcessor cache exhausted — "
                "more quant-path batches than cached float-path batches "
                "(calibration streams out of sync).")
        native_inp = self.native_inps.pop(0).to(device=inp.device, dtype=torch.float32)

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            native_inp = native_inp.unsqueeze(0)

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            native_inp = native_inp.reshape((-1, native_inp.shape[-1]))
        # house (S4 launch fix): count TOKEN rows, the same unit as stock
        # GPTQ's `batch_token_size` — expected_nsamples is
        # total_calibration_tokens, so counting forwards (old
        # `inp.shape[0]` on the 3D tensor = 1) made the rank-starvation
        # fallback fire on EVERY module of every real run (256 << 524288).
        # H/G renorm and sqrt(2/n) scaling are ratio-consistent in either
        # unit; the gate, avg_loss and samples telemetry are not.
        batch_size = inp.shape[0]
        # After reshape both are [tokens, columns]; transpose to
        # [columns, tokens] for covariance accumulation.
        inp = inp.t()
        native_inp = native_inp.t()

        if inp.shape != native_inp.shape:
            raise ValueError(
                f"Qronos `{self.name}`: quant-path and native-path activation "
                f"shapes differ ({tuple(inp.shape)} vs {tuple(native_inp.shape)}); "
                "the NativeProcessor cache is out of sync with the calibration "
                "batches."
            )

        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32, device=inp.device)
            self.G = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32, device=inp.device)
        else:
            renorm = self.nsamples / (self.nsamples + batch_size)
            self.H *= renorm
            self.G *= renorm

        self.nsamples += batch_size
        scale = math.sqrt(2 / self.nsamples)
        inp = scale * inp
        native_inp = scale * native_inp

        # R6: identical scale on both operands of both products keeps H and G
        # on a shared scale; the update math is invariant to that shared scale.
        self.H += inp.matmul(inp.t())
        self.G += native_inp.matmul(inp.t())
        del native_inp, inp

    # -------------------------------------------------------------- lifecycle

    def to_device(self, device: torch.device):
        """Rehome hook for module_looper._rehome_processor_task (int M6): move
        every live solver tensor together with the module."""
        if self.H is not None:
            self.H = self.H.to(device=device)
        if self.G is not None:
            self.G = self.G.to(device=device)
        if self.module_copy is not None:
            self.module_copy = self.module_copy.to(device=device)

    def free(self):
        super().free()
        if getattr(self, "G", None) is not None:
            self.G = None
        # int m2: drop any unconsumed float-path activations.
        if getattr(self, "native_inps", None) is not None:
            self.native_inps = None

    # ------------------------------------------------------------ inverse ops

    def _qronos_inverse(self, H: torch.Tensor):
        """Return (iH, damp_used) where iH = (H + damp I)^-1 via Cholesky.

        Damping basis mirrors the reference: percdamp * lambda_max(H), with a
        bounded escalation ladder for numerically hostile blocks (each step
        logged so the arm stays auditable). Deterministic start vector (math
        m4). Catches RuntimeError too — CUDA linear-algebra failures do not
        always surface as LinAlgError. Returns (None, None) if every attempt
        fails.
        """
        gen = torch.Generator(device=H.device)
        # int(): self.columns arrives as numpy.int64 from
        # get_number_of_rows_and_cols, which manual_seed rejects.
        gen.manual_seed((int(self.columns) * 1000003 + len(self.name or "")) & 0x7FFFFFFF)
        lam_max = _power_iteration(H, 30, generator=gen)
        base = float(lam_max)
        if not math.isfinite(base) or base <= 0:
            # math m5: floor RELATIVE to H's scale (falls back to an absolute
            # epsilon only when H is exactly zero everywhere).
            diag_scale = float(torch.diag(H).abs().max())
            base = max(float(torch.mean(torch.diag(H)).clamp(min=0)),
                       1e-8 * max(diag_scale, 1.0))
        damp = self.qronos_percdamp * base
        log.info(
            f"Quantization(Qronos): `{self.name}` lambda_max~{base:.4e} "
            f"damp={damp:.4e}")

        for attempt in range(6):
            try:
                iH = H.clone()
                iH.diagonal().add_(damp)
                L = torch.linalg.cholesky(iH)
                iH = torch.cholesky_inverse(L)
                del L
                if attempt > 0:
                    log.warn(
                        f"Quantization(Qronos): `{self.name}` needed damp "
                        f"escalation x{attempt} (damp={damp:.3e}).")
                return iH, damp
            except (torch.linalg.LinAlgError, RuntimeError) as e:
                log.warn(
                    f"Quantization(Qronos): `{self.name}` inverse attempt "
                    f"{attempt} failed ({type(e).__name__}); escalating damp.")
                damp *= 10.0
                continue
        return None, None

    def _stock_fallback(self, blocksize, H=None):
        """Route to the stock RTN fallback (string loss, proper telemetry —
        int M3). Restores self.H if the caller had already detached it."""
        if H is not None:
            self.H = H
        if self.H is None:
            self.H = self.create_H(
                target_device=getattr(self.module, "target_device", None))
        strategy = resolve_fallback_strategy(self.fallback)
        log.warn(
            f"Quantization(Qronos): `{self.name}` -> stock "
            f"`{strategy.value}` fallback path.")
        return self._fallback_quantize(strategy, blocksize)

    # --------------------------------------------------------------- quantize

    @torch.inference_mode()
    def quantize(self, blocksize=128):
        start = time.time()

        if self.qcfg.desc_act:
            raise NotImplementedError(
                "Qronos v1 does not support desc_act/act_order (DEVIATION-4); "
                "set desc_act=False.")
        if self.qcfg.static_groups:
            raise NotImplementedError(
                "Qronos v1 does not support static_groups (DEVIATION-5).")
        if getattr(self.qcfg, "act_group_aware", False):
            raise NotImplementedError(
                "Qronos v1 does not implement GAR (act_group_aware); set it "
                "to False explicitly on the Qronos arm so config and "
                "artifact metadata state the truth (int C1).")
        if self.qcfg.group_size == 0:
            raise ValueError("group_size must be -1 or a positive integer")

        # int C3: zero-batch / under-threshold modules take the configured
        # fallback instead of crashing on H=None or solving a rank-starved H.
        if self.H is None or should_use_fallback(
                self.fallback, float(self.nsamples), self.expected_nsamples):
            return self._stock_fallback(blocksize)

        # int M7: remember where the weight lives; Q returns there.
        result_device = torch.device(self.module.weight.data.device)
        # math M1: the step-1 bound check is against the LAYER's dtype (the
        # reference asserts representability in weight.dtype, not fp32).
        weight_finfo = torch.finfo(self.module.weight.dtype)

        if self.module_copy is None:
            W = self.clone_module(device=self.H.device)
        else:
            W = self.module_copy
            self.module_copy = None

        # Original float weights (the reference's weight_orig): W is the live
        # working copy; keep a frozen fp32 original for steps 1 and 2.
        W = W.to(dtype=torch.float32)
        W_orig = W.detach().clone()
        # (adversarial L-4: the rev-1/2 full-matrix find_params here was dead
        # code AND an 80-iteration MSE search over the whole matrix — removed.)

        H = self.H
        G = self.G
        self.H = None
        self.G = None

        # int m4: loud errors, not -O-strippable asserts.
        if torch.isnan(H).any():
            raise ValueError(f"Qronos `{self.name}`: NaN in H")
        if torch.isnan(G).any():
            raise ValueError(f"Qronos `{self.name}`: NaN in G")

        columns = self.columns
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else columns

        # n5: a single-column module has no post-downdate system at all.
        if columns == 1:
            return self._stock_fallback(blocksize, H=H)

        # int M9: honour activation-weighted MSE scale search like stock GPTQ.
        activation_importance = None
        if getattr(self.qcfg, "activation_weighted_mse", False):
            activation_importance = torch.diag(H).clamp_min(0).to(
                device=W.device, dtype=W.dtype)
            imp_mean = activation_importance.mean()
            if torch.isfinite(imp_mean) and imp_mean > 0:
                activation_importance = activation_importance / imp_mean
            else:
                activation_importance = None

        def _find_group_params(src: torch.Tensor, lo: int, hi: int):
            imp = None
            if activation_importance is not None:
                imp = activation_importance[lo:hi]
            self.quantizer.find_params(src[:, lo:hi], weight=True, importance=imp)

        # math C1 / adversarial M-1 (rev 3): dead-column handling is now
        # FAITHFUL to the reference — only the working copy's dead columns are
        # zeroed. G's dead COLUMNS are exactly zero already (x_hat_d == 0);
        # G's dead ROWS carry the float path's real signal and MUST survive
        # (they are how surviving columns compensate for a channel the quant
        # path lost — the point of Qronos). No H[dead,dead]=1: the damped
        # inverse handles it (damp > 0 always), iH[d,d] ~= 1/damp, and step 2
        # then RESTORES W_orig's dead columns exactly like the reference.
        dead = torch.diag(H) == 0
        if bool(dead.all()):
            # review-final F3: an all-zero H means the hooks fired but every
            # input channel was identically zero — the solve would be a
            # silent RTN-on-originals with a near-zero numeric loss. Route to
            # the stock fallback so the telemetry says what happened.
            log.warn(
                f"Quantization(Qronos): `{self.name}` H is identically zero "
                f"across all {columns} columns (nsamples={self.nsamples}) — "
                "calibration produced no signal; using stock fallback.")
            return self._stock_fallback(blocksize, H=H)
        if dead.any():
            W[:, dead] = 0

        iH, damp = self._qronos_inverse(H)
        if iH is None:
            log.warn(
                f"Quantization(Qronos): `{self.name}` Hessian inversion failed "
                "after damp escalation; using stock fallback (null correction).")
            return self._stock_fallback(blocksize, H=H)

        # math m1: capture the pre-downdate inverse diagonal — the correct
        # loss basis for column 0 (1/iH[0,0] ~ the downdate-free d^2).
        c00_pre = float(iH[0, 0])

        # Dhi: UNDAMPED reciprocal diagonal (R5); zero where dead.
        Dh = torch.diag(H).clone()
        Dhi = torch.where(dead, torch.zeros_like(Dh), 1.0 / Dh)

        # math O1: the weakly-activated-column-0 fingerprint, BEFORE any grid
        # is fitted. Step 1's undamped 1/H[0,0] amplifies q_arg by ~1/eps for
        # a column whose quant-path magnitude is eps of typical; catch the
        # whole ladder here (the dtype bound only catches fp16 overflow).
        # review-final F7: threshold at 1.5e-2 x median catches the 3-5x
        # grid-widening band (eps<=0.1) the 1e-3 version measurably missed,
        # while the harmless eps=0.3 row (8.5e-2 ratio) stays silent.
        diag_med = torch.median(Dh[~dead]) if (~dead).any() else Dh.new_zeros(())
        if not bool(dead[0]) and float(diag_med) > 0 and \
                float(Dh[0]) < 1.5e-2 * float(diag_med):
            log.warn(
                f"Quantization(Qronos): `{self.name}` column 0 is weakly "
                f"activated (H[0,0]={float(Dh[0]):.3e} vs median diag "
                f"{float(diag_med):.3e}) — step-1 amplification will widen "
                "group 0's grid; check calibration coverage.")

        # ---- STEP 1: closed-form optimal (unquantized) column 0 ------------
        # q_arg = (W_orig . G[:,0]) * Dhi[0] - (W . triu(H,1)[0,:]) * Dhi[0]
        u0 = torch.zeros(columns, dtype=torch.float32, device=H.device)
        u0[1:] = H[0, 1:]
        q_arg = (W_orig.matmul(G[:, 0]) - W.matmul(u0)) * Dhi[0]
        # math M1: bound against the layer dtype's representable range, like
        # the reference (fp16 overflows here are silent in fp32 otherwise).
        if not torch.isfinite(q_arg).all() or \
                q_arg.abs().max() > weight_finfo.max:
            raise ValueError(
                f"Qronos `{self.name}`: step-1 column exceeds the layer "
                f"dtype's range (max |q_arg|={q_arg.abs().max():.3e}, "
                f"dtype max={weight_finfo.max:.3e}) — weakly-activated "
                "column 0 (tiny H[0,0]); inspect calibration coverage.")
        # review-final F7 belt-and-braces: the dtype-independent amplitude
        # fingerprint (fires at eps~0.1 where the H[0,0] ratio check is near
        # its threshold).
        if float(q_arg.abs().max()) > 4.0 * max(float(W_orig[:, 0].abs().max()),
                                                torch.finfo(torch.float32).tiny):
            log.warn(
                f"Quantization(Qronos): `{self.name}` step-1 column amplified "
                f"{float(q_arg.abs().max()):.3e} vs original "
                f"{float(W_orig[:, 0].abs().max()):.3e} — weakly-activated "
                "column 0; group-0 grid will widen.")
        W[:, 0] = q_arg

        # ---- SMW downdate: iH -> inverse over columns 1..IC-1 (R2) ---------
        c00 = iH[0, 0]
        b = iH[1:, 0:1]  # [IC-1, 1]
        iH = iH[1:, 1:] - b.matmul(b.t()) / c00
        del b

        # ---- STEP 2 with two-pass group-0 grid (adversarial H-1) -----------
        # Pass 1: provisional grid on the values available now
        # ([q_arg | original cols 1..g-1]) -> provisional q0 -> step-2 solve.
        # Pass 2: re-fit the grid on the values that will ACTUALLY be
        # quantized on it ([q_arg | step-2 cols 1..g-1]), re-quantize column
        # 0, and delta-update the step-2 solution (exact: W[:,1:] is linear
        # in q0). The stored grid and Q stay packer-consistent.
        scale = []
        zero = []
        g0_hi = min(group_size, columns)
        _find_group_params(W, 0, g0_hi)
        q0 = self.quantizer.quantize(W[:, 0].unsqueeze(1)).flatten()

        Gh_right = G[:, 1:].clone()
        idx = torch.arange(1, columns, device=H.device)
        Gh_right[idx, idx - 1] += damp  # (G + damp*I)[:, 1:]
        del G  # last use of G (int M5)
        h0iH = H[0, 1:].matmul(iH)          # [IC-1], cached for the delta pass
        W[:, 1:] = (
            W_orig.matmul(Gh_right.matmul(iH))
            - torch.outer(q0, h0iH)
        )
        del Gh_right
        del H  # adversarial M-2 / math n4: H's last use was h0iH; Dh/Dhi keep the diag

        # Pass 2+ (only meaningful when group 0 spans more than column 0).
        # math O2: iterate the refit to an actual fixed point — the delta
        # update moves W[:, 1:g] by a rank-1 term, so a grid fitted just
        # before it can be one step stale. Terminates immediately in the
        # common case (q0 unchanged); capped at 3 rounds.
        if g0_hi > 1:
            for _ in range(3):
                _find_group_params(W, 0, g0_hi)
                q0_final = self.quantizer.quantize(W[:, 0].unsqueeze(1)).flatten()
                if torch.equal(q0_final, q0):
                    break
                W[:, 1:] += torch.outer(q0 - q0_final, h0iH)
                q0 = q0_final
        scale.append(self.quantizer.scale.clone())
        zero.append(self.quantizer.zero.clone())
        del h0iH
        del W_orig  # adversarial M-2: dead after step 2

        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)
        Q[:, 0] = q0
        # math m1: column-0 loss on the pre-downdate inverse-diagonal basis
        # (1/c00_pre ~ H-scale d^2). c00_pre is a diagonal entry of the
        # inverse of a PD matrix, hence strictly positive (math O6) — for a
        # dead column 0 it is ~1/damp (large), so no clamp is needed.
        Losses[:, 0] = (W[:, 0] - q0) ** 2 / c00_pre

        # ---- L: upper Cholesky factor of iH (conditioned by c) -------------
        cC = self.qronos_chol_c
        try:
            iH.mul_(cC)
            L = torch.linalg.cholesky(iH, upper=True)
            L = L / math.sqrt(cC)
        except (torch.linalg.LinAlgError, RuntimeError) as e:
            log.warn(
                f"Quantization(Qronos): `{self.name}` Cholesky of iH failed "
                f"({type(e).__name__}); keeping step-1/2 reconstruction and "
                "finishing with per-group RTN (null block correction).")
            L = None
        del iH

        chol_degraded = L is None
        if chol_degraded:
            # Degraded path: finish remaining columns by plain per-group RTN
            # on the step-2 reconstruction (R8: still emit all group scales).
            for j in range(1, columns):
                if j % group_size == 0:
                    _find_group_params(W, j, min(j + group_size, columns))
                    scale.append(self.quantizer.scale.clone())
                    zero.append(self.quantizer.zero.clone())
                q = self.quantizer.quantize(W[:, j].unsqueeze(1)).flatten()
                Q[:, j] = q
        else:
            # ---- BLOCK LOOP over columns 1..IC-1 (R2 index shift) ----------
            # int M2 + math O4: edges are the UNION of group boundaries and
            # blocksize multiples — every group boundary stays an edge (all
            # prior diffusion visible at find_params), while blocksize still
            # caps the block width (matters when group_size is -1/huge: an
            # uncapped block degenerates to one unblocked rank-1 sweep).
            edge_set = {1, columns}
            edge_set.update(range(group_size, columns, group_size))
            step = max(1, int(blocksize))
            if step < group_size:
                # blocksize only caps anything when smaller than a group;
                # otherwise its offset grid just interleaves 1-wide slivers
                # between group boundaries (review-final F4).
                edge_set.update(range(1 + step, columns, step))
            block_edges = sorted(e for e in edge_set if 1 <= e <= columns)
            for bi in range(len(block_edges) - 1):
                i1, i2 = block_edges[bi], block_edges[bi + 1]
                if i1 >= i2:
                    continue
                count = i2 - i1
                Err1 = torch.zeros(
                    (W.shape[0], count), dtype=torch.float32, device=W.device)
                # L block for columns i1..i2-1 lives at L-index i1-1..i2-1.
                Lblk = L[i1 - 1:i2 - 1, i1 - 1:i2 - 1]

                for i in range(count):
                    j = i1 + i  # absolute column
                    if j % group_size == 0:
                        _find_group_params(W, j, min(j + group_size, columns))
                        scale.append(self.quantizer.scale.clone())
                        zero.append(self.quantizer.zero.clone())

                    w = W[:, j]
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q[:, j] = q
                    d = Lblk[i, i]
                    err = (w - q) / d
                    Losses[:, j] = (w - q) ** 2 / d ** 2
                    # in-block diffusion (columns j..i2-1)
                    W[:, j:i2] -= torch.outer(err, Lblk[i, i:])
                    Err1[:, i] = err

                # out-of-block diffusion (columns i2..)
                if i2 < columns:
                    W[:, i2:] -= Err1.matmul(L[i1 - 1:i2 - 1, i2 - 1:])
                del Err1
            del L

        torch_sync()

        # n2: the R1/R8 invariant, asserted for real.
        expected_groups = (columns + group_size - 1) // group_size
        if len(scale) != expected_groups:
            raise RuntimeError(
                f"Qronos `{self.name}`: emitted {len(scale)} scale groups, "
                f"expected {expected_groups}")

        if chol_degraded:
            # math M3 / adversarial L-2: a degraded module must not report a
            # healthy-looking number. Use the processor's string-loss channel.
            mean_abs_err = (Q - W).abs().mean().item()
            avg_loss = f"fallback(chol): {mean_abs_err:.7f}"
        else:
            # int M4: /2 to match the GPTQ/FOEM loss convention.
            avg_loss = torch.sum(Losses).item() / 2 / self.nsamples
            # math M1(ii): reject inf as well as NaN.
            if not math.isfinite(avg_loss):
                raise ValueError(
                    f"Quantization(Qronos): non-finite loss for `{self.name}`")
        del Losses
        duration = time.time() - start

        g_idx = [i // group_size for i in range(columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        else:
            Q = Q.type_as(self.module.weight.data)
        # int M7: hand Q back on the weight's home device.
        Q = Q.to(device=result_device, non_blocking=False)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        # math M1(iii): the visible fingerprint of a blown group-0 grid.
        if scale.shape[1] > 1:
            g0_med = scale[:, 0].abs().median()
            rest_med = scale[:, 1:].abs().median()
            if float(rest_med) > 0 and float(g0_med) > 20.0 * float(rest_med):
                log.warn(
                    f"Quantization(Qronos): `{self.name}` group-0 scale "
                    f"median {float(g0_med):.3e} is >20x the other groups' "
                    f"median {float(rest_med):.3e} — weakly-activated "
                    "column 0 suspected; check calibration coverage.")

        # int m1: report the RELATIVE damp basis (a fraction, comparable with
        # the GPTQ/FOEM damp_percent column); lambda_max was logged already.
        return Q, scale, zero, g_idx, duration, avg_loss, self.qronos_percdamp, self.nsamples


__all__ = ["Qronos"]
