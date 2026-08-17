# Tests for the Qronos → GPTQ-Pro port (qronos_gptqmodel.Qronos),
# revision 3.1 (post integration + math + adversarial reviews: faithful
# dead-column handling, iterated two-pass group-0 grid, group-boundary ∪
# blocksize block edges, stock fallbacks, deterministic damp,
# GAR/tp-pad/multi-device raises).
#
# Run:  cd /home/user/target-quant/qronos && .venv/bin/python -m pytest tests/ -x -q
#
# The load-bearing test is test_matches_reference_transcription: an
# INDEPENDENT transcription of reference/qronos.py's exact op order
# (single_layer_update, groups=1 linear case, act_order off) is run on the
# same W/H/G with the same quantization grid, and the port's output must
# match column-for-column. The transcription is written from the reference
# file, not from qronos_gptqmodel.py — reviewers should diff it against
# reference/qronos.py, never against the port. Two deliberate convention
# deviations are mirrored in the transcription because they are grid/partition
# CHOICES, not algebra (NOTES.md DEVIATION-6/6b): group-0 grid from W_orig,
# and group-aligned block partition.

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root: gptqmodel package + qronos_gptqmodel.py
sys.path.insert(0, _ROOT)

from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402

from qronos_gptqmodel import Qronos, _power_iteration  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------- utils

def make_solver(oc=8, ic=16, group_size=8, bits=4, sym=True, mse=0.0,
                desc_act=False, static_groups=False, act_group_aware=False,
                weight=None, module=None):
    if module is None:
        module = nn.Linear(ic, oc, bias=False)
        if weight is not None:
            with torch.no_grad():
                module.weight.copy_(weight)
    named = NamedModule(module, name="test_lin", full_name="model.test_lin", layer_index=0)
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    qcfg = QuantizeConfig(bits=bits, group_size=group_size, sym=sym,
                          desc_act=desc_act, static_groups=static_groups,
                          act_group_aware=act_group_aware, mse=mse)
    solver = Qronos(named, qcfg=qcfg)
    solver.quantizer.configure(perchannel=True)
    return solver, module, named


def feed(solver, named, x_float_batches, x_quant_batches):
    """Push (float, quant) activation batch pairs the way the looper does:
    NativeProcessor caches the float inputs first; add_batch gets the quant
    inputs."""
    for xf in x_float_batches:
        solver.native_inps.append(xf)
    for xq in x_quant_batches:
        solver.add_batch(xq, out=None)


class GridTracker:
    """Independent group-grid bookkeeping for the reference transcription.

    Uses the SAME Quantizer object as the port (grid parity by construction)
    with the port's rev-3 conventions: group 0 via the TWO-PASS refit (done
    inline in reference_qronos, since it interacts with step 2); groups >= 1
    found when the sequential processing first reaches their boundary, on
    the current W.
    """

    def __init__(self, quantizer, group_size, columns):
        self.quantizer = quantizer
        self.group_size = group_size
        self.columns = columns
        self.found = 0  # group 0 handled by the caller's two-pass

    def quantize_col(self, W, j):
        g = j // self.group_size
        if g > self.found:
            lo = g * self.group_size
            hi = min(lo + self.group_size, self.columns)
            self.quantizer.find_params(W[:, lo:hi], weight=True)
            self.found = g
        return self.quantizer.quantize(W[:, j].unsqueeze(1)).flatten()


def _oracle_power_iteration(mat, iters, generator):
    """Oracle-OWN transcription of the reference's power iteration
    (magr.py: torch.rand start, iterate, final Rayleigh quotient) so the
    differential does not share `_power_iteration` with the port (math O3).
    Must mirror the PORT's iteration order exactly (rand start, lam from the
    pre-normalized v each step) — verified equal within fp tolerance by
    test_power_iteration_oracle_agrees."""
    n = mat.shape[-1]
    v = torch.rand(n, dtype=torch.float32, device=mat.device, generator=generator)
    nrm = torch.linalg.vector_norm(v)
    if nrm == 0 or not torch.isfinite(nrm):
        v = torch.ones(n, dtype=torch.float32, device=mat.device)
        nrm = torch.linalg.vector_norm(v)
    v = v / nrm
    lam = torch.zeros((), dtype=torch.float32)
    for _ in range(iters):
        w = mat @ v
        wn = torch.linalg.vector_norm(w)
        if wn == 0 or not torch.isfinite(wn):
            return torch.zeros((), dtype=torch.float32)
        lam = torch.dot(v, w)
        v = w / wn
    return lam


def solver_damp(H, ic, name="test_lin", percdamp=1e-5):
    """Reproduce the solver's deterministic damp so the oracle and the port
    agree without sharing RNG call-order (math m4 / M2.3)."""
    gen = torch.Generator(device=H.device)
    gen.manual_seed((ic * 1000003 + len(name)) & 0x7FFFFFFF)
    lam = _oracle_power_iteration(H, 30, generator=gen)
    base = float(lam)
    if not math.isfinite(base) or base <= 0:
        diag_scale = float(torch.diag(H).abs().max())
        base = max(float(torch.mean(torch.diag(H)).clamp(min=0)),
                   1e-8 * max(diag_scale, 1.0))
    return percdamp * base


def reference_qronos(W_orig_in, H_in, G_in, quantizer, group_size,
                     damp=None, chol_c=1e4):
    """Transcription of reference/qronos.py single_layer_update for the
    groups==1 Linear, act_order=False case. Keeps the reference's exact op
    order and indexing (incl. the SMW downdate, the -1 shifts, and the
    FAITHFUL dead-column handling: only W's dead columns zeroed, per
    qronos.py:143-144). The two port grid conventions (two-pass group-0
    refit, group-aligned blocks) are mirrored deliberately — they are
    choices, not algebra; the differential covers the algebra."""
    W = W_orig_in.clone().to(torch.float32)
    W_orig = W_orig_in.clone().to(torch.float32)
    H = H_in.clone().to(torch.float32)
    G = G_in.clone().to(torch.float32)
    columns = W.shape[1]
    tracker = GridTracker(quantizer, group_size, columns)

    # dead columns (reference: zeroes ONLY the working weight copy)
    dead = H.diag() == 0
    W[:, dead] = 0

    Dh = H.diag().clone()
    Dhi = torch.where(Dh != 0, 1.0 / Dh, torch.zeros_like(Dh))
    Uh = torch.triu(H, 1)

    if damp is None:
        damp = solver_damp(H, columns)
    iH = H.clone()
    iH.diagonal().add_(damp)
    iH = torch.cholesky_inverse(torch.linalg.cholesky(iH))

    # Step 1 (reference: q_arg = Gw - Uv, both scaled by Dhi[0])
    Gw = W_orig.matmul(G[:, 0] * Dhi[0])
    Uv = W.matmul(Uh[0, :] * Dhi[0])
    W[:, 0] = Gw - Uv

    # SMW downdate (reference: A -= b b^T / c)
    c = iH[0, 0]
    b = iH[1:, [0]]
    iH = iH[1:, 1:] - b.matmul(b.t()) / c

    # Step 2 with the two-pass group-0 grid (port rev 3)
    g0_hi = min(group_size, columns)
    quantizer.find_params(W[:, 0:g0_hi], weight=True)  # provisional
    q0 = quantizer.quantize(W[:, 0].unsqueeze(1)).flatten()
    Ih = torch.diag(torch.full((columns,), damp))
    Gh = G + Ih
    h0iH = H[0, 1:] @ iH
    W[:, 1:] = W_orig.matmul(Gh[:, 1:] @ iH) - torch.outer(q0, h0iH)
    if g0_hi > 1:
        # iterated refit to a fixed point, mirroring the port (math O2)
        for _ in range(3):
            quantizer.find_params(W[:, 0:g0_hi], weight=True)
            q0_final = quantizer.quantize(W[:, 0].unsqueeze(1)).flatten()
            if torch.equal(q0_final, q0):
                break
            W[:, 1:] += torch.outer(q0 - q0_final, h0iH)
            q0 = q0_final

    # Cholesky of iH (conditioned)
    L = torch.linalg.cholesky(iH * chol_c, upper=True) / math.sqrt(chol_c)

    Q = torch.zeros_like(W)
    Q[:, 0] = q0

    # Block loop — group-aligned partition (first block [1, g))
    block_edges = [1] + list(range(group_size, columns, group_size)) + [columns]
    for bi in range(len(block_edges) - 1):
        i1, i2 = block_edges[bi], block_edges[bi + 1]
        if i1 >= i2:
            continue
        count = i2 - i1
        err_block = torch.zeros((W.shape[0], count), dtype=torch.float32)
        h_inv_block = L[i1 - 1:i2 - 1, i1 - 1:i2 - 1]
        for i in range(count):
            j = i1 + i
            q = tracker.quantize_col(W, j)
            Q[:, j] = q
            d = h_inv_block[i, i]
            err = (W[:, j] - q) / d
            err_block[:, i] = err
            W[:, j:i2] -= err.unsqueeze(1).matmul(h_inv_block[i, i:].unsqueeze(0))
        W[:, i2:] -= err_block.matmul(L[i1 - 1:i2 - 1, i2 - 1:])
    return Q


def synth_data(oc, ic, n_batches=4, tokens=32, noise=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    xf, xq = [], []
    for _ in range(n_batches):
        x = torch.randn(tokens, ic, generator=g)
        xf.append(x)
        xq.append(x + noise * torch.randn(tokens, ic, generator=g))
    return xf, xq


# --------------------------------------------------------------------- tests

def test_stats_orientation():
    """H must be built from quant-path inputs, G = float x quant^T (R4)."""
    oc, ic = 4, 6
    solver, lin, named = make_solver(oc=oc, ic=ic, group_size=-1)
    xf = [torch.randn(8, ic)]
    xq = [torch.randn(8, ic)]
    feed(solver, named, xf, xq)

    # nsamples counts TOKEN ROWS (post-reshape), the same unit as stock
    # GPTQ's batch_token_size — required so the rank-starvation fallback
    # gate compares like units against expected_nsamples
    # (total_calibration_tokens). One 2D batch of 8 rows => nsamples == 8
    # => scale == 2/8. The shared H/G scale cancels in the update math
    # (NOTES.md R6); this test's real assertion is the orientation.
    scale = 2.0 / 8
    H_expect = scale * xq[0].t().matmul(xq[0])
    G_expect = scale * xf[0].t().matmul(xq[0])
    assert torch.allclose(solver.H, H_expect, atol=1e-5), "H must be x_hat x_hat^T"
    assert torch.allclose(solver.G, G_expect, atol=1e-5), \
        "G must be x x_hat^T (float rows, quant cols) — reference qronos.py:88"
    assert not torch.allclose(solver.G, scale * xq[0].t().matmul(xf[0]), atol=1e-5), \
        "G orientation flipped (gpfq.py convention instead of qronos.py)"


def test_matches_reference_transcription():
    """Port output == independent transcription of the reference, same grid."""
    torch.manual_seed(7)
    oc, ic, gsz = 8, 16, 8
    W0 = torch.randn(oc, ic)
    solver, lin, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic)
    feed(solver, named, xf, xq)

    H = solver.H.clone()
    G = solver.G.clone()

    torch.manual_seed(123)  # power iteration uses randn: fix seed for both runs
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = solver.quantize()

    ref_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    torch.manual_seed(123)
    Q_ref = reference_qronos(W0.to(torch.float32), H, G, ref_solver.quantizer,
                             group_size=gsz)

    assert torch.allclose(Q.to(torch.float32), Q_ref, atol=1e-4, rtol=1e-4), \
        f"port vs reference transcription max diff: {(Q.float() - Q_ref).abs().max()}"
    assert g_idx.tolist() == [i // gsz for i in range(ic)]
    assert scale.shape[1] == math.ceil(ic / gsz)
    assert math.isfinite(avg_loss)
    assert damp == pytest.approx(1e-5)  # m1: relative basis reported


def test_matches_reference_multiblock_multigroup():
    """Same, at sizes exercising several groups and a ragged tail group."""
    torch.manual_seed(11)
    oc, ic, gsz = 16, 40, 8  # 5 groups
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic, n_batches=6, tokens=64, seed=3)
    feed(solver, named, xf, xq)
    H, G = solver.H.clone(), solver.G.clone()

    torch.manual_seed(321)
    Q, scale, zero, g_idx, *_ = solver.quantize()

    ref_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    torch.manual_seed(321)
    Q_ref = reference_qronos(W0.float(), H, G, ref_solver.quantizer,
                             group_size=gsz)
    assert torch.allclose(Q.float(), Q_ref, atol=1e-4, rtol=1e-4), \
        f"max diff {(Q.float() - Q_ref).abs().max()}"
    assert scale.shape[1] == 5


def test_identical_paths_not_worse_than_rtn():
    """With x == x_hat (G == H): Qronos must not be worse than plain RTN on
    the calibration objective ||W X - Q X||_F."""
    torch.manual_seed(5)
    oc, ic, gsz = 8, 16, 8
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, _ = synth_data(oc, ic, noise=0.0, seed=9)
    feed(solver, named, xf, xf)  # float == quant

    X = torch.cat([b for b in xf], dim=0).t()  # [ic, tokens]
    Q, *_ = solver.quantize()
    err_qronos = (W0.float().matmul(X) - Q.float().matmul(X)).norm()

    # plain RTN on the same grid conventions
    rtn_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    Wf = W0.float()
    Q_rtn = torch.zeros_like(Wf)
    for g0 in range(0, ic, gsz):
        g1 = min(g0 + gsz, ic)
        rtn_solver.quantizer.find_params(Wf[:, g0:g1], weight=True)
        for j in range(g0, g1):
            Q_rtn[:, j] = rtn_solver.quantizer.quantize(
                Wf[:, j].unsqueeze(1)).flatten()
    err_rtn = (Wf.matmul(X) - Q_rtn.matmul(X)).norm()

    assert err_qronos <= err_rtn * 1.05, \
        f"Qronos ({err_qronos:.4f}) worse than RTN ({err_rtn:.4f}) on its own objective"


def test_dead_columns():
    torch.manual_seed(2)
    oc, ic, gsz = 4, 8, 4
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic, seed=4)
    for t in (*xf, *xq):
        t[:, 3] = 0.0  # column 3 never activates
    feed(solver, named, xf, xq)
    Q, scale, zero, g_idx, *_ = solver.quantize()
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert scale.shape[1] == 2


def test_group_scale_count_and_gidx():
    oc, ic, gsz = 4, 24, 8
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz)
    xf, xq = synth_data(oc, ic, seed=6)
    feed(solver, named, xf, xq)
    Q, scale, zero, g_idx, *_ = solver.quantize()
    assert scale.shape[1] == 3, "one scale set per group (R1/R8)"
    assert g_idx.tolist() == [i // gsz for i in range(ic)]


def test_desc_act_raises():
    solver, _, named = make_solver(desc_act=True)
    xf, xq = synth_data(8, 16)
    feed(solver, named, xf, xq)
    with pytest.raises(NotImplementedError):
        solver.quantize()


def test_static_groups_raises():
    solver, _, named = make_solver(static_groups=True)
    xf, xq = synth_data(8, 16)
    feed(solver, named, xf, xq)
    with pytest.raises(NotImplementedError):
        solver.quantize()


def test_act_group_aware_raises():
    """C1: GAR is not implemented — it must refuse, never silently skip."""
    solver, _, named = make_solver(act_group_aware=True)
    xf, xq = synth_data(8, 16)
    feed(solver, named, xf, xq)
    with pytest.raises(NotImplementedError):
        solver.quantize()


def test_conv2d_rejected_at_construction():
    """m5: unsupported module types fail at preprocess, not mid-calibration."""
    conv = nn.Conv2d(3, 4, 3)
    with pytest.raises(NotImplementedError):
        make_solver(module=conv, group_size=-1)


def test_mismatched_native_cache_raises():
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    solver.native_inps.append(torch.randn(16, 8))
    with pytest.raises(ValueError):
        solver.add_batch(torch.randn(8, 8), out=None)  # 8 tokens vs 16 cached


def test_zero_batches_takes_stock_fallback():
    """C3: a module that saw no calibration traffic must not crash — it takes
    the configured stock fallback and still returns a full valid tuple."""
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = solver.quantize()
    assert torch.isfinite(Q.float()).all()
    assert isinstance(avg_loss, str) and avg_loss.startswith("fallback(")
    assert scale.shape[1] == 2
    assert nsamples == 0


def test_singular_H_falls_back_to_stock(monkeypatch):
    """Total inversion failure -> stock fallback with string loss (M3)."""
    torch.manual_seed(3)
    oc, ic, gsz = 4, 8, 4
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic, seed=8)
    feed(solver, named, xf, xq)
    monkeypatch.setattr(solver, "_qronos_inverse", lambda H: (None, None))
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = solver.quantize()
    assert torch.isfinite(Q.float()).all()
    assert isinstance(avg_loss, str) and avg_loss.startswith("fallback("), \
        "fallback must be labeled, not reported as a perfect 0.0 loss (M3)"
    assert scale.shape[1] == 2, "fallback must emit scales for ALL groups (R8)"


def test_nan_H_raises():
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    xf, xq = synth_data(4, 8, seed=10)
    feed(solver, named, xf, xq)
    solver.H[0, 0] = float("nan")
    with pytest.raises(ValueError):
        solver.quantize()


def test_activation_weighted_mse_runs():
    """M9: importance-weighted scale search path executes without error and
    produces a full artifact (uses mse>0 so find_params' search is live)."""
    torch.manual_seed(4)
    oc, ic, gsz = 8, 16, 8
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, mse=2.0)
    solver.qcfg.activation_weighted_mse = True
    xf, xq = synth_data(oc, ic, seed=13)
    feed(solver, named, xf, xq)
    Q, scale, zero, g_idx, *_ = solver.quantize()
    assert torch.isfinite(Q.float()).all()
    assert scale.shape[1] == 2


def test_free_clears_buffers():
    """m2: free() releases G and any unconsumed native activations."""
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    xf, xq = synth_data(4, 8, seed=14)
    feed(solver, named, xf, xq)
    solver.native_inps.append(torch.randn(32, 8))  # simulate leftover
    solver.free()
    assert solver.native_inps is None


def test_to_device_moves_G():
    """M6: the rehome hook moves H and G together (CPU->CPU no-op form)."""
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    xf, xq = synth_data(4, 8, seed=15)
    feed(solver, named, xf, xq)
    solver.to_device(torch.device("cpu"))
    assert solver.H.device.type == "cpu" and solver.G.device.type == "cpu"


def test_power_iteration_matches_eigh():
    torch.manual_seed(12)
    A = torch.randn(32, 32)
    H = A @ A.t()
    lam = _power_iteration(H, 60)
    lam_true = torch.linalg.eigvalsh(H)[-1]
    # Purpose-appropriate tolerance: lam only sets damp = percdamp * lam, so
    # a ~1% estimate error changes damp by ~1% — immaterial. Small eigengaps
    # legitimately converge slowly.
    assert abs(float(lam) - float(lam_true)) / float(lam_true) < 0.02


def test_dead_column_differential_quant_path_only():
    """math C1: a channel alive in the FLOAT path but dead in the QUANT path
    must be handled exactly like the reference (its G row carries real
    signal that surviving columns use to compensate)."""
    torch.manual_seed(21)
    oc, ic, gsz, DEAD = 8, 16, 8, 5
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic, seed=22)
    for b in xq:
        b[:, DEAD] = 0.0  # dead on the quant path ONLY
    feed(solver, named, xf, xq)
    H, G = solver.H.clone(), solver.G.clone()
    assert float(G[DEAD, :].abs().max()) > 0, "probe needs live float-path signal"

    Q, *_ = solver.quantize()
    ref_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    Q_ref = reference_qronos(W0.float(), H, G, ref_solver.quantizer, group_size=gsz)
    assert torch.allclose(Q.float(), Q_ref, atol=1e-4, rtol=1e-4), \
        f"dead-column (quant-only) divergence: {(Q.float() - Q_ref).abs().max()}"


def test_dead_column_differential_both_paths():
    """math C1(c): a channel dead in BOTH paths — the reference restores
    W_orig for it in step 2; the port must match."""
    torch.manual_seed(23)
    oc, ic, gsz, DEAD = 8, 16, 8, 3
    W0 = torch.randn(oc, ic)
    solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    xf, xq = synth_data(oc, ic, seed=24)
    for b in (*xf, *xq):
        b[:, DEAD] = 0.0
    feed(solver, named, xf, xq)
    H, G = solver.H.clone(), solver.G.clone()

    Q, *_ = solver.quantize()
    ref_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
    Q_ref = reference_qronos(W0.float(), H, G, ref_solver.quantizer, group_size=gsz)
    assert torch.allclose(Q.float(), Q_ref, atol=1e-4, rtol=1e-4), \
        f"dead-column (both-paths) divergence: {(Q.float() - Q_ref).abs().max()}"


def test_empty_native_cache_named_error():
    """adversarial L-3: an exhausted float-path cache raises a NAMED error."""
    solver, _, named = make_solver(oc=4, ic=8, group_size=4)
    with pytest.raises(ValueError, match="cache exhausted"):
        solver.add_batch(torch.randn(8, 8), out=None)


def test_randomized_robustness_sweep():
    """adversarial H-2b: exact bit-comparison on a chaotic quantized output is
    a coin flip across configs (round-half ties), so this sweep asserts the
    STABLE quantities instead — over 30 random configs: the port completes,
    the artifact is well-formed, and its calibration objective is within 2%
    of the transcription's (level flips on ties move single columns, not the
    objective)."""
    rng = torch.Generator().manual_seed(777)
    worse = []
    for trial in range(30):
        oc = int(torch.randint(2, 12, (1,), generator=rng))
        ic = int(torch.randint(4, 40, (1,), generator=rng))
        gsz_pool = [g for g in (2, 4, 8, 16, -1) if g == -1 or g <= ic]
        gsz = gsz_pool[int(torch.randint(0, len(gsz_pool), (1,), generator=rng))]
        noise = float(torch.rand(1, generator=rng)) * 2.0
        W0 = torch.randn(oc, ic, generator=rng)
        solver, _, named = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
        xf = [torch.randn(24, ic, generator=rng) for _ in range(3)]
        xq = [a + noise * torch.randn(24, ic, generator=rng) for a in xf]
        feed(solver, named, xf, xq)
        H, G = solver.H.clone(), solver.G.clone()
        Q, scale, zero, g_idx, duration, avg_loss, damp, ns = solver.quantize()
        eff_g = gsz if gsz != -1 else ic
        assert torch.isfinite(Q.float()).all()
        assert scale.shape[1] == math.ceil(ic / eff_g)
        assert int(g_idx.max()) < scale.shape[1]

        ref_solver, _, _ = make_solver(oc=oc, ic=ic, group_size=gsz, weight=W0)
        Q_ref = reference_qronos(W0.float(), H, G, ref_solver.quantizer,
                                 group_size=eff_g)
        Xq = torch.cat(xq, dim=0).t()
        Xf = torch.cat(xf, dim=0).t()
        obj_port = (W0.float().matmul(Xf) - Q.float().matmul(Xq)).norm()
        obj_ref = (W0.float().matmul(Xf) - Q_ref.matmul(Xq)).norm()
        rel = float((obj_port - obj_ref) / obj_ref.clamp(min=1e-9))
        if rel > 0.02:
            worse.append((trial, oc, ic, gsz, round(noise, 2), round(rel, 4)))
    assert not worse, f"port objective >2% worse than transcription on: {worse}"


def test_power_iteration_oracle_agrees():
    """math O3: the oracle's own transcription must agree with the port's
    _power_iteration given the same seed (guards against either side
    drifting silently)."""
    torch.manual_seed(31)
    A = torch.randn(24, 24)
    H = A @ A.t()
    g1 = torch.Generator(); g1.manual_seed(12345)
    g2 = torch.Generator(); g2.manual_seed(12345)
    lam_port = _power_iteration(H, 30, generator=g1)
    lam_oracle = _oracle_power_iteration(H, 30, generator=g2)
    assert torch.allclose(lam_port, lam_oracle, rtol=1e-6)
