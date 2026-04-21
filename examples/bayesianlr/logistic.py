"""
Compare ALDI, unadjusted PICKLES, PICKLES. 

Step size hardcoded, need to manually control. todo make friendly
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pandas as pd
import sys

sys.path.append("../../src")


from affine_invariant_samplers import (
    sampler_aldi,
    sampler_pickles_unadjusted,
    effective_sample_size,
    sampler_pickles
)

# 


df = pd.read_csv("data/framingham.csv")

print(f"Original shape: {df.shape}")
df = df.dropna()
print(f"Shape after removing missing data: {df.shape}")

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

print(f"Number of observations (n): {len(y)}")
print(f"Number of features (d): {X.shape[1]}")
print("Feature names:", list(X.columns))


scaler = StandardScaler()          # mean=0, std=1
X_scaled = scaler.fit_transform(X)

X = jnp.array(X_scaled)
y = jnp.array(y.to_numpy(dtype='float32'))

# ──────────────────────────────────────────────────────────────────────────────
# Target:  logistic regression
# ──────────────────────────────────────────────────────────────────────────────

def logistic(r):
    return 1 / (1 + jnp.exp(-r))

def logistic_loss_stable(params):
    """Uses log-sum-exp trick — avoids overflow in sigmoid."""
    params_w = params[:,:15]
    params_b = params[:,15:]
    logits = jnp.squeeze(X[None,:,:] @ params_w[:,:,None], axis=2) + params_b # (N, df_len)
    # log(1 + exp(-logit)) for y=1, log(1 + exp(logit)) for y=0
    loss = jnp.logaddexp(0, jnp.where(y == 1, -logits, logits))

    regularization = jnp.sum(params**2, axis=1)
    return -jnp.mean(loss, axis=1) - 0.01 * regularization


# def make_gaussian(dim=20, kappa=1000.0, seed=0):
#     """Return (log_prob_batched, cov, prec)."""
#     eigvals = jnp.logspace(0, jnp.log10(kappa), dim)
#     Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (dim, dim)))
#     cov  = Q @ jnp.diag(eigvals) @ Q.T
#     prec = Q @ jnp.diag(1.0 / eigvals) @ Q.T

#     def log_prob(x):                          # (batch, D) -> (batch,)
#         return -0.5 * jnp.sum((x @ prec) * x, axis=-1)

#     return log_prob, cov, prec


# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Report helper
# ──────────────────────────────────────────────────────────────────────────────

def _report(name, samples, info, elapsed):
    flat    = jnp.asarray(samples).reshape(-1, samples.shape[-1])
    me, ve  = float(jnp.mean(flat)), float(jnp.mean(jnp.var(flat, axis=0)))
    ess     = effective_sample_size(samples)
    grads   = info.get("n_grad_evals")
    grads_s = f"{grads:>10d}" if grads is not None else f"{'–':>10s}"

    ss = info.get("step_size")
    if ss is None:
        ss = info.get("final_step_size")
    
    accept = info.get('acceptance_rate')
    print(f"  {name:<24s}  x_e mean={me:5.5f} var={ve:5.5f}   "
          f"min_ESS={float(ess.min()):7.1f}   grad_evals={grads_s}   ss={ss}"
          f"time={elapsed:5.1f}s")
    if accept is not None:
        print(f"accept={accept}")


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim      = 16
    n_chains = 100
    n_samp   = 100000
    warmup   = 5000
    seed     = 123

    log_prob = logistic_loss_stable
    init = jax.random.normal(jax.random.key(42), (n_chains, dim))

    print("=" * 110)
    print("Finish data input")
    print("=" * 110)

    

    results = {}

    # aldi: step_size=0.1 is a reasonable compromise.  Too small and x_even
    # barely migrates off the origin; too large and x_odd becomes unstable.
    # Even so, aldi is expected to show visible bias on Rosenbrock.
    t0 = time.time()
    print(f"Unadjusted Langevin on LR  "
          f"n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 110)
    s, info = sampler_aldi(log_prob, init, n_samp, warmup=warmup,
                            step_size=4.0, seed=seed, verbose=False)
    _report("aldi",               s, info, time.time() - t0)
    results["aldi"] = s

    # pickles_unadjusted: its PGN adaptation rescales h by the ensemble
    # gradient norm, so a base step of 0.2 is safe and much larger than
    # what aldi can tolerate.  Expect nearly-correct moments.
    print(f"pickles  "
          f"n_chains={n_chains}  n_samp={n_samp}  warmup={warmup}")
    print("=" * 110)
    t0 = time.time()
    s, info = sampler_pickles_unadjusted(
        log_prob, init, n_samp, warmup=warmup,
        step_size=4.0, gamma=2.0, seed=seed, verbose=False)
    _report("pickles_unadjusted", s, info, time.time() - t0)
    results["pickles_unadjusted"] = s


    t0 = time.time()
    s, info = sampler_pickles(log_prob, init, n_samp, warmup=warmup,
                               step_size=4.0, gamma=2.0, seed=seed, verbose=False, adapt_L=False)
    _report("pickles", s, info, time.time() - t0)
    results["pickles"] = s
    print("=" * 110)


    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    # 2D marginal of (x_0, x_1) with exact contours
    # xr = np.linspace(-2.5, 2.5, 400)
    # yr = np.linspace(-1.5, 6.0, 400)
    # gx, gy = np.meshgrid(xr, yr)
    # true_density = np.exp(-(b * (gy - gx ** 2) ** 2 + (gx - a) ** 2))

    # fig_c, axes_c = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.2),
    #                               sharex=True, sharey=True)
    # for ax, (name, s) in zip(axes_c, results.items()):
    #     flat = np.asarray(s).reshape(-1, dim)
    #     ax.hist2d(flat[:, 0], flat[:, 1], bins=80,
    #               range=[[xr[0], xr[-1]], [yr[0], yr[-1]]],
    #               cmap="Blues", density=True)
    #     ax.contour(gx, gy, true_density, levels=8, colors="k",
    #                linewidths=0.8, alpha=0.7)
    #     ax.set_title(name, fontsize=11)
    #     ax.set_xlabel("x₀ (even)")
    # axes_c[0].set_ylabel("x₁ (odd)")
    # fig_c.suptitle("Unadjusted Langevin, Rosenbrock (x₀, x₁) marginal  "
    #                "(black = true contours)",
    #                y=0.99)
    # fig_c.tight_layout()

    # # Per-method corner plots on the first 4 dims with full truth overlays
    # # (see example_rosenbrock.py for the derivation).
    # K = 4
    # labels = [f"x{i}" + (" (e)" if i % 2 == 0 else " (o)") for i in range(K)]
    # truths = [a if i % 2 == 0 else a ** 2 + 0.5 for i in range(K)]

    # xe_grid = np.linspace(-2.0, 4.0, 300)
    # xe_pdf  = np.exp(-(xe_grid - a) ** 2) / np.sqrt(np.pi)

    # xo_grid = np.linspace(-1.0, 6.0, 300)
    # xe_q = np.linspace(-3.0, 5.0, 600)
    # p_xe = np.exp(-(xe_q - a) ** 2) / np.sqrt(np.pi)
    # dxe  = xe_q[1] - xe_q[0]
    # sqrt_b_over_pi = np.sqrt(b / np.pi)
    # xo_pdf = np.zeros_like(xo_grid)
    # for xe_k, pk in zip(xe_q, p_xe):
    #     xo_pdf += pk * sqrt_b_over_pi * np.exp(-b * (xo_grid - xe_k ** 2) ** 2)
    # xo_pdf *= dxe

    # def grid_and_pdf_1d(i):
    #     return (xe_grid, xe_pdf) if i % 2 == 0 else (xo_grid, xo_pdf)

    # truth_1d_r = {i: grid_and_pdf_1d(i) for i in range(K)}

    # truth_2d_r = {}
    # for i in range(K):
    #     for j in range(i):
    #         xg, pxg = grid_and_pdf_1d(j)
    #         yg, pyg = grid_and_pdf_1d(i)
    #         same_pair = (i // 2 == j // 2) and (i % 2 != j % 2)
    #         if same_pair:
    #             Xg, Yg = np.meshgrid(xg, yg)
    #             pdf_2d = np.exp(-(b * (Yg - Xg ** 2) ** 2 + (Xg - a) ** 2))
    #         else:
    #             pdf_2d = np.outer(pyg, pxg)
    #         truth_2d_r[(i, j)] = (xg, yg, pdf_2d)

    # for name, s in results.items():
    #     s_sub = np.asarray(s).reshape(-1, dim)[:, :K]
    #     fig = corner_plot(s_sub, labels=labels, truths=truths,
    #                       truth_1d=truth_1d_r, truth_2d=truth_2d_r,
    #                       title=f"{name} — first {K} dims")

    # plt.show()
