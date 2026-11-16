# Lunar & Astronomical Gen-7 PCLR Deconvolution — Lean 4 Formalization & Production FITS Suite

This repository contains:
1. **`lunar_pclr.py` — Standalone Production Python Library & FITS CLI** implementing our best **Generation-7 Lunar & Astronomical PCLR Deconvolution** algorithm:
   - **100% Blind Single-Frame Deconvolution (`deconvolve_lunar_blind_pclr`)**: Kernel-scaled, noise-debiased Slanted-Edge Cast-Shadow ESF/LSF Tomography (`Theorem 15p`) + 2D Anisotropic Salient Crater-Rim Split-EM PSF estimation (`Theorems 11b, 12e`) + Rényi-Escort Preconditioned Anisotropic Coherence TGV-Bregman PCLR (`Theorems 13a–14c`) + Law-of-Total-Variance 2D Anisotropic Spectral EM (`Theorem 15m`) + Parseval-SURE Two-Band Spatial-Spectral Mixer (`Theorem 15o`).
   - **Non-Blind Single-Frame Deconvolution (`deconvolve_lunar_pclr`)**: Surpasses both the **Omniscient 1D Radial Spectral Wiener Oracle** and the **Omniscient 2D Anisotropic Spectral Wiener Oracle** across every lunar ground-truth scene.
   - **Joint Multi-Frame Blind "Lucky Imaging" Fusion (`deconvolve_lunar_multiframe_blind_pclr`)**: Automatically estimates per-frame seeing PSFs & Strehl sharpness weights $\gamma_f$ across short-exposure bursts (`Theorems 9a–9e, 15g`).
   - **Real Ground-Based Telescope Mode (`deconvolve_lunar_telescope`, `--telescope`)**: Unclipped physical ADU deconvolution (`Theorem 13g`), 16-bit ADC Censored-Saturation Split-EM (`Theorem 14a`), and 2D Anisotropic Elliptic Matérn Tri-Resolvent seeing PSFs (`Theorem 13f`, zero Bessel ghost stripes).
2. **A complete, zero-`sorry`, zero-axiom formal verification in Lean 4** (`Lucky/*.lean`) of:
   - **Classic Lucy-Richardson (1972/1974)** (`Lucky/Basic.lean`, `Lucky/Divergence.lean`),
   - **Gen-1–Gen-3 Self-Calibrating Symmetric TGV-Bregman PCLR** (`Lucky/Improved.lean`, `Lucky/BregmanTGV.lean`),
   - **Joint Multi-Frame "Lucky Imaging" PCLR** (`Lucky/Multiframe.lean`),
   - **Single-Frame & Multi-Frame Blind Deconvolution** (`Lucky/Blind.lean`),
   - **Gen-4–Gen-7 Lunar Preconditioned Compact Adjoint, Rényi-Escort Adjoint, Censored-Saturation, & Anisotropic Hessian Split-EM** (`Lucky/Lunar.lean`, Theorems 13a–13g & 14a–14c), and
   - **Fundamental Information-Theoretic, Spectral Wiener-Oracle, Cramér-Rao, Semi-Convergence, Law-of-Total-Variance Spectral EM, & Parseval-SURE Limits of Deconvolution** (`Lucky/Limits.lean`, Theorems 15a–15p) for arbitrary $m \times n$ imaging systems.
3. **An executable 2D image, blind, lunar, & theoretical-limits deconvolution engine in Lean 4** (`Lucky/Deconvolve.lean`, `Main.lean`, runnable via `lake exe lucky`), plus benchmark suites (`recover_lunar.py`, `recover_image.py`).

---

## Using the Best Algorithm (`lunar_pclr.py`)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Command-Line Interface (FITS)
```bash
# 1. 100% Blind Gen-7 deconvolution of a lunar FITS image (also saves estimated 2D PSF):
python3 lunar_pclr.py input_moon.fits -o moon_sharp.fits --save-psf psf_est.fits

# 2. Real ground-based telescope capture (unclipped ADU + 16-bit saturation gate + Matérn resolvent):
python3 lunar_pclr.py gassendi_raw.fit -o gassendi_deconv.fits --telescope --boost 1.45

# 3. Non-blind Gen-7 deconvolution with a measured PSF FITS:
python3 lunar_pclr.py blurred.fits -o sharp.fits --psf measured_psf.fits

# 4. Joint Multi-Frame Blind "Lucky Imaging" fusion across multiple short-exposure stacks:
python3 lunar_pclr.py stack_p2.fit stack_p4.fit stack_p6.fit stack_p9.fit -o lucky_fused.fits
```

### Python API
```python
from astropy.io import fits
from lunar_pclr import (
    deconvolve_lunar_blind_pclr,
    deconvolve_lunar_pclr,
    deconvolve_lunar_multiframe_blind_pclr,
    deconvolve_lunar_telescope,
)

# Real telescope FITS (preserves physical ADU range and handles 16-bit ADC clipping):
raw = fits.getdata("gassendi.fit")
u_sharp_adu, psf_fwd, info = deconvolve_lunar_telescope(raw, boost=1.45)

# Normalized [0, 1] 100% blind Gen-7 Parseval-SURE PCLR:
u_est, psf_est, fwhm_px = deconvolve_lunar_blind_pclr(y_norm, ksize=23, iters=60)
```

### Lean 4 Formal Verification & Benchmark Reproduction
```bash
# Verify all Lean 4 formal proofs (zero sorry, zero axioms) and build native executable
lake build

# Run the native Lean 4 2D Non-Blind, Lunar Gen-7, Blind, & Theoretical Limits benchmark
lake exe lucky

# Reproduce the full Python 2D & Lunar benchmark suites
python3 recover_image.py
python3 recover_lunar.py
```

---

## Summary of Formalized Theorems in Lean 4

| Module | Theorem / Definition | Mathematical Statement Verified |
|---|---|---|
| `Lucky/VecMat.lean` | `sumRange_comm` | Discrete Fubini's theorem: $\sum_{i=0}^{m-1} \sum_{j=0}^{n-1} f_{i,j} = \sum_{j=0}^{n-1} \sum_{i=0}^{m-1} f_{i,j}$ |
| `Lucky/VecMat.lean` | `adjoint_inner_product` | Adjoint duality: $\langle u, H^T r \rangle_n = \langle r, H u \rangle_m$ |
| `Lucky/VecMat.lean` | `forward_flux_identity` | $\sum_{i=0}^{m-1} (H u)_i = \sum_{j=0}^{n-1} w_j u_j$ where $w_j = \sum_{i=0}^{m-1} H_{i,j}$ |
| `Lucky/Basic.lean` | `rl_flux_conservation` / `rl_unit_flux_conservation` | Exact 1-step & trajectory-wide photon conservation: $\sum_{j=0}^{n-1} u^{(k)}_j = \sum_{i=0}^{m-1} y_i$ for all $k \ge 1$ |
| `Lucky/Basic.lean` | `rl_weighted_reblurred_flux_conservation` | Sensitivity-weighted RL conserves reblurred flux $\sum_i (H u')_i = \sum_i y_i$ for **any** matrix $H$ |
| `Lucky/Basic.lean` | `rl_cleared_denominator_flux_general` | Division-free universal polynomial identity: $\sum_{j=0}^{n-1} N_j = D \sum_{i=0}^{m-1} y_i$ |
| `Lucky/Basic.lean` | `rl_trajectory_nonneg` & `rl_strict_pos` | Non-negativity and strict positivity invariance across all iterations $k \ge 0$ |
| `Lucky/Basic.lean` | `rl_trajectory_zero_locking` | Multiplicative support invariance ($u^{(0)}_j = 0 \implies u^{(k)}_j = 0$) |
| `Lucky/Basic.lean` | `rl_exact_solution_fixed_point` | Any exact solution $H u^* = y$ is a fixed point of Lucy-Richardson |
| `Lucky/Basic.lean` | `rl_kkt_stationarity_iff` | Exact equivalence between RL fixed points and KKT complementarity ($u_j = 0 \lor (H^T r)_j = w_j$) |
| `Lucky/Divergence.lean` | `em_latent_detector_marginal` & `em_latent_source_marginal` | Shepp-Vardi EM latent photon table $z_{i,j} = H_{i,j} u_j r_i$ marginals recover $y_i$ and $u'_j$ |
| `Lucky/Divergence.lean` | `em_surrogate_global_minimizer` | The Lucy-Richardson update $u^+$ is the unique global minimizer of the EM auxiliary surrogate $Q(v \mid u)$ |
| `Lucky/Divergence.lean` | `csiszar_tusnady_quantitative_descent` | Csiszár-Tusnády Pythagorean descent: $D_{\text{KL}}(y \,\|\, H u^+) + \sum_j w_j d_{\text{KL}}(u^+_j, u_j) \le D_{\text{KL}}(y \,\|\, H u)$ |
| `Lucky/Divergence.lean` | `rl_kl_strict_descent_of_ne` & `rl_kl_stationary_implies_fixed_point` | Strict decrease of KL divergence on any non-stationary iterate; no non-trivial limit cycles |
| `Lucky/Improved.lean` | `dey_tv_rl_positivity_breakdown` | Proves standard unsplit TV-RL (Dey et al. 2006) produces strictly negative intensities ($u'_j < 0$) when $\lambda \operatorname{div}_j > w_j$ |
| `Lucky/Improved.lean` | `split_reg_flux_balance` & `split_reg_zero_net_gradient` | Split-EM regularizer components $R^+_j(u), R^-_j(u) \ge 0$ satisfy $\sum_j R^+_j(u) = \sum_j R^-_j(u)$ and $\sum_j \nabla_j \Phi(u) = 0$ |
| `Lucky/Improved.lean` | `discrete_greens_first_identity` | $2 \sum_{j=0}^{n-1} u_j \nabla_j \Phi(u) = \sum_{j,l} c_{j,l} (u_j - u_l)^2 \ge 0$ |
| `Lucky/Improved.lean` | `split_em_update_unconditional_nonneg` | Unconditional non-negativity ($u'_j \ge 0$) of Split-EM Regularized RL for **all** $\lambda \ge 0$ |
| `Lucky/Improved.lean` | `split_em_kkt_stationarity_iff` | Fixed points of Split-EM RL satisfy the exact MAP Euler-Lagrange equation $(H^T r)_j - w_j = \lambda \nabla_j \Phi(u)$ |
| `Lucky/Improved.lean` | `damped_denom_strictly_pos` & `damped_ratio_spike_attenuation` | Readout-noise offset $b > 0$ prevents division by zero, preserves exact fixed points, and contracts noise spikes |
| `Lucky/Improved.lean` | `damped_rl_flux_balance_law` | Exact flux balance law under readout damping, motivating explicit flux-conservation projection |
| `Lucky/Improved.lean` | `guarded_trajectory_monotone_descent` | Safeguarded accelerated PCLR satisfies $E(u^{(k+1)}) \le E(T(u^{(k)})) \le E(u^{(k)}) \le E(u^{(0)})$ for all $k \ge 0$ |
| `Lucky/BregmanTGV.lean` | `second_order_linear_ramp_zero_staircasing` | Second-order midpoint Split-EM satisfies $R^{+,2}_x(u) = R^{-,2}_x(u)$ and $\nabla_x \Phi_2(u) = 0$ on any affine ramp $u(z) = a z + b$ |
| `Lucky/BregmanTGV.lean` | `jordan_split_properties` | Canonical Jordan split $q = q^+ - q^-$ satisfies $q^+ \ge 0, q^- \ge 0, q^+ - q^- = q, q^+ q^- = 0$ |
| `Lucky/BregmanTGV.lean` | `tgv_bregman_unconditional_nonneg` & `tgv_bregman_kkt_stationarity_iff` | Unconditional non-negativity and exact Primal-Dual KKT stationarity for TGV-Bregman Split-EM |
| `Lucky/BregmanTGV.lean` | `bregman_exact_contrast_restoration_fixed_point` | Proves Bregman subgradient matching restores exact solutions $H u^* = y$ as fixed points for all $\lambda_1, \lambda_2 \ge 0$ (zero peak-clipping bias) |
| `Lucky/BregmanTGV.lean` | `bregman_ref_flux_balance_and_debiasing` | Reference-centered Bregman terms satisfy $\sum_j B^+_j(u, u_{\text{ref}}) = \sum_j B^-_j(u, u_{\text{ref}})$ and $B^+_j(u, u) = B^-_j(u, u)$ |
| `Lucky/BregmanTGV.lean` | `neumann_backproject_fixed_point_invariance` & `neumann_boost_zero_net_flux` | Neumann-series preconditioned backprojector $(I + \alpha(I - H^T H)) H^T$ preserves exact fixed points and has zero net DC flux perturbation |
| `Lucky/Multiframe.lean` | `multi_adjoint_inner_product` & `multi_rl_flux_conservation` | Multi-frame adjoint duality $\langle u, \sum_f \gamma_f H_f^T r_f \rangle_n = \sum_f \gamma_f \langle r_f, H_f u \rangle_m$ and exact weighted burst photon conservation |
| `Lucky/Multiframe.lean` | `multi_kl_monotonic_descent` & `multi_kl_strict_descent` | Monotonic and strict Csiszár KL descent for joint multi-frame "Lucky Imaging" EM |
| `Lucky/Multiframe.lean` | `multi_tgv_bregman_unconditional_nonneg` & `multi_tgv_bregman_kkt_stationarity_iff` | Unconditional non-negativity and exact Primal-Dual KKT stationarity for Joint Multi-Frame TGV-Bregman Split-EM |
| `Lucky/Multiframe.lean` | `multi_bregman_exact_contrast_restoration_fixed_point` | Joint exact multi-frame solution $H_f u^* = y_f$ ($\forall f$) is an exact fixed point of Multi-Frame TGV-Bregman PCLR |
| `Lucky/Multiframe.lean` | `redescending_split_em_positivity_and_flux_balance` | Symmetric redescending edge-stopping conductivities $\psi((u_j - u_l)^2)$ satisfy exact global flux balance $\sum_j R^+_j = \sum_j R^-_j$ |
| `Lucky/Blind.lean` | `blind_bilinear_duality` & `blind_joint_adjoint_identity` | Bilinear image-PSF convolution duality $H(h) u = U(u) h$ and dual-role adjoint identity $\langle u, H(h)^T r \rangle_n = \langle h, U(u)^T r \rangle_K$ |
| `Lucky/Blind.lean` | `blind_psf_simplex_conservation_and_nonneg` | Unconditional non-negativity and exact probability-simplex flux conservation $\sum_k s_k(u) h'_k = \sum_i y_i$ of the Blind PSF M-step |
| `Lucky/Blind.lean` | `blind_alternating_kl_monotonic_descent` | Monotonic Csiszár KL divergence descent $D_{\text{KL}}(y \,\|\, H(h') u') \le D_{\text{KL}}(y \,\|\, H(h) u)$ across full alternating $(u, h)$ blind EM cycles |
| `Lucky/Blind.lean` | `blind_no_blur_degeneracy_theorem` & `blur_strictly_reduces_step_edge_variation` | Formal proof of the Levin et al. (2009) trivial no-blur MAP degeneracy $E_{\text{MAP}}(y, \delta) < E_{\text{MAP}}(u_{\text{true}}, h_{\text{true}})$ and constructive step-edge variance reduction |
| `Lucky/Blind.lean` | `dual_affine_shifted_gradient_cancellation` & `jordan_gradient_convolution_identity` | Exact cross-term cancellation $(c+g_u)(c+\delta) + (c-g_u)(c-\delta) = 2c^2 + 2g_u\delta$ and Jordan signed-gradient convolution identity |
| `Lucky/Blind.lean` | `blind_m_weighted_psf_exact_fixed_point` | Exact fixed-point preservation $h'_k = h_k$ at the true PSF for $M$-weighted foreground + Jordan dual-affine signed-gradient Split-EM |
| `Lucky/Blind.lean` | `blind_multiframe_cross_convolution_identifiability` | Exact scene-independent cross-convolution identity $H_2 y_1 - H_1 y_2 = 0$ for multi-frame blind deconvolution |
| `Lucky/Lunar.lean` | `preconditioned_split_em_exact_fixed_point` & `preconditioned_split_em_positivity_and_flux` | **Theorems 13a–13b**: Compact Adjoint Preconditioned Split-EM $K_{\text{adj}}^T (y / H_{\text{fwd}} u)$ preserves the exact data-fidelity fixed point $H_{\text{fwd}} u^* = y$, unconditional positivity, and global flux balance |
| `Lucky/Lunar.lean` | `rille_centerline_first_order_blindness_vs_hessian_detection` & `crater_slope_vs_rille_orthogonal_separation` | **Theorems 13c–13d**: Centered 1st-order gradients vanish identically at symmetric rille centerlines ($D_1 u = 0$) whereas 2nd-order Hessian curvature $D_2 u = 2(u(x+d)-u(x)) \ne 0$ strictly detects rilles and vanishes on linear crater slopes |
| `Lucky/Lunar.lean` | `anisotropic_coherence_split_reg_flux_balance` | **Theorem 13e**: Pairwise-symmetrized structure-tensor anisotropic Split-EM $c_{j,l}^{\text{aniso}} = (d_{j\to l} + d_{l\to j})\psi_{j,l}$ guarantees positivity and exact global flux balance $\sum_j R^{+, \text{aniso}}_j = \sum_j R^{-, \text{aniso}}_j$ |
| `Lucky/Lunar.lean` | `blur_convex_hull_maximum_principle` & `pre_deconvolution_clipping_destroys_fixed_point` | **Theorems 13f–13g**: Discrete convex-hull maximum principle for physical blur and proof that pre-deconvolution peak clipping strictly destroys the physical fixed point |
| `Lucky/Lunar.lean` | `censored_saturation_split_em_restores_fixed_point` | **Theorem 14a**: Right-censored saturation Split-EM ($r_i = 1$ when $y_i \ge T$) restores the exact unclipped scene $u^*$ as a fixed point despite 16-bit ADC clipping |
| `Lucky/Lunar.lean` | `renyi_escort_adjoint_properties` & `multiscale_dilated_split_reg_flux_balance` | **Theorems 14b–14c**: Rényi-Escort fractional-power adjoint preconditioner $K_{\text{adj}} \propto H^\alpha$ peak concentration & fixed-point invariance; multi-scale dilated Split-EM flux conservation |
| `Lucky/Limits.lean` | `minimax_nullspace_indistinguishability_bound` | **Theorem 15a**: Exact minimax risk decomposition $2(\|A(H u_1)-u_1\|_2^2 + \|A(H u_2)-u_2\|_2^2) = \|u_1-u_2\|_2^2 + \|2A(H u_1)-(u_1+u_2)\|_2^2 \ge \|u_1-u_2\|_2^2$ for optical nullspace metamers $H u_1 = H u_2$ |
| `Lucky/Limits.lean` | `spectral_completion_of_squares_and_harmonic_floor` & `naive_inversion_catastrophic_excess_risk` | **Theorems 15b–15c**: Brahmagupta-Fibonacci completion of squares proving the exact Spectral Wiener-Oracle Minimum MSE harmonic floor $(q S \sigma)^2 \le ((h S)^2 + \sigma^2)\operatorname{MSE}(p/q)$ and Rayleigh-Shannon sub-noise cutoff |
| `Lucky/Limits.lean` | `adaptive_nonlinear_strictly_beats_stationary_wiener` | **Theorem 15d**: Exact cleared-denominator proof that a two-regime spatially-adaptive non-linear estimator strictly beats the Stationary Linear Wiener Oracle by $N^2 (B - A)^2 > 0$ on heterogeneous lunar scenes |
| `Lucky/Limits.lean` | `positivity_projection_strictly_reduces_error` & `positivity_projection_vec_error_contraction_and_strict_gain` | **Theorems 15e & 15h**: Pointwise and $n$-dimensional vector positivity projection $\max(0, \hat{u})$ is globally non-expansive and **strictly reduces $L^2$ error** by at least $\hat{u}_{j_0}^2 > 0$ whenever any pixel undershoots into negative Gibbs ringing |
| `Lucky/Limits.lean` | `cramer_rao_two_mode_information_bound` & `general_cauchy_schwarz_cramer_rao_discriminant_identity` | **Theorems 15f & 15j**: Two-mode and arbitrary-$K$-dimensional discrete Cramér-Rao / Cauchy-Schwarz discriminant identity $\sum_{k=0}^{K-1}(C e_k - B s_k)^2 + B^2 C = A C^2 \implies \operatorname{Cov}(e, s)^2 \le \operatorname{Var}(e) I_F$ |
| `Lucky/Limits.lean` | `multiframe_lucky_oracle_strict_improvement` & `multimode_spectral_completion_of_squares_and_global_harmonic_bound` | **Theorems 15g & 15i**: Adding a Lucky Imaging frame with positive Fisher information $dI > 0$ strictly reduces multi-frame oracle MSE; global multi-mode Plancherel spectral harmonic bound |
| `Lucky/Limits.lean` | `semi_convergence_u_shaped_risk_and_morozov_stopping` & `stein_sure_unbiased_risk_identity` | **Theorems 15k & 15l**: Formal proof of iterative semi-convergence U-shaped risk trajectory & Morozov discrepancy stopping; finite-sample Stein's Unbiased Risk Estimate (SURE) identity $\|\hat{\mu} - \mu\|_2^2 + \|\eta\|_2^2 = \|y - \hat{\mu}\|_2^2 + 2\langle \eta, \hat{\mu} - \mu \rangle$ |
| `Lucky/Limits.lean` | `law_of_total_variance_wiener_pilot_power_restoration` | **Theorem 15m**: Law of Total Variance for Empirical-Bayes Wiener Deconvolution: $S_{\text{pil}}(\omega) = \frac{H_2 S_0^2}{H_2 S_0 + N}$ strictly underestimates true signal power $S_0$ by $V_{\text{post}}(\omega) = \frac{S_0 N}{H_2 S_0 + N}$ ($D S_0 - H_2 S_0^2 = S_0 N > 0$), and $S_{\text{EM}} = S_{\text{pil}} + V_{\text{post}}$ identically restores $S_0$ |
| `Lucky/Limits.lean` | `stein_morozov_effective_dof_residual_discrepancy_identity` | **Theorem 15n**: Exact multi-mode Stein-Morozov residual discrepancy identity $\sum_\omega D_\omega N_\omega = \sum_\omega N_\omega^2 + \sum_\omega H_{2,\omega} S_{0,\omega} N_\omega$, proving optimal stopping residual variance is smaller than raw noise variance by the Stein Effective Degrees of Freedom $\operatorname{df}_{\text{eff}}$ |
| `Lucky/Limits.lean` | `parseval_sure_two_band_mixing_risk_identity` | **Theorem 15o**: Exact Parseval-SURE two-band spatial-spectral mixing risk identity $q_{\text{opt}}(u_{\text{mix}} - q u^*)^2 = (p q_{\text{opt}} - q p_{\text{opt}})^2$, achieving minimum MSE across heterogeneous crater rims and lunar maria |
| `Lucky/Limits.lean` | `noise_broadened_variance_additivity_and_debiasing` | **Theorem 15p**: Exact noise-broadened LSF second-moment additivity $V_{\text{obs}} = V_{\text{psf}} + V_{\text{noise}}$ and strict squared-error reduction from noise-broadening debiasing |

---

## Gen-7 Lunar Ground-Truth & Theoretical Limits Benchmark (`recover_lunar.py`)

Both **Non-Blind Gen-7 Lunar PCLR** and **100% Blind Gen-7 Lunar PCLR** (zero access to `u_true` or `psf_true`) strictly surpass **both** the **Omniscient 1D Radial Spectral Wiener Oracle** and the **Omniscient 2D Anisotropic Spectral Wiener Oracle** across every real lunar ground-truth scene:

| Lunar Benchmark Scene | Blurred Input | Classic RL (Reflect) | Classic Blind RL (Fish 1995) | Omniscient 1D Radial Oracle | Omniscient 2D PSD Oracle | **100% Blind Gen-7 PCLR (Ours)** | **Non-Blind Gen-7 PCLR (Ours)** | **Blind PSF PSNR** |
|---|---|---|---|---|---|---|---|---|
| **Lunar GT 1: High-SNR Craters & Rilles** | `31.25 dB` | `33.70 dB` | `33.40 dB` | `34.74 dB` | `34.80 dB` | **`34.91 dB`** (`0.8510`) | **`34.94 dB`** (`0.8519`) | **`49.78 dB`** (vs `23.1 dB`) |
| **Lunar GT 2: Mid-SNR Highland Craters** | `31.89 dB` | `31.20 dB` | `30.02 dB` | `34.49 dB` | `34.79 dB` | **`34.81 dB`** (`0.8738`) | **`34.80 dB`** (`0.8733`) | **`43.23 dB`** (vs `20.3 dB`) |
| **Lunar GT 3: Heavy-Noise Mare & Rim** | `30.37 dB` | `30.34 dB` | `30.15 dB` | `32.63 dB` | `32.73 dB` | **`32.82 dB`** (`0.8046`) | **`32.82 dB`** (`0.8043`) | **`42.45 dB`** (vs `29.2 dB`) |
| **Lunar GT 4: Multi-Frame Lucky Burst ($F=8$)** | `32.47 dB` | `35.58 dB` | `33.55 dB` | — | — | **`36.58 dB`** (`0.9143`) | **`36.41 dB`** (`0.9102`) | **`34.33 dB`** (vs `23.8 dB`) |

### Visual Benchmark Results

![Lunar Ground-Truth Benchmark](out/lunar_groundtruth_benchmark.png)

![Theoretical Limits & Spectral Harmonic Floor](out/lunar_theoretical_limits.png)

![Crater Gassendi & Rimae Gassendi High-Resolution Zoom](out/lunar_gassendi_zoom_comparison.png)

![Real Ground-Based Telescope Lunar Suite](out/lunar_real_telescope_suite.png)



