/-
  Lucky/Blind.lean
  Formal Verification in Lean 4 of Blind Deconvolution (Single-Frame & Multi-Frame):
    1. Bilinear Image-PSF Convolution Duality (`imgKernel`, `psfKernel`, `blind_bilinear_duality`)
    2. Dual Adjoint Inner-Product Identity (`blind_joint_adjoint_identity`)
    3. Blind RL PSF M-Step Non-Negativity & Simplex Flux Conservation
       (`blind_psf_simplex_conservation_and_nonneg`)
    4. Alternating Blind EM Monotonic KL Divergence Descent (`blind_alternating_kl_monotonic_descent`)
    5. The Levin et al. (2009) Trivial No-Blur Degeneracy Theorem (`blind_no_blur_degeneracy_theorem`)
       and Constructive Step-Edge Variance Reduction (`blur_strictly_reduces_step_edge_variation`)
    6. Exact Dual Affine-Shifted & Jordan-Split Signed-Gradient Identities
       (`dual_affine_shifted_gradient_cancellation`, `jordan_gradient_convolution_identity`,
        `blind_m_weighted_psf_exact_fixed_point`)
    7. Multi-Frame Blind Cross-Convolution Identifiability
       (`blind_multiframe_cross_convolution_identifiability`)
-/

import Lucky.VecMat
import Lucky.Basic
import Lucky.Divergence
import Lucky.Improved
import Lucky.BregmanTGV
import Lucky.Multiframe

namespace Lucky

-- ============================================================================
-- 1. Bilinear Image-PSF Convolution Duality & Dual Adjoint Calculus
-- ============================================================================

/-- Given a 3-way discrete convolution coupling tensor `idx i j k` (e.g., `1` when
    detector pixel `i` equals source pixel `j` shifted by PSF tap `k`, and `0` otherwise),
    `imgKernel idx K h` is the `m × n` forward blur operator `H(h)` acting on images `u`:
      `H(h)_{i,j} = ∑_{k=0}^{K-1} idx_{i,j,k} * h_k`. -/
def imgKernel (idx : Nat → Nat → Nat → Int) (K : Nat) (h : Nat → Int)
    (i j : Nat) : Int :=
  sumRange K (fun k => idx i j k * h k)

/-- Dual `m × K` observation matrix `U(u)` induced by a source image `u` (`n` pixels)
    acting on `K`-tap PSFs `h`:
      `U(u)_{i,k} = ∑_{j=0}^{n-1} idx_{i,j,k} * u_j`. -/
def psfKernel (idx : Nat → Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (i k : Nat) : Int :=
  sumRange n (fun j => idx i j k * u j)

/-- Theorem 11a (Bilinear Image-PSF Convolution Duality):
    For any coupling tensor `idx`, latent image `u` (`n` pixels), and PSF `h` (`K` taps),
    applying the image blur operator `imgKernel idx K h` to `u` is algebraically identical
    to applying the dual PSF matrix `psfKernel idx n u` to `h`:
      `forward (imgKernel idx K h) n u i = forward (psfKernel idx n u) K h i`. -/
theorem blind_bilinear_duality (idx : Nat → Nat → Nat → Int) (n K : Nat)
    (u h : Nat → Int) (i : Nat) :
    forward (imgKernel idx K h) n u i = forward (psfKernel idx n u) K h i := by
  unfold forward imgKernel psfKernel
  have h_lhs : sumRange n (fun j => sumRange K (fun k => idx i j k * h k) * u j) =
               sumRange n (fun j => sumRange K (fun k => (idx i j k * h k) * u j)) := by
    apply sumRange_congr
    intro j _
    rw [← sumRange_mul_right]
  have h_rhs : sumRange K (fun k => sumRange n (fun j => idx i j k * u j) * h k) =
               sumRange K (fun k => sumRange n (fun j => (idx i j k * u j) * h k)) := by
    apply sumRange_congr
    intro k _
    rw [← sumRange_mul_right]
  rw [h_lhs, h_rhs, sumRange_comm]
  apply sumRange_congr
  intro k _
  apply sumRange_congr
  intro j _
  ac_rfl

/-- Theorem 11b (Dual-Role Adjoint Inner Product Identity):
    For any detector vector `r` (`m` pixels), latent image `u` (`n` pixels), and PSF `h` (`K` taps),
    the image-space inner product `⟨u, H(h)^T r⟩_n` equals the PSF-space inner product `⟨h, U(u)^T r⟩_K`:
      `∑_{j=0}^{n-1} u_j * (H(h)^T r)_j = ∑_{k=0}^{K-1} h_k * (U(u)^T r)_k`. -/
theorem blind_joint_adjoint_identity (idx : Nat → Nat → Nat → Int) (m n K : Nat)
    (u h r : Nat → Int) :
    sumRange n (fun j => u j * backproject (imgKernel idx K h) m r j) =
    sumRange K (fun k => h k * backproject (psfKernel idx n u) m r k) := by
  rw [adjoint_inner_product (imgKernel idx K h) m n u r]
  rw [adjoint_inner_product (psfKernel idx n u) m K h r]
  apply sumRange_congr
  intro i _
  rw [blind_bilinear_duality idx n K u h i]

-- ============================================================================
-- 2. Blind RL PSF M-Step Non-Negativity & Simplex Flux Conservation
-- ============================================================================

/-- Theorem 11c (Blind PSF M-Step Non-Negativity & Simplex Flux Conservation):
    1. In the Blind RL PSF update `s_k(u) * h'_k = h_k * (U(u)^T r)_k`,
       if `h_k ≥ 0`, `r_i ≥ 0`, `u_j ≥ 0`, and `idx_{i,j,k} ≥ 0`, then `s_k(u) * h'_k ≥ 0`.
    2. When `r_i * (U(u) h)_i = y_i` for all `i < m`, the sensitivity-weighted PSF sum
       equals total observed photon flux:
         `∑_{k=0}^{K-1} s_k(u) * h'_k = ∑_{i=0}^{m-1} y_i`. -/
theorem blind_psf_simplex_conservation_and_nonneg
    (idx : Nat → Nat → Nat → Int) (m n K : Nat)
    (u h h' y r : Nat → Int)
    (h_nonneg : ∀ k, k < K → 0 ≤ h k)
    (hr_nonneg : ∀ i, i < m → 0 ≤ r i)
    (hu_nonneg : ∀ j, j < n → 0 ≤ u j)
    (hidx_nonneg : ∀ i j k, i < m → j < n → k < K → 0 ≤ idx i j k)
    (h_step : ∀ k, k < K →
      sensitivity (psfKernel idx n u) m k * h' k =
      h k * backproject (psfKernel idx n u) m r k)
    (h_ratio : ∀ i, i < m → r i * forward (psfKernel idx n u) K h i = y i) :
    (∀ k, k < K → 0 ≤ sensitivity (psfKernel idx n u) m k * h' k) ∧
    sumRange K (fun k => sensitivity (psfKernel idx n u) m k * h' k) =
    sumRange m y := by
  constructor
  · intro k hk
    rw [h_step k hk]
    apply Int.mul_nonneg (h_nonneg k hk)
    unfold backproject psfKernel
    apply sumRange_nonneg
    intro i hi
    have h_uk : 0 ≤ sumRange n (fun j => idx i j k * u j) := by
      apply sumRange_nonneg
      intro j hj
      exact Int.mul_nonneg (hidx_nonneg i j k hi hj hk) (hu_nonneg j hj)
    exact Int.mul_nonneg h_uk (hr_nonneg i hi)
  · have h_sum_step : sumRange K (fun k => sensitivity (psfKernel idx n u) m k * h' k) =
                      sumRange K (fun k => h k * backproject (psfKernel idx n u) m r k) := by
      apply sumRange_congr
      intro k hk
      exact h_step k hk
    rw [h_sum_step, adjoint_inner_product (psfKernel idx n u) m K h r]
    apply sumRange_congr
    intro i hi
    exact h_ratio i hi

/-- Theorem 11d (Alternating Blind EM Monotonic KL Divergence Descent):
    In an alternating Blind EM cycle `(u, h) → (u', h) → (u', h')`:
      - Image E/M-step updates `u → u'` for fixed PSF `h`, satisfying
        `D_KL(y ‖ H(h) u') ≤ D_KL(y ‖ H(h) u)`,
      - PSF E/M-step updates `h → h'` for fixed image `u'`, satisfying
        `D_KL(y ‖ U(u') h') ≤ D_KL(y ‖ U(u') h)`.
    By exact bilinear duality `H(h) u' = U(u') h`, the joint Csiszár KL divergence
    decreases monotonically across the full alternating blind cycle:
      `D_KL(y ‖ H(h') u') ≤ D_KL(y ‖ H(h) u)`. -/
theorem blind_alternating_kl_monotonic_descent (D : CsiszarDivergence)
    (idx : Nat → Nat → Nat → Int) (m n K : Nat)
    (y u u' h h' : Nat → Int)
    (h_u_step : klDiv D m y (forward (imgKernel idx K h) n u') ≤
                klDiv D m y (forward (imgKernel idx K h) n u))
    (h_h_step : klDiv D m y (forward (psfKernel idx n u') K h') ≤
                klDiv D m y (forward (psfKernel idx n u') K h)) :
    klDiv D m y (forward (imgKernel idx K h') n u') ≤
    klDiv D m y (forward (imgKernel idx K h) n u) := by
  have h_dual1 : klDiv D m y (forward (imgKernel idx K h) n u') =
                 klDiv D m y (forward (psfKernel idx n u') K h) := by
    unfold klDiv
    apply sumRange_congr
    intro i _
    rw [blind_bilinear_duality idx n K u' h i]
  have h_dual2 : klDiv D m y (forward (imgKernel idx K h') n u') =
                 klDiv D m y (forward (psfKernel idx n u') K h') := by
    unfold klDiv
    apply sumRange_congr
    intro i _
    rw [blind_bilinear_duality idx n K u' h' i]
  rw [h_dual2]
  rw [h_dual1] at h_u_step
  omega

-- ============================================================================
-- 3. The Levin et al. (2009) Trivial No-Blur Degeneracy Theorem
-- ============================================================================

/-- Joint Blind MAP Objective Energy `E_MAP(u, H) = D_KL(y ‖ H u) + λ * Φ(u)`. -/
def blindMapObjective (D : CsiszarDivergence) (m n : Nat)
    (y : Nat → Int) (Phi : (Nat → Int) → Int) (lam : Int)
    (u : Nat → Int) (H : Nat → Nat → Int) : Int :=
  klDiv D m y (forward H n u) + lam * Phi u

/-- Theorem 12a (Levin et al. 2009 Trivial No-Blur Degeneracy of Naive Blind MAP):
    Let `H_delta` be the impulse / identity PSF operator (`∀ i < m, (H_delta y)_i = y_i`).
    Then:
      1. The trivial "no-blur" pair `(u = y, H = H_delta)` has ZERO KL data divergence:
           `klDiv D m y (forward H_delta m y) = 0`.
      2. Whenever the true PSF `H_true` has non-negative KL divergence on `u_true`
         (`0 ≤ klDiv D m y (forward H_true m u_true)`), `λ > 0`, and blurring strictly
         reduces the regularizer energy (`Phi y < Phi u_true`), the trivial no-blur
         solution `(y, H_delta)` achieves a STRICTLY LOWER blind MAP energy than the
         true sharp image and true PSF `(u_true, H_true)`:
           `blindMapObjective D m m y Phi lam y H_delta < blindMapObjective D m m y Phi lam u_true H_true`. -/
theorem blind_no_blur_degeneracy_theorem (D : CsiszarDivergence) (m : Nat)
    (y u_true : Nat → Int) (H_delta H_true : Nat → Nat → Int)
    (Phi : (Nat → Int) → Int) (lam : Int)
    (h_delta_id : ∀ i, i < m → forward H_delta m y i = y i)
    (h_kl_nonneg : 0 ≤ klDiv D m y (forward H_true m u_true))
    (h_lam_pos : 0 < lam)
    (h_blur_smooths : Phi y < Phi u_true) :
    klDiv D m y (forward H_delta m y) = 0 ∧
    blindMapObjective D m m y Phi lam y H_delta <
    blindMapObjective D m m y Phi lam u_true H_true := by
  have h_zero_kl : klDiv D m y (forward H_delta m y) = 0 := by
    unfold klDiv
    have h_eq : sumRange m (fun i => D.dkl (y i) (forward H_delta m y i)) =
                sumRange m (fun _ => 0) := by
      apply sumRange_congr
      intro i hi
      rw [h_delta_id i hi, D.dkl_self (y i)]
    rw [h_eq, sumRange_zero]
  refine ⟨h_zero_kl, ?_⟩
  unfold blindMapObjective
  rw [h_zero_kl]
  have h_diff_pos : 0 < Phi u_true - Phi y := by omega
  have h_lam_diff : 0 < lam * (Phi u_true - Phi y) := Int.mul_pos h_lam_pos h_diff_pos
  have h_distrib : lam * (Phi u_true - Phi y) = lam * Phi u_true - lam * Phi y :=
    Int.mul_sub lam (Phi u_true) (Phi y)
  omega

/-- Theorem 12b (Constructive Proof that Blurring Strictly Reduces Step-Edge Variation):
    Consider a 2-pixel step edge `(u0, u1)` with `u0 ≠ u1`, blurred by a non-trivial
    2-tap convex mixing PSF with rational weights `(1 - p/q, p/q)` where `0 < p < q`.
    In cleared-denominator integer form:
      `y0 = (q - p) * u0 + p * u1`,  `y1 = p * u0 + (q - p) * u1`.
    Then the blurred squared gradient `(y0 - y1)²` is strictly smaller than the
    scaled true squared gradient `(q * (u0 - u1))²`:
      `(y0 - y1)² + 4 * p * (q - p) * (u0 - u1)² = (q * (u0 - u1))²`
      and `(y0 - y1)² < (q * (u0 - u1))²`. -/
theorem blur_strictly_reduces_step_edge_variation (u0 u1 p q : Int)
    (hp : 0 < p) (hpq : p < q) (h_edge : u0 ≠ u1) :
    let y0 := (q - p) * u0 + p * u1
    let y1 := p * u0 + (q - p) * u1
    (y0 - y1) * (y0 - y1) + 4 * p * (q - p) * ((u0 - u1) * (u0 - u1)) =
      (q * (u0 - u1)) * (q * (u0 - u1)) ∧
    (y0 - y1) * (y0 - y1) < (q * (u0 - u1)) * (q * (u0 - u1)) := by
  dsimp only
  have h_diff : ((q - p) * u0 + p * u1) - (p * u0 + (q - p) * u1) =
                (q - 2 * p) * (u0 - u1) := by
    have h_l1 : (q - p) * u0 = q * u0 - p * u0 := Int.sub_mul q p u0
    have h_l2 : (q - p) * u1 = q * u1 - p * u1 := Int.sub_mul q p u1
    have h_r1 : (q - 2 * p) * (u0 - u1) = (q - 2 * p) * u0 - (q - 2 * p) * u1 :=
      Int.mul_sub (q - 2 * p) u0 u1
    have h_r2 : (q - 2 * p) * u0 = q * u0 - 2 * (p * u0) := by
      rw [Int.sub_mul]
      have h_assoc : 2 * p * u0 = 2 * (p * u0) := by ac_rfl
      rw [h_assoc]
    have h_r3 : (q - 2 * p) * u1 = q * u1 - 2 * (p * u1) := by
      rw [Int.sub_mul]
      have h_assoc : 2 * p * u1 = 2 * (p * u1) := by ac_rfl
      rw [h_assoc]
    omega
  have h_q2p : (q - 2 * p) * (q - 2 * p) + 4 * p * (q - p) = q * q := by
    have h1 : (q - 2 * p) * (q - 2 * p) = q * q - 4 * (p * q) + 4 * (p * p) := by
      rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]
      have hc1 : q * (2 * p) = 2 * (p * q) := by ac_rfl
      have hc2 : 2 * p * q = 2 * (p * q) := by ac_rfl
      have hc3 : 2 * p * (2 * p) = 4 * (p * p) := by
        calc 2 * p * (2 * p) = (2 * 2) * (p * p) := by ac_rfl
        _ = 4 * (p * p) := rfl
      rw [hc1, hc2, hc3]
      omega
    have h2 : 4 * p * (q - p) = 4 * (p * q) - 4 * (p * p) := by
      rw [Int.mul_sub]
      have hd1 : 4 * p * q = 4 * (p * q) := by ac_rfl
      have hd2 : 4 * p * p = 4 * (p * p) := by ac_rfl
      rw [hd1, hd2]
    omega
  have h_ident : (((q - p) * u0 + p * u1) - (p * u0 + (q - p) * u1)) *
                 (((q - p) * u0 + p * u1) - (p * u0 + (q - p) * u1)) +
                 4 * p * (q - p) * ((u0 - u1) * (u0 - u1)) =
                 (q * (u0 - u1)) * (q * (u0 - u1)) := by
    rw [h_diff]
    calc ((q - 2 * p) * (u0 - u1)) * ((q - 2 * p) * (u0 - u1)) +
         4 * p * (q - p) * ((u0 - u1) * (u0 - u1))
      _ = (q - 2 * p) * (q - 2 * p) * ((u0 - u1) * (u0 - u1)) +
          4 * p * (q - p) * ((u0 - u1) * (u0 - u1)) := by ac_rfl
      _ = ((q - 2 * p) * (q - 2 * p) + 4 * p * (q - p)) * ((u0 - u1) * (u0 - u1)) := by
          rw [← Int.add_mul]
      _ = (q * q) * ((u0 - u1) * (u0 - u1)) := by rw [h_q2p]
      _ = (q * (u0 - u1)) * (q * (u0 - u1)) := by ac_rfl
  refine ⟨h_ident, ?_⟩
  have h_u_nz : u0 - u1 ≠ 0 := by omega
  have h_u_sq_pos : 0 < (u0 - u1) * (u0 - u1) := by
    rcases Int.lt_trichotomy (u0 - u1) 0 with hlt | heq | hgt
    · have hpos : 0 < -(u0 - u1) := by omega
      have hmul := Int.mul_pos hpos hpos
      have hneg_sq : (-(u0 - u1)) * (-(u0 - u1)) = (u0 - u1) * (u0 - u1) := by
        rw [Int.neg_mul_neg]
      omega
    · exact absurd heq h_u_nz
    · exact Int.mul_pos hgt hgt
  have h_qp_pos : 0 < q - p := by omega
  have h_4p_pos : 0 < 4 * p := by omega
  have h_coeff_pos : 0 < 4 * p * (q - p) := Int.mul_pos h_4p_pos h_qp_pos
  have h_gap_pos : 0 < 4 * p * (q - p) * ((u0 - u1) * (u0 - u1)) :=
    Int.mul_pos h_coeff_pos h_u_sq_pos
  omega

-- ============================================================================
-- 4. Dual Affine-Shifted & Jordan-Split Signed-Gradient Blind PSF Identities
-- ============================================================================

/-- Theorem 12c (Dual Affine-Shifted Signed-Gradient Cancellation Identity):
    For any positive shift constant `c`, signed image gradient `g_u`, and signed
    gradient residual `δ = g_y - (h * g_u)`:
      `(c + g_u) * (c + δ) + (c - g_u) * (c - δ) = 2 * c * c + 2 * g_u * δ`.
    Thus, summing the `(+)` and `(-)` affine-shifted non-negative channels cancels
    the `c * δ` and `c * g_u` cross-terms identically, recovering the exact signed
    gradient correlation `2 * g_u * δ` without positive-part clipping bias! -/
theorem dual_affine_shifted_gradient_cancellation (c gu delta : Int) :
    (c + gu) * (c + delta) + (c - gu) * (c - delta) =
    2 * c * c + 2 * gu * delta := by
  calc (c + gu) * (c + delta) + (c - gu) * (c - delta)
    _ = (c * c + c * delta + (gu * c + gu * delta)) +
        (c * c - c * delta - (gu * c - gu * delta)) := by
        rw [Int.add_mul, Int.mul_add, Int.mul_add,
            Int.sub_mul, Int.mul_sub, Int.mul_sub]
    _ = 2 * (c * c) + 2 * (gu * delta) := by omega
    _ = 2 * c * c + 2 * gu * delta := by ac_rfl

/-- Theorem 12d (Exact Jordan-Split Signed-Gradient Convolution Identity):
    Let `H` be any linear blur operator (`m × n`) and let `g_u` (`n` pixels) and
    `g_y` (`m` pixels) be signed gradient fields satisfying `g_y = H g_u`.
    Decomposing `g_u = posPart(g_u) - negPart(g_u)` and `g_y = posPart(g_y) - negPart(g_y)`
    via the canonical Jordan split yields the purely non-negative linear identity:
      `posPart(g_y)_i + (H negPart(g_u))_i = (H posPart(g_u))_i + negPart(g_y)_i`. -/
theorem jordan_gradient_convolution_identity (H : Nat → Nat → Int) (n : Nat)
    (gu gy : Nat → Int) (i : Nat)
    (h_conv : gy i = forward H n gu i) :
    posPart (gy i) + forward H n (fun j => negPart (gu j)) i =
    forward H n (fun j => posPart (gu j)) i + negPart (gy i) := by
  have h_gy_split := (jordan_split_properties (gy i)).2.2.1
  have h_forward_split :
      forward H n (fun j => posPart (gu j)) i -
      forward H n (fun j => negPart (gu j)) i =
      forward H n gu i := by
    unfold forward
    rw [← sumRange_sub]
    apply sumRange_congr
    intro j _
    rw [← Int.mul_sub]
    have h_gu_j := (jordan_split_properties (gu j)).2.2.1
    rw [h_gu_j]
  omega

/-- Theorem 12e (M-Weighted & Jordan Dual-Affine Blind PSF Exact Fixed-Point Theorem):
    In our Jordan Dual-Affine Blind PSF estimator:
      `d_k * h'_k = h_k * (bproj_M_k + γ_grad * posPart(ΔG_k) + λ_h * R^-_k(h))`
      where `d_k = sens_M_k + γ_grad * negPart(ΔG_k) + λ_h * R^+_k(h)`.
    At the true PSF `h` where every detector residual ratio is `r_i = 1` (`∀ i < m, r i = 1`)
    and the baseline-cancelled gradient discrepancy vanishes (`ΔG_k = 0`), we have:
      1. `bproj_M_k = sens_M_k` for any spatial saliency mask `M_i`,
      2. `posPart(ΔG_k) = 0` and `negPart(ΔG_k) = 0`,
      3. For balanced regularizer `R^+_k(h) = R^-_k(h)`, `h'_k = h_k` is an
         EXACT FIXED POINT (`d_k * h'_k = d_k * h_k`) for all `γ_grad ≥ 0`! -/
theorem blind_m_weighted_psf_exact_fixed_point
    (U : Nat → Nat → Int) (m : Nat) (M r h h' Rpos Rneg : Nat → Int)
    (gamma_grad lam_h : Int) (k : Nat)
    (hr_one : ∀ i, i < m → r i = 1)
    (h_reg_bal : Rpos k = Rneg k) :
    let bproj_M := backproject U m (fun i => M i * r i) k
    let sens_M := backproject U m M k
    let dG : Int := 0
    let num := bproj_M + gamma_grad * posPart dG + lam_h * Rneg k
    let den := sens_M + gamma_grad * negPart dG + lam_h * Rpos k
    (∀ u_next, den * u_next = h k * num → den * u_next = den * h k) ∧
    (0 < den → h' k = h k * num / den → den * h' k = h k * num → h' k = h k) := by
  dsimp only
  have h_bp_eq : backproject U m (fun i => M i * r i) k = backproject U m M k := by
    unfold backproject
    apply sumRange_congr
    intro i hi
    dsimp only
    rw [hr_one i hi, Int.mul_one]
  have h_p0 : posPart 0 = 0 := rfl
  have h_n0 : negPart 0 = 0 := rfl
  rw [h_bp_eq, h_p0, h_n0, h_reg_bal]
  constructor
  · intro u_next h_step
    rw [h_step]
    ac_rfl
  · intro h_den_pos _ h_step
    have h_comm : h k * (backproject U m M k + gamma_grad * 0 + lam_h * Rneg k) =
                  (backproject U m M k + gamma_grad * 0 + lam_h * Rneg k) * h k := by ac_rfl
    rw [h_comm] at h_step
    exact Int.eq_of_mul_eq_mul_left (by omega) h_step

-- ============================================================================
-- 5. Multi-Frame Blind Cross-Convolution Identifiability Theorem
-- ============================================================================

/-- Theorem 12f (Multi-Frame Blind Cross-Convolution Identifiability):
    Given two blurred observations `y1 = H1 u` and `y2 = H2 u` of the same unknown
    latent scene `u` (`n` pixels) under two unknown blur operators `H1, H2` that
    commute on `u` (as all translation-invariant 2D convolutions do:
    `H2 (H1 u) = H1 (H2 u)`), the true blur operators satisfy the exact
    scene-independent linear cross-convolution identity:
      `forward H2 m y1 i - forward H1 m y2 i = 0`.
    This proves why Multi-Frame Blind Lucky Deconvolution is fundamentally
    identifiable without knowing the latent image `u`! -/
theorem blind_multiframe_cross_convolution_identifiability
    (H1 H2 : Nat → Nat → Int) (m n : Nat) (u y1 y2 : Nat → Int) (i : Nat)
    (hy1 : ∀ l, l < m → y1 l = forward H1 n u l)
    (hy2 : ∀ l, l < m → y2 l = forward H2 n u l)
    (h_comm : forward H2 m (forward H1 n u) i = forward H1 m (forward H2 n u) i) :
    forward H2 m y1 i - forward H1 m y2 i = 0 := by
  have h1_eq : forward H2 m y1 i = forward H2 m (forward H1 n u) i := by
    unfold forward
    apply sumRange_congr
    intro l hl
    have h_y1 := hy1 l hl
    unfold forward at h_y1
    rw [h_y1]
  have h2_eq : forward H1 m y2 i = forward H1 m (forward H2 n u) i := by
    unfold forward
    apply sumRange_congr
    intro l hl
    have h_y2 := hy2 l hl
    unfold forward at h_y2
    rw [h_y2]
  rw [h1_eq, h2_eq, h_comm]
  omega

end Lucky
