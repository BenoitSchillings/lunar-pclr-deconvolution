/-
  Lucky/BregmanTGV.lean
  Formal Verification in Lean 4 of Second-Generation Improvements to Lucy-Richardson
  (TGV-Bregman PCLR):
    1. Second-Order Midpoint Split-EM Regularizer (`R^{+,2}`, `R^{-,2}`) &
       Zero-Staircasing Theorem on Arbitrary Affine Ramps `u(x) = a * x + b`
    2. Canonical Jordan Decomposition `q = q⁺ - q⁻` (`q⁺ ≥ 0, q⁻ ≥ 0, q⁺ * q⁻ = 0`)
       for Signed Dual Bregman Subgradients
    3. Unconditional Positivity, Exact Primal-Dual KKT Stationarity, and
       Zero-Peak-Clipping Fixed-Point Restoration of TGV-Bregman Split-EM
    4. Reference-Centered Bregman Operator `B⁺(u, u_ref), B⁻(u, u_ref)` Flux Balance
       and Exact Debiasing at `u = u_ref`
    5. Neumann-Series Preconditioned Backprojector `(I + α(I - H^T H)) H^T`:
       Fixed-Point Invariance & Zero DC Flux Perturbation
-/

import Lucky.VecMat
import Lucky.Basic
import Lucky.Divergence
import Lucky.Improved

namespace Lucky

-- ============================================================================
-- 1. Second-Order Midpoint Split-EM Regularizer & Zero-Staircasing Theorem
-- ============================================================================

/-- Repulsive (self) part of the 1D/directional second-order midpoint regularizer
    at pixel `x` with step `d`: `R^{+,2}_x(u) = 2 * c_x * u(x)`. -/
def regPos2D1 (c : Nat → Int) (u : Int → Int) (x : Int) (idx : Nat) : Int :=
  2 * c idx * u x

/-- Attractive (symmetric midpoint) part of the second-order regularizer
    at pixel `x` with step `d`: `R^{-,2}_x(u) = c_x * (u(x + d) + u(x - d))`. -/
def regNeg2D1 (c : Nat → Int) (u : Int → Int) (x d : Int) (idx : Nat) : Int :=
  c idx * (u (x + d) + u (x - d))

/-- Full second-order discrete Laplacian regularizer gradient:
    `∇_x Φ₂(u) = R^{+,2}_x(u) - R^{-,2}_x(u) = c_x * (2 u(x) - (u(x + d) + u(x - d)))`. -/
def regGrad2D1 (c : Nat → Int) (u : Int → Int) (x d : Int) (idx : Nat) : Int :=
  regPos2D1 c u x idx - regNeg2D1 c u x d idx

/-- Theorem 6a (Non-Negativity of Second-Order Midpoint Split Components):
    Whenever `c_x ≥ 0` and `u ≥ 0`, both `R^{+,2}_x(u) ≥ 0` and `R^{-,2}_x(u) ≥ 0`. -/
theorem second_order_split_nonneg (c : Nat → Int) (u : Int → Int) (x d : Int) (idx : Nat)
    (hc : 0 ≤ c idx) (hu_x : 0 ≤ u x) (hu_plus : 0 ≤ u (x + d)) (hu_minus : 0 ≤ u (x - d)) :
    0 ≤ regPos2D1 c u x idx ∧ 0 ≤ regNeg2D1 c u x d idx := by
  unfold regPos2D1 regNeg2D1
  have hc2 : 0 ≤ 2 * c idx := by omega
  have hsum : 0 ≤ u (x + d) + u (x - d) := by omega
  exact ⟨Int.mul_nonneg hc2 hu_x, Int.mul_nonneg hc hsum⟩

/-- Theorem 6b (Zero-Staircasing Theorem on Arbitrary Linear / Affine Ramps):
    For ANY affine intensity ramp `u(z) = a * z + b` (with arbitrary slope `a` and offset `b`),
    the second-order midpoint components balance identically (`R^{+,2}_x(u) = R^{-,2}_x(u)`)
    and the second-order gradient vanishes (`∇_x Φ₂(u) = 0`) at every pixel `x` and step `d`!
    By contrast, a first-order difference `u(x + d) - u(x) = a * d` is non-zero whenever `a * d ≠ 0`. -/
theorem second_order_linear_ramp_zero_staircasing (c : Nat → Int) (a b x d : Int) (idx : Nat) :
    let u := fun z : Int => a * z + b
    regPos2D1 c u x idx = regNeg2D1 c u x d idx ∧
    regGrad2D1 c u x d idx = 0 ∧
    (u (x + d) - u x = a * d) := by
  dsimp
  unfold regGrad2D1 regPos2D1 regNeg2D1
  have h_mid : (a * (x + d) + b) + (a * (x - d) + b) = 2 * (a * x + b) := by
    simp only [Int.mul_add, Int.mul_sub]
    omega
  have h_eq : 2 * c idx * (a * x + b) = c idx * ((a * (x + d) + b) + (a * (x - d) + b)) := by
    rw [h_mid]
    ac_rfl
  refine ⟨h_eq, ?_, ?_⟩
  · change 2 * c idx * (a * x + b) - c idx * ((a * (x + d) + b) + (a * (x - d) + b)) = 0
    rw [h_eq, Int.sub_self]
  · simp only [Int.mul_add]
    omega

-- ============================================================================
-- 2. Canonical Jordan Decomposition `q = q⁺ - q⁻` for Signed Dual Subgradients
-- ============================================================================

/-- Non-negative positive part `q⁺ = max(0, q)` of a signed dual Bregman variable `q`. -/
def posPart (q : Int) : Int :=
  if 0 ≤ q then q else 0

/-- Non-negative negative part `q⁻ = max(0, -q)` of a signed dual Bregman variable `q`. -/
def negPart (q : Int) : Int :=
  if q ≤ 0 then -q else 0

/-- Theorem 7a (Canonical Jordan Decomposition Properties):
    For every integer `q : Int`:
      1. `0 ≤ posPart q` and `0 ≤ negPart q` (unconditional non-negativity),
      2. `posPart q - negPart q = q` (exact signed reconstruction),
      3. `posPart q * negPart q = 0` (complementarity / orthogonality). -/
theorem jordan_split_properties (q : Int) :
    0 ≤ posPart q ∧
    0 ≤ negPart q ∧
    posPart q - negPart q = q ∧
    posPart q * negPart q = 0 := by
  unfold posPart negPart
  by_cases h1 : 0 ≤ q
  · by_cases h2 : q ≤ 0
    · have hq0 : q = 0 := by omega
      simp [hq0]
    · simp [h1, h2]
  · have h2 : q ≤ 0 := by omega
    simp [h1, h2]

-- ============================================================================
-- 3. Primal-Dual TGV-Bregman Split-EM Positivity & Exact Contrast Restoration
-- ============================================================================

/-- Denominator of the TGV-Bregman Split-EM update at pixel `j`:
    `D_j = w_j + λ₁ R^{+,1}_j + λ₂ R^{+,2}_j + q⁻_j`. -/
def tgvBregmanDenom (w_j lam1 r1Pos_j lam2 r2Pos_j q_j : Int) : Int :=
  w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j

/-- Numerator of the TGV-Bregman Split-EM update at pixel `j`:
    `N_j = u_j * ((H^T r)_j + λ₁ R^{-,1}_j + λ₂ R^{-,2}_j + q⁺_j)`. -/
def tgvBregmanNum (u_j bproj_j lam1 r1Neg_j lam2 r2Neg_j q_j : Int) : Int :=
  u_j * (bproj_j + lam1 * r1Neg_j + lam2 * r2Neg_j + posPart q_j)

/-- Theorem 7b (Unconditional Positivity of TGV-Bregman Split-EM Update):
    For ANY signed dual Bregman subgradient `q_j ∈ ℤ` and any `λ₁, λ₂ ≥ 0`,
    whenever `w_j > 0`, `u_j ≥ 0`, `bproj_j ≥ 0`, and `r1Pos, r1Neg, r2Pos, r2Neg ≥ 0`,
    the denominator is strictly positive (`≥ w_j > 0`), the numerator is non-negative (`≥ 0`),
    and any update `u'_j` satisfying `tgvBregmanDenom * u'_j = tgvBregmanNum` satisfies `0 ≤ u'_j`! -/
theorem tgv_bregman_unconditional_nonneg
    (w_j u_j bproj_j lam1 r1Pos_j r1Neg_j lam2 r2Pos_j r2Neg_j q_j u'_j : Int)
    (hw : 0 < w_j) (hu : 0 ≤ u_j) (hbproj : 0 ≤ bproj_j)
    (hlam1 : 0 ≤ lam1) (hr1p : 0 ≤ r1Pos_j) (hr1n : 0 ≤ r1Neg_j)
    (hlam2 : 0 ≤ lam2) (hr2p : 0 ≤ r2Pos_j) (hr2n : 0 ≤ r2Neg_j)
    (h_step : tgvBregmanDenom w_j lam1 r1Pos_j lam2 r2Pos_j q_j * u'_j =
              tgvBregmanNum u_j bproj_j lam1 r1Neg_j lam2 r2Neg_j q_j) :
    0 < tgvBregmanDenom w_j lam1 r1Pos_j lam2 r2Pos_j q_j ∧ 0 ≤ u'_j := by
  unfold tgvBregmanDenom tgvBregmanNum at *
  have h_jp := jordan_split_properties q_j
  have h_p1 : 0 ≤ lam1 * r1Pos_j := Int.mul_nonneg hlam1 hr1p
  have h_p2 : 0 ≤ lam2 * r2Pos_j := Int.mul_nonneg hlam2 hr2p
  have h_n1 : 0 ≤ lam1 * r1Neg_j := Int.mul_nonneg hlam1 hr1n
  have h_n2 : 0 ≤ lam2 * r2Neg_j := Int.mul_nonneg hlam2 hr2n
  have hd_pos : 0 < w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j := by omega
  have h_inner_nonneg : 0 ≤ bproj_j + lam1 * r1Neg_j + lam2 * r2Neg_j + posPart q_j := by omega
  have hn_nonneg : 0 ≤ u_j * (bproj_j + lam1 * r1Neg_j + lam2 * r2Neg_j + posPart q_j) :=
    Int.mul_nonneg hu h_inner_nonneg
  refine ⟨hd_pos, ?_⟩
  by_cases hu'_neg : u'_j < 0
  · have h_lhs_neg : (w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j) * u'_j < 0 :=
      Int.mul_neg_of_pos_of_neg hd_pos hu'_neg
    omega
  · omega

/-- Theorem 7c (Exact Primal-Dual TGV-Bregman KKT Stationarity Equivalence):
    A pixel `j` is a fixed point of the TGV-Bregman Split-EM update
      `tgvBregmanDenom * u_j = tgvBregmanNum`
    if and only if:
      `u_j = 0  ∨  (bproj_j - w_j) + q_j = λ₁ * (r1Pos_j - r1Neg_j) + λ₂ * (r2Pos_j - r2Neg_j)`. -/
theorem tgv_bregman_kkt_stationarity_iff
    (w_j u_j bproj_j lam1 r1Pos_j r1Neg_j lam2 r2Pos_j r2Neg_j q_j : Int) :
    tgvBregmanDenom w_j lam1 r1Pos_j lam2 r2Pos_j q_j * u_j =
    tgvBregmanNum u_j bproj_j lam1 r1Neg_j lam2 r2Neg_j q_j ↔
    (u_j = 0 ∨ (bproj_j - w_j) + q_j = lam1 * (r1Pos_j - r1Neg_j) + lam2 * (r2Pos_j - r2Neg_j)) := by
  unfold tgvBregmanDenom tgvBregmanNum
  have h_jp := (jordan_split_properties q_j).2.2.1
  have h_comm : (w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j) * u_j =
                u_j * (w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j) := Int.mul_comm _ _
  rw [h_comm]
  constructor
  · intro h
    have h_factor : u_j * ((bproj_j + lam1 * r1Neg_j + lam2 * r2Neg_j + posPart q_j) -
                           (w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j)) = 0 := by
      rw [Int.mul_sub, ← h, Int.sub_self]
    cases Int.mul_eq_zero.mp h_factor with
    | inl hu0 => exact Or.inl hu0
    | inr hgrad =>
      apply Or.inr
      simp only [Int.mul_sub]
      omega
  · intro h
    cases h with
    | inl hu0 =>
      rw [hu0, Int.zero_mul, Int.zero_mul]
    | inr hgrad =>
      simp only [Int.mul_sub] at hgrad
      have heq : w_j + lam1 * r1Pos_j + lam2 * r2Pos_j + negPart q_j =
                 bproj_j + lam1 * r1Neg_j + lam2 * r2Neg_j + posPart q_j := by
        omega
      rw [heq]

/-- Theorem 7d (Zero Peak-Clipping Bias / Exact Contrast Restoration Theorem):
    At an exact solution `H u* = y` (where `bproj_j = w_j`), standard regularized RL (`q_j = 0`)
    fails to have `u*` as a fixed point whenever `λ₁ ∇Φ₁(u*) + λ₂ ∇Φ₂(u*) ≠ 0` (clipping star peaks!).
    With Bregman subgradient matching `q_j = λ₁ (r1Pos_j - r1Neg_j) + λ₂ (r2Pos_j - r2Neg_j)`,
    `u*` is an EXACT fixed point for ALL regularization weights `λ₁, λ₂ ≥ 0`! -/
theorem bregman_exact_contrast_restoration_fixed_point
    (w_j u_j bproj_j lam1 r1Pos_j r1Neg_j lam2 r2Pos_j r2Neg_j q_j : Int)
    (h_exact_data : bproj_j = w_j)
    (h_bregman_subgrad : q_j = lam1 * (r1Pos_j - r1Neg_j) + lam2 * (r2Pos_j - r2Neg_j)) :
    tgvBregmanDenom w_j lam1 r1Pos_j lam2 r2Pos_j q_j * u_j =
    tgvBregmanNum u_j bproj_j lam1 r1Neg_j lam2 r2Neg_j q_j := by
  rw [tgv_bregman_kkt_stationarity_iff]
  apply Or.inr
  omega

-- ============================================================================
-- 4. Reference-Centered Primal Bregman Split Operator `B⁺(u, u_ref), B⁻(u, u_ref)`
-- ============================================================================

/-- Primal Reference-Centered Bregman repulsive part:
    `B⁺_j(u, u_ref) = R⁺_j(u) + R⁻_j(u_ref) ≥ 0`. -/
def bregmanRefPos (c : Nat → Nat → Int) (n : Nat) (u uRef : Nat → Int) (j : Nat) : Int :=
  regPos c n u j + regNeg c n uRef j

/-- Primal Reference-Centered Bregman attractive part:
    `B⁻_j(u, u_ref) = R⁻_j(u) + R⁺_j(u_ref) ≥ 0`. -/
def bregmanRefNeg (c : Nat → Nat → Int) (n : Nat) (u uRef : Nat → Int) (j : Nat) : Int :=
  regNeg c n u j + regPos c n uRef j

/-- Theorem 7e (Exact Global Flux Balance & Self-Debiasing of Reference-Centered Bregman):
    For any symmetric conductivity matrix `c_{j,l} = c_{l,j}`:
      1. `∑_{j=0}^{n-1} B⁺_j(u, u_ref) = ∑_{j=0}^{n-1} B⁻_j(u, u_ref)` for all `u, u_ref`,
      2. When `u_ref = u`, `B⁺_j(u, u) = B⁻_j(u, u)` pointwise at every pixel `j < n`! -/
theorem bregman_ref_flux_balance_and_debiasing (c : Nat → Nat → Int) (n : Nat)
    (u uRef : Nat → Int)
    (h_symm : ∀ j l, j < n → l < n → c j l = c l j) :
    sumRange n (bregmanRefPos c n u uRef) = sumRange n (bregmanRefNeg c n u uRef) ∧
    (∀ j, bregmanRefPos c n u u j = bregmanRefNeg c n u u j) := by
  unfold bregmanRefPos bregmanRefNeg
  constructor
  · rw [sumRange_add, sumRange_add]
    rw [split_reg_flux_balance c n u h_symm]
    rw [split_reg_flux_balance c n uRef h_symm]
  · intro j
    omega

-- ============================================================================
-- 5. Neumann-Series Preconditioned Backprojector `(I + α(I - H^T H)) H^T`
-- ============================================================================

/-- Neumann-preconditioned backprojection at pixel `j`:
    Given `g_j = (H^T r)_j - w_j`, applies `(I + α(I - H^T H))` to accelerate
    mid/high spatial frequency recovery:
      `neumannBproj_j = w_j + g_j + α * (g_j - (H^T (H g))_j)`. -/
def neumannBackproject (H : Nat → Nat → Int) (m n : Nat) (r : Nat → Int)
    (alpha : Int) (j : Nat) : Int :=
  let g := fun k => backproject H m r k - sensitivity H m k
  let hg := backproject H m (forward H n g) j
  sensitivity H m j + g j + alpha * (g j - hg)

/-- Theorem 8a (Exact Fixed-Point Invariance of Neumann-Preconditioned Backprojection):
    At any exact solution `H u* = y` (where `r_i = 1` for all `i < m`),
    `neumannBackproject H m n r alpha j = sensitivity H m j` for ALL `α ∈ ℤ`! -/
theorem neumann_backproject_fixed_point_invariance (H : Nat → Nat → Int) (m n : Nat)
    (r : Nat → Int) (alpha : Int) (j : Nat)
    (hr : ∀ i, i < m → r i = 1) :
    neumannBackproject H m n r alpha j = sensitivity H m j := by
  unfold neumannBackproject
  let g := fun k => backproject H m r k - sensitivity H m k
  have hg0 : ∀ k, g k = 0 := by
    intro k
    dsimp [g]
    rw [backproject_const_ratio H m r 1 k hr]
    omega
  have h_fwd0 : ∀ i, forward H n g i = 0 := by
    intro i
    unfold forward
    have h0 : sumRange n (fun k => H i k * g k) =
              sumRange n (fun _ => 0) := by
      apply sumRange_congr
      intro k _
      rw [hg0 k, Int.mul_zero]
    rw [h0, sumRange_zero]
  have h_bproj0 : backproject H m (forward H n g) j = 0 := by
    change sumRange m (fun i => H i j * forward H n g i) = 0
    have h0 : sumRange m (fun i => H i j * forward H n g i) =
              sumRange m (fun _ => 0) := by
      apply sumRange_congr
      intro i _
      rw [h_fwd0 i, Int.mul_zero]
    rw [h0, sumRange_zero]
  change sensitivity H m j + g j + alpha * (g j - backproject H m (forward H n g) j) = sensitivity H m j
  rw [hg0 j, h_bproj0]
  omega

/-- Theorem 8b (Zero Net DC Flux Perturbation of the Neumann Correction `g - H^T (H g)`):
    When `H` is doubly normalized (`∑_{i=0}^{m-1} H_{i,j} = 1` and `∑_{j=0}^{n-1} H_{i,j} = 1`,
    as holds for any normalized shift-invariant PSF), for ANY residual vector `g`,
      `∑_{j=0}^{n-1} (H^T (H g))_j = ∑_{j=0}^{n-1} g_j`,
    and therefore the Neumann high-frequency boost `g - H^T (H g)` has IDENTICALLY ZERO sum:
      `∑_{j=0}^{n-1} (g_j - (H^T (H g))_j) = 0`! -/
theorem neumann_boost_zero_net_flux (H : Nat → Nat → Int) (m n : Nat) (g : Nat → Int)
    (h_col : ∀ j, j < n → sensitivity H m j = 1)
    (h_row : ∀ i, i < m → sumRange n (fun j => H i j) = 1) :
    sumRange n (fun j => g j - backproject H m (forward H n g) j) = 0 := by
  rw [sumRange_sub]
  -- Show sum_j (H^T (H g))_j = sum_i (H g)_i * 1 = sum_j g_j
  have h_adj : sumRange n (fun j => backproject H m (forward H n g) j) =
               sumRange n (fun j => 1 * backproject H m (forward H n g) j) := by
    apply sumRange_congr
    intro j _
    omega
  rw [h_adj, adjoint_inner_product H m n (fun _ => 1) (forward H n g)]
  have h_ones : ∀ i, i < m → forward H n (fun _ => 1) i = 1 := by
    intro i hi
    unfold forward
    have h1 : sumRange n (fun j => H i j * 1) = sumRange n (fun j => H i j) := by
      apply sumRange_congr
      intro j _
      omega
    rw [h1, h_row i hi]
  have h_rhs : sumRange m (fun i => forward H n g i * forward H n (fun _ => 1) i) =
               sumRange m (fun i => forward H n g i) := by
    apply sumRange_congr
    intro i hi
    rw [h_ones i hi, Int.mul_one]
  rw [h_rhs, forward_unit_psf_flux H m n g h_col, Int.sub_self]

end Lucky
