/-
  Lucky/Improved.lean
  Formal Verification of Our Improved Lucy-Richardson Deconvolution Algorithm (PCLR):
  Positivity & Conservation-Preserving Accelerated Split-EM Lucy-Richardson.

  Key Formal Theorems:
    1. Positivity Breakdown of Standard Unsplit TV-RL (Dey et al. 2006) for large `λ`
    2. Split-EM Anisotropic Edge-Preserving Regularizer (`R⁺`, `R⁻`):
       - Non-negativity of split components
       - Zero net flux (`∑ R⁺ = ∑ R⁻`, `∑ ∇Φ = 0`)
       - Flat-field invariance (`∇Φ(c) = 0`)
       - Discrete Green's First Identity (`2 ∑ u_j ∇_j Φ(u) = ∑_{j,l} c_{j,l} (u_j - u_l)² ≥ 0`)
    3. Unconditional Positivity & Exact MAP KKT Stationarity of Split-EM Regularized RL
    4. Readout-Noise / Background Damping (`(y_i + b) / ((H u)_i + b)`):
       - Division-by-zero immunity (`(H u)_i + b ≥ b > 0`)
       - Exact fixed-point invariance
       - Strict attenuation of noise spike ratios
    5. Monotone-Safeguarded Accelerated Iteration (Nesterov / Biggs-Andrews):
       - Guaranteed descent at least as fast as unaccelerated RL (`E(G) ≤ E(T(u)) ≤ E(u)`)
       - Strict per-step gain when extrapolation succeeds (`E(A) < E(T(u)) → E(G) < E(T(u))`)
       - Positivity preservation across accelerated trajectories
-/

import Lucky.VecMat
import Lucky.Basic
import Lucky.Divergence

namespace Lucky

-- ============================================================================
-- 1. Why Standard Unsplit TV-RL (Dey et al. 2006) Violates Positivity
-- ============================================================================

/-- In Dey et al. (2006) TV-RL, the update is `(w_j - λ * div_j) * u'_j = u_j * (H^T r)_j`.
    Theorem 1 proves that whenever `λ * div_j > w_j` at a positive pixel (`u_j > 0`, `(H^T r)_j > 0`),
    any solution `u'_j` to the unsplit TV-RL update is forced to be STRICTLY NEGATIVE (`u'_j < 0`),
    violating physical photon non-negativity! -/
theorem dey_tv_rl_positivity_breakdown (w_j lam div_j u_j bproj_j u'_j : Int)
    (hu : 0 < u_j) (hbproj : 0 < bproj_j)
    (h_overreg : w_j < lam * div_j)
    (h_dey_step : (w_j - lam * div_j) * u'_j = u_j * bproj_j) :
    u'_j < 0 := by
  have h_rhs_pos : 0 < u_j * bproj_j := Int.mul_pos hu hbproj
  have h_denom_neg : w_j - lam * div_j < 0 := by omega
  by_cases h_nonneg : 0 ≤ u'_j
  · have h_lhs_nonpos : (w_j - lam * div_j) * u'_j ≤ 0 :=
      Int.mul_nonpos_of_nonpos_of_nonneg (Int.le_of_lt h_denom_neg) h_nonneg
    omega
  · omega

-- ============================================================================
-- 2. Split-EM Edge-Preserving Anisotropic Diffusion Regularizer (`R⁺` and `R⁻`)
-- ============================================================================

/-- Repulsive (diagonal / self) part of the anisotropic diffusion gradient:
    `R⁺_j(u) = ∑_{l=0}^{n-1} c_{j,l} u_j`. -/
def regPos (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (j : Nat) : Int :=
  sumRange n (fun l => c j l * u j)

/-- Attractive (neighborhood smoothing) part of the anisotropic diffusion gradient:
    `R⁻_j(u) = ∑_{l=0}^{n-1} c_{j,l} u_l`. -/
def regNeg (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (j : Nat) : Int :=
  sumRange n (fun l => c j l * u l)

/-- Full anisotropic regularizer gradient:
    `∇_j Φ(u) = R⁺_j(u) - R⁻_j(u) = ∑_{l=0}^{n-1} c_{j,l} (u_j - u_l)`. -/
def regGrad (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (j : Nat) : Int :=
  regPos c n u j - regNeg c n u j

/-- Theorem 2a: Pointwise difference expansion of `∇_j Φ(u)`. -/
theorem regGrad_eq_sum_diff (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (j : Nat) :
    regGrad c n u j = sumRange n (fun l => c j l * (u j - u l)) := by
  unfold regGrad regPos regNeg
  rw [← sumRange_sub]
  apply sumRange_congr
  intro l _
  rw [Int.mul_sub]

/-- Theorem 2b: Non-negativity of both split components `R⁺_j(u) ≥ 0` and `R⁻_j(u) ≥ 0`
    whenever edge weights `c_{j,l} ≥ 0` and intensities `u_l ≥ 0`. -/
theorem split_reg_nonneg (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (j : Nat)
    (hc : ∀ l, l < n → 0 ≤ c j l) (hu : ∀ l, l < n → 0 ≤ u l) (hj : j < n) :
    0 ≤ regPos c n u j ∧ 0 ≤ regNeg c n u j := by
  constructor
  · unfold regPos
    apply sumRange_nonneg
    intro l hl
    exact Int.mul_nonneg (hc l hl) (hu j hj)
  · unfold regNeg
    apply sumRange_nonneg
    intro l hl
    exact Int.mul_nonneg (hc l hl) (hu l hl)

/-- Theorem 2c (Exact Global Flux Balance of Split Regularizer):
    For any symmetric conductivity / edge-stopping matrix `c_{j,l} = c_{l,j}`,
      `∑_{j=0}^{n-1} R⁺_j(u) = ∑_{j=0}^{n-1} R⁻_j(u)`. -/
theorem split_reg_flux_balance (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (h_symm : ∀ j l, j < n → l < n → c j l = c l j) :
    sumRange n (regPos c n u) = sumRange n (regNeg c n u) := by
  unfold regPos regNeg
  rw [sumRange_comm n n (fun j l => c j l * u j)]
  apply sumRange_congr
  intro j hj
  apply sumRange_congr
  intro l hl
  rw [h_symm l j hl hj]

/-- Corollary 2d (Zero Net Flux of the Regularization Gradient):
    For any symmetric edge-stopping matrix `c_{j,l} = c_{l,j}`,
      `∑_{j=0}^{n-1} ∇_j Φ(u) = 0`. -/
theorem split_reg_zero_net_gradient (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (h_symm : ∀ j l, j < n → l < n → c j l = c l j) :
    sumRange n (regGrad c n u) = 0 := by
  unfold regGrad
  rw [sumRange_sub, split_reg_flux_balance c n u h_symm, Int.sub_self]

/-- Theorem 2e (Constant / Flat Image Annihilation):
    On any constant image `u_l = C`, `R⁺_j(u) = R⁻_j(u)` and `∇_j Φ(u) = 0`. -/
theorem split_reg_constant_zero (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (C : Int) (j : Nat) (hj : j < n) (hu : ∀ l, l < n → u l = C) :
    regGrad c n u j = 0 := by
  rw [regGrad_eq_sum_diff]
  have h0 : sumRange n (fun l => c j l * (u j - u l)) = sumRange n (fun _ => 0) := by
    apply sumRange_congr
    intro l hl
    rw [hu j hj, hu l hl, Int.sub_self, Int.mul_zero]
  rw [h0, sumRange_zero]

/-- Algebraic helper for Discrete Green's First Identity:
    `c * a * (a - b) + c * b * (b - a) = c * ((a - b) * (a - b))`. -/
theorem green_term_identity (c a b : Int) :
    c * a * (a - b) + c * b * (b - a) = c * ((a - b) * (a - b)) := by
  have h1 : c * a * (a - b) = c * (a * (a - b)) := by ac_rfl
  have h2 : c * b * (b - a) = c * (b * (b - a)) := by ac_rfl
  have h3 : a * (a - b) + b * (b - a) = (a - b) * (a - b) := by
    simp only [Int.mul_sub, Int.sub_mul]
    have hcomm : a * b = b * a := Int.mul_comm a b
    omega
  rw [h1, h2, ← Int.mul_add, h3]

/-- Theorem 2f (Discrete Green's First Identity / Dirichlet Energy Identity):
    For any symmetric edge-stopping matrix `c_{j,l} = c_{l,j}`,
      `2 * ∑_{j=0}^{n-1} u_j * ∇_j Φ(u) = ∑_{j=0}^{n-1} ∑_{l=0}^{n-1} c_{j,l} * (u_j - u_l)^2`. -/
theorem discrete_greens_first_identity (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (h_symm : ∀ j l, j < n → l < n → c j l = c l j) :
    2 * sumRange n (fun j => u j * regGrad c n u j) =
    sumRange n (fun j => sumRange n (fun l => c j l * ((u j - u l) * (u j - u l)))) := by
  have h_half1 : sumRange n (fun j => u j * regGrad c n u j) =
                 sumRange n (fun j => sumRange n (fun l => c j l * u j * (u j - u l))) := by
    apply sumRange_congr
    intro j _
    rw [regGrad_eq_sum_diff, ← sumRange_mul_left]
    apply sumRange_congr
    intro l _
    ac_rfl
  have h_half2 : sumRange n (fun j => u j * regGrad c n u j) =
                 sumRange n (fun j => sumRange n (fun l => c j l * u l * (u l - u j))) := by
    rw [h_half1, sumRange_comm]
    apply sumRange_congr
    intro j hj
    apply sumRange_congr
    intro l hl
    rw [h_symm l j hl hj]
  have h_two : 2 * sumRange n (fun j => u j * regGrad c n u j) =
               sumRange n (fun j => u j * regGrad c n u j) +
               sumRange n (fun j => u j * regGrad c n u j) := by
    omega
  rw [h_two]
  conv =>
    lhs
    lhs
    rw [h_half1]
  conv =>
    lhs
    rhs
    rw [h_half2]
  rw [← sumRange_add]
  apply sumRange_congr
  intro j _
  rw [← sumRange_add]
  apply sumRange_congr
  intro l _
  exact green_term_identity (c j l) (u j) (u l)

/-- Corollary 2g (Dissipativity / Convexity of the Split Regularizer):
    When `c_{j,l} ≥ 0` and `c_{j,l} = c_{l,j}`,
      `0 ≤ 2 * ∑_{j=0}^{n-1} u_j * ∇_j Φ(u)`. -/
theorem split_reg_dissipative_nonneg (c : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (h_symm : ∀ j l, j < n → l < n → c j l = c l j)
    (hc : ∀ j l, j < n → l < n → 0 ≤ c j l) :
    0 ≤ 2 * sumRange n (fun j => u j * regGrad c n u j) := by
  rw [discrete_greens_first_identity c n u h_symm]
  apply sumRange_nonneg
  intro j hj
  apply sumRange_nonneg
  intro l hl
  exact Int.mul_nonneg (hc j l hj hl) (int_sq_nonneg (u j - u l))

-- ============================================================================
-- 3. Unconditional Positivity & KKT Stationarity of Split-EM Regularized RL
-- ============================================================================

/-- Denominator of the Split-EM Regularized Lucy-Richardson update at pixel `j`:
    `D_j(u) = w_j + λ * R⁺_j(u)`. -/
def splitEmDenom (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u : Nat → Int) (j : Nat) : Int :=
  sensitivity H m j + lam * regPos c n u j

/-- Numerator of the Split-EM Regularized Lucy-Richardson update at pixel `j`:
    `N_j(u, r) = u_j * ((H^T r)_j + λ * R⁻_j(u))`. -/
def splitEmNum (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u r : Nat → Int) (j : Nat) : Int :=
  u j * (backproject H m r j + lam * regNeg c n u j)

/-- Theorem 3a (Unconditional Strict Positivity of the Split-EM Denominator):
    Unlike Dey et al.'s TV-RL denominator `w_j - λ div_j` (which can vanish or become negative),
    the Split-EM denominator `w_j + λ R⁺_j(u)` is ≥ `w_j > 0` for ALL `λ ≥ 0`! -/
theorem split_em_denom_pos (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u : Nat → Int) (j : Nat) (hj : j < n)
    (hw_pos : 0 < sensitivity H m j) (hlam : 0 ≤ lam)
    (hc : ∀ l, l < n → 0 ≤ c j l) (hu : ∀ l, l < n → 0 ≤ u l) :
    0 < splitEmDenom H c m n lam u j := by
  unfold splitEmDenom
  have h_rpos : 0 ≤ regPos c n u j := (split_reg_nonneg c n u j hc hu hj).1
  have h_prod : 0 ≤ lam * regPos c n u j := Int.mul_nonneg hlam h_rpos
  omega

/-- Theorem 3b (Unconditional Non-Negativity of the Split-EM Numerator):
    For all `λ ≥ 0`, `u ≥ 0`, `H ≥ 0`, `r ≥ 0`, `c ≥ 0`, `splitEmNum ≥ 0`. -/
theorem split_em_num_nonneg (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u r : Nat → Int) (j : Nat) (hj : j < n)
    (hlam : 0 ≤ lam)
    (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i)
    (hc : ∀ l, l < n → 0 ≤ c j l) (hu : ∀ l, l < n → 0 ≤ u l) :
    0 ≤ splitEmNum H c m n lam u r j := by
  unfold splitEmNum
  have h_bp : 0 ≤ backproject H m r j := backproject_nonneg H m r j hH hr
  have h_rneg : 0 ≤ regNeg c n u j := (split_reg_nonneg c n u j hc hu hj).2
  have h_prod : 0 ≤ lam * regNeg c n u j := Int.mul_nonneg hlam h_rneg
  exact Int.mul_nonneg (hu j hj) (by omega)

/-- Theorem 3c (Unconditional Positivity Preservation of Split-EM Update):
    Any update `u'_j` satisfying `splitEmDenom_j * u'_j = splitEmNum_j` (with `w_j > 0`)
    is guaranteed to be non-negative (`0 ≤ u'_j`) for EVERY regularization weight `λ ≥ 0`! -/
theorem split_em_update_unconditional_nonneg
    (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u r u' : Nat → Int) (j : Nat) (hj : j < n)
    (hw_pos : 0 < sensitivity H m j) (hlam : 0 ≤ lam)
    (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i)
    (hc : ∀ l, l < n → 0 ≤ c j l) (hu : ∀ l, l < n → 0 ≤ u l)
    (h_step : splitEmDenom H c m n lam u j * u' j = splitEmNum H c m n lam u r j) :
    0 ≤ u' j := by
  have hd_pos : 0 < splitEmDenom H c m n lam u j :=
    split_em_denom_pos H c m n lam u j hj hw_pos hlam hc hu
  have hn_nonneg : 0 ≤ splitEmNum H c m n lam u r j :=
    split_em_num_nonneg H c m n lam u r j hj hlam hH hr hc hu
  by_cases hu'_neg : u' j < 0
  · have h_lhs_neg : splitEmDenom H c m n lam u j * u' j < 0 :=
      Int.mul_neg_of_pos_of_neg hd_pos hu'_neg
    omega
  · omega

/-- Theorem 3d (Exact MAP Regularized KKT Stationarity Equivalence):
    A pixel `j` is stationary under the Split-EM Regularized RL update
      `splitEmDenom_j * u_j = splitEmNum_j`
    if and only if the exact regularized MAP Karush-Kuhn-Tucker condition holds:
      `u_j = 0  ∨  (H^T r)_j - w_j = λ * ∇_j Φ(u)`. -/
theorem split_em_kkt_stationarity_iff
    (H : Nat → Nat → Int) (c : Nat → Nat → Int) (m n : Nat)
    (lam : Int) (u r : Nat → Int) (j : Nat) :
    splitEmDenom H c m n lam u j * u j = splitEmNum H c m n lam u r j ↔
    (u j = 0 ∨ backproject H m r j - sensitivity H m j = lam * regGrad c n u j) := by
  unfold splitEmDenom splitEmNum regGrad
  have h_comm : (sensitivity H m j + lam * regPos c n u j) * u j =
                u j * (sensitivity H m j + lam * regPos c n u j) := Int.mul_comm _ _
  rw [h_comm]
  constructor
  · intro h
    have h_factor : u j * ((backproject H m r j + lam * regNeg c n u j) -
                           (sensitivity H m j + lam * regPos c n u j)) = 0 := by
      rw [Int.mul_sub, ← h, Int.sub_self]
    cases Int.mul_eq_zero.mp h_factor with
    | inl hu0 => exact Or.inl hu0
    | inr hgrad =>
      apply Or.inr
      rw [Int.mul_sub]
      omega
  · intro h
    cases h with
    | inl hu0 =>
      rw [hu0, Int.zero_mul, Int.zero_mul]
    | inr hgrad =>
      rw [Int.mul_sub] at hgrad
      have heq : sensitivity H m j + lam * regPos c n u j =
                 backproject H m r j + lam * regNeg c n u j := by
        omega
      rw [heq]

-- ============================================================================
-- 4. Readout-Noise / Background Offset Damping `(y_i + b) / ((H u)_i + b)`
-- ============================================================================

/-- Damped detector denominator `(H u)_i + b` with readout-noise / sky offset `b > 0`. -/
def dampedDenom (H : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (b : Int) (i : Nat) : Int :=
  forward H n u i + b

/-- Theorem 4a (Division-by-Zero Immunity):
    For any non-negative `H ≥ 0`, `u ≥ 0`, and positive background offset `b > 0`,
    the damped denominator `(H u)_i + b ≥ b > 0` is strictly positive even when `(H u)_i = 0`! -/
theorem damped_denom_strictly_pos (H : Nat → Nat → Int) (n : Nat) (u : Nat → Int)
    (b : Int) (i : Nat) (hb : 0 < b)
    (hH : ∀ j, j < n → 0 ≤ H i j) (hu : ∀ j, j < n → 0 ≤ u j) :
    b ≤ dampedDenom H n u b i ∧ 0 < dampedDenom H n u b i := by
  unfold dampedDenom
  have hf : 0 ≤ forward H n u i := forward_nonneg H n u i hH hu
  omega

/-- Theorem 4b (Exact Fixed-Point Preservation under Readout Damping):
    Adding offset `b` preserves exact solutions: `y_i + b = (H u)_i + b ↔ (H u)_i = y_i`. -/
theorem damped_exact_fixed_point (H : Nat → Nat → Int) (n : Nat) (u y : Nat → Int)
    (b : Int) (i : Nat) :
    y i + b = dampedDenom H n u b i ↔ forward H n u i = y i := by
  unfold dampedDenom
  omega

/-- Theorem 4c (Strict Noise-Spike Ratio Contraction Toward Unity):
    For any `y_i ≠ ŷ_i` with `ŷ_i > 0` and offset `b > 0`,
    the relative discrepancy of the damped ratio `(y_i + b) / (ŷ_i + b) - 1 = (y_i - ŷ_i) / (ŷ_i + b)`
    is strictly smaller in cross-multiplied magnitude than the undamped discrepancy `(y_i - ŷ_i) / ŷ_i`:
      `|y_i - ŷ_i| * ŷ_i < |y_i - ŷ_i| * (ŷ_i + b)`. -/
theorem damped_ratio_spike_attenuation (y_i yhat_i b : Int)
    (hne : y_i ≠ yhat_i) (_hyhat : 0 < yhat_i) (hb : 0 < b) :
    Int.natAbs (y_i - yhat_i) * yhat_i < Int.natAbs (y_i - yhat_i) * (yhat_i + b) := by
  have h_abs_pos : (0 : Int) < Int.natAbs (y_i - yhat_i) := by
    omega
  have h_diff : Int.natAbs (y_i - yhat_i) * (yhat_i + b) - Int.natAbs (y_i - yhat_i) * yhat_i =
                Int.natAbs (y_i - yhat_i) * b := by
    rw [Int.mul_add]
    omega
  have h_prod_pos : (0 : Int) < Int.natAbs (y_i - yhat_i) * b := Int.mul_pos h_abs_pos hb
  omega

/-- Theorem 4d (Exact Flux Balance Law of Readout-Damped Lucy-Richardson):
    When the damped ratio satisfies `r_i * ((H u)_i + b) = S * (y_i + b)` for all `i < m`,
    the unnormalized RL step plus the background backprojection satisfies the exact conservation law:
      `∑_{j=0}^{n-1} (rlStep H m u r)_j + b * ∑_{i=0}^{m-1} r_i = S * ∑_{i=0}^{m-1} (y_i + b)`.
    This reveals why raw background damping drifts total flux whenever `∑ r_i ≠ S * m`,
    motivating our explicit Flux-Rescaling Projection in PCLR! -/
theorem damped_rl_flux_balance_law (H : Nat → Nat → Int) (m n : Nat)
    (u y r : Nat → Int) (b S : Int)
    (h_damped : ∀ i, i < m → r i * dampedDenom H n u b i = S * (y i + b)) :
    sumRange n (rlStep H m u r) + b * sumRange m r =
    S * sumRange m (fun i => y i + b) := by
  unfold rlStep
  rw [adjoint_inner_product H m n u r]
  rw [← sumRange_mul_left m b r]
  rw [← sumRange_add]
  rw [← sumRange_mul_left m S (fun i => y i + b)]
  apply sumRange_congr
  intro i hi
  have h_lhs : r i * forward H n u i + b * r i = r i * dampedDenom H n u b i := by
    unfold dampedDenom
    rw [Int.mul_add]
    have hcomm : b * r i = r i * b := Int.mul_comm b (r i)
    omega
  rw [h_lhs, h_damped i hi]

-- ============================================================================
-- 5. Monotone-Safeguarded Accelerated Lucy-Richardson (Nesterov / Biggs-Andrews)
-- ============================================================================

/-- Safeguarded Accelerated Step:
    Given objective functional `E`, base Split-EM RL step `T`, and extrapolated
    Nesterov / Biggs-Andrews proposal `accelProposal`, accepts `accelProposal` iff
    it achieves an objective value `≤ E(T u)`; otherwise falls back to `T u`. -/
def guardedAccelStep (E : (Nat → Int) → Int)
    (baseStep accelProposal : Nat → Int) : Nat → Int :=
  if E accelProposal ≤ E baseStep then accelProposal else baseStep

/-- Theorem 5a (Superior One-Step Descent over Unaccelerated RL):
    The safeguarded accelerated step ALWAYS achieves an objective value
    at least as low as the unaccelerated step `baseStep`:
      `E(guardedAccelStep E baseStep accelProposal) ≤ E(baseStep)`. -/
theorem guarded_accel_beats_base (E : (Nat → Int) → Int)
    (baseStep accelProposal : Nat → Int) :
    E (guardedAccelStep E baseStep accelProposal) ≤ E baseStep := by
  unfold guardedAccelStep
  split
  · assumption
  · exact Int.le_refl (E baseStep)

/-- Theorem 5b (Strict Acceleration Gain):
    Whenever the extrapolated proposal improves upon the base step (`E(accelProposal) < E(baseStep)`),
    the safeguarded accelerated update achieves STRICTLY lower objective than standard unaccelerated RL! -/
theorem guarded_accel_strict_gain (E : (Nat → Int) → Int)
    (baseStep accelProposal : Nat → Int)
    (h_strict : E accelProposal < E baseStep) :
    E (guardedAccelStep E baseStep accelProposal) < E baseStep := by
  unfold guardedAccelStep
  split
  · exact h_strict
  · omega

/-- Theorem 5c (Global Monotone Descent Guarantee):
    If the base Split-EM RL step satisfies `E(baseStep) ≤ E(u)`, then the safeguarded
    accelerated step is unconditionally monotone decreasing:
      `E(guardedAccelStep E baseStep accelProposal) ≤ E(u)`,
    even if the raw Nesterov extrapolation `accelProposal` overshoots! -/
theorem guarded_accel_monotone_descent (E : (Nat → Int) → Int)
    (u baseStep accelProposal : Nat → Int)
    (h_base_descent : E baseStep ≤ E u) :
    E (guardedAccelStep E baseStep accelProposal) ≤ E u := by
  have h1 := guarded_accel_beats_base E baseStep accelProposal
  omega

/-- Theorem 5d (Positivity Preservation of Safeguarded Acceleration):
    If both `baseStep` and the clamped extrapolation `accelProposal` are non-negative,
    then `guardedAccelStep` is non-negative at every pixel `j < n`. -/
theorem guarded_accel_nonneg (E : (Nat → Int) → Int)
    (baseStep accelProposal : Nat → Int) (n : Nat)
    (hbase : ∀ j, j < n → 0 ≤ baseStep j)
    (haccel : ∀ j, j < n → 0 ≤ accelProposal j) :
    ∀ j, j < n → 0 ≤ guardedAccelStep E baseStep accelProposal j := by
  intro j hj
  unfold guardedAccelStep
  split
  · exact haccel j hj
  · exact hbase j hj

/-- Multi-step trajectory of the Monotone-Safeguarded Accelerated PCLR algorithm
    tracking `(current_iterate, previous_iterate)`. -/
def guardedAccelState (E : (Nat → Int) → Int)
    (T : (Nat → Int) → (Nat → Int))
    (A : (Nat → Int) → (Nat → Int) → (Nat → Int))
    (u0 : Nat → Int) : Nat → (Nat → Int) × (Nat → Int)
  | 0 => (u0, u0)
  | k + 1 =>
    let (uk, uPrev) := guardedAccelState E T A u0 k
    (guardedAccelStep E (T uk) (A uk uPrev), uk)

/-- The `k`-th image iterate of the safeguarded accelerated PCLR trajectory. -/
def guardedAccelTrajectory (E : (Nat → Int) → Int)
    (T : (Nat → Int) → (Nat → Int))
    (A : (Nat → Int) → (Nat → Int) → (Nat → Int))
    (u0 : Nat → Int) (k : Nat) : Nat → Int :=
  (guardedAccelState E T A u0 k).1

/-- Theorem 5e (Trajectory-Wide Monotone Objective Descent):
    If the base Split-EM RL operator `T` never increases `E` (`∀ u, E(T u) ≤ E u`),
    then along the entire safeguarded accelerated trajectory `u^{(k)}`,
    every step decreases `E` at least as much as `T` and satisfies `E(u^{(k)}) ≤ E(u^{(0)})`! -/
theorem guarded_trajectory_monotone_descent (E : (Nat → Int) → Int)
    (T : (Nat → Int) → (Nat → Int))
    (A : (Nat → Int) → (Nat → Int) → (Nat → Int))
    (u0 : Nat → Int)
    (hT_descent : ∀ u, E (T u) ≤ E u) :
    ∀ k, E (guardedAccelTrajectory E T A u0 (k + 1)) ≤ E (T (guardedAccelTrajectory E T A u0 k)) ∧
         E (guardedAccelTrajectory E T A u0 (k + 1)) ≤ E (guardedAccelTrajectory E T A u0 k) ∧
         E (guardedAccelTrajectory E T A u0 k) ≤ E u0 := by
  intro k
  induction k with
  | zero =>
    simp only [guardedAccelTrajectory, guardedAccelState]
    have h_beat := guarded_accel_beats_base E (T u0) (A u0 u0)
    have h_base := hT_descent u0
    refine ⟨h_beat, by omega, Int.le_refl (E u0)⟩
  | succ t ih =>
    have h_beat : E (guardedAccelTrajectory E T A u0 (t + 1 + 1)) ≤
                  E (T (guardedAccelTrajectory E T A u0 (t + 1))) := by
      simp only [guardedAccelTrajectory, guardedAccelState]
      exact guarded_accel_beats_base E (T (guardedAccelState E T A u0 (t + 1)).1)
        (A (guardedAccelState E T A u0 (t + 1)).1 (guardedAccelState E T A u0 (t + 1)).2)
    have h_base := hT_descent (guardedAccelTrajectory E T A u0 (t + 1))
    have h_prev_u0 : E (guardedAccelTrajectory E T A u0 (t + 1)) ≤ E u0 := by
      omega
    refine ⟨h_beat, by omega, h_prev_u0⟩

end Lucky
