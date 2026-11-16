/-
  Lucky/Basic.lean
  Formal Verification of the Classic & Sensitivity-Weighted Lucy-Richardson (RL)
  Deconvolution Algorithm in Lean 4:
    1. Exact Photon / Flux Conservation (1-step, trajectory-wide, and denominator-cleared)
    2. Positivity & Strict Positivity Invariance
    3. Multiplicative Zero-Locking (Support Invariance)
    4. Exact Solution Fixed-Point Property
    5. KKT Complementarity & Stationarity Equivalence
-/

import Lucky.VecMat

namespace Lucky

/-- One step of the standard Lucy-Richardson update given detector ratio vector `r`:
    `u'_{j} = u_j * (H^T r)_j = u_j * ∑_{i=0}^{m-1} H_{i,j} r_i`. -/
def rlStep (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j : Nat) : Int :=
  u j * backproject H m r j

/-- Multi-step trajectory of Lucy-Richardson given a sequence of ratio vectors `rSeq k`. -/
def rlTrajectory (H : Nat → Nat → Int) (m : Nat) (u0 : Nat → Int)
    (rSeq : Nat → Nat → Int) : Nat → Nat → Int
  | 0 => u0
  | k + 1 => rlStep H m (rlTrajectory H m u0 rSeq k) (rSeq k)

-- ============================================================================
-- 1. Exact Photon / Flux Conservation Theorems
-- ============================================================================

/-- Theorem 1 (Fundamental Lucy-Richardson Flux Identity):
    For any `m × n` forward operator `H`, source estimate `u`, observed image `y`,
    and scale factor `S` (where `S = 1` in exact real/rational division `r_i = y_i / (H u)_i`,
    or `S = ∏_p (H u)_p` in cleared-denominator integer arithmetic),
    if `r_i * (H u)_i = S * y_i` for all `i < m`, then:
      `∑_{j=0}^{n-1} (rlStep H m u r)_j = S * ∑_{i=0}^{m-1} y_i`. -/
theorem rl_flux_conservation (H : Nat → Nat → Int) (m n : Nat) (u y r : Nat → Int) (S : Int)
    (h_ratio : ∀ i, i < m → r i * forward H n u i = S * y i) :
    sumRange n (rlStep H m u r) = S * sumRange m y := by
  unfold rlStep
  rw [adjoint_inner_product H m n u r]
  rw [← sumRange_mul_left m S y]
  apply sumRange_congr
  intro i hi
  exact h_ratio i hi

/-- Corollary 1a (Unscaled Exact Flux Conservation):
    When `r_i * (H u)_i = y_i` for all `i < m` (`S = 1`), a single Lucy-Richardson step
    conserves the exact total observed photon count:
      `∑_{j=0}^{n-1} u'_j = ∑_{i=0}^{m-1} y_i`,
    regardless of the initial sum `∑_{j=0}^{n-1} u_j`! -/
theorem rl_unit_flux_conservation (H : Nat → Nat → Int) (m n : Nat) (u y r : Nat → Int)
    (h_ratio : ∀ i, i < m → r i * forward H n u i = y i) :
    sumRange n (rlStep H m u r) = sumRange m y := by
  have h := rl_flux_conservation H m n u y r 1 (by
    intro i hi
    rw [Int.one_mul]
    exact h_ratio i hi)
  omega

/-- Theorem 1b (Sensitivity-Weighted Lucy-Richardson Flux Conservation):
    In windowed / boundary-truncated imaging where column sensitivity is `w_j = ∑_i H_{i,j}`,
    the sensitivity-weighted RL update satisfies `w_j * u'_j = u_j * (H^T r)_j`.
    Then the predicted detector flux `∑_{i=0}^{m-1} (H u')_i = ∑_{j=0}^{n-1} w_j u'_j`
    equals `S * ∑_{i=0}^{m-1} y_i` for ANY matrix `H` (without requiring `w_j = 1`)! -/
theorem rl_weighted_reblurred_flux_conservation (H : Nat → Nat → Int) (m n : Nat)
    (u u' y r : Nat → Int) (S : Int)
    (h_step : ∀ j, j < n → sensitivity H m j * u' j = u j * backproject H m r j)
    (h_ratio : ∀ i, i < m → r i * forward H n u i = S * y i) :
    sumRange m (forward H n u') = S * sumRange m y := by
  rw [forward_flux_identity H m n u']
  have h_congr : sumRange n (fun j => sensitivity H m j * u' j) =
                 sumRange n (rlStep H m u r) := by
    apply sumRange_congr
    intro j hj
    exact h_step j hj
  rw [h_congr]
  exact rl_flux_conservation H m n u y r S h_ratio

/-- Theorem 1c (Multi-Step Trajectory Flux Invariance):
    Along any `K`-step Lucy-Richardson trajectory, for every iteration `k ≥ 1`,
    total flux equals `∑_{i=0}^{m-1} y_i`. -/
theorem rl_trajectory_flux_conservation (H : Nat → Nat → Int) (m n : Nat)
    (u0 y : Nat → Int) (rSeq : Nat → Nat → Int) (k : Nat)
    (h_ratio : ∀ t i, i < m → rSeq t i * forward H n (rlTrajectory H m u0 rSeq t) i = y i) :
    sumRange n (rlTrajectory H m u0 rSeq (k + 1)) = sumRange m y := by
  simp only [rlTrajectory]
  exact rl_unit_flux_conservation H m n (rlTrajectory H m u0 rSeq k) y (rSeq k) (h_ratio k)

/-- Theorem 1d (Division-Free Universal Polynomial Flux Identity for 2×2 Systems):
    Clearing denominators by `ŷ₀ * ŷ₁` where `ŷ_i = (H u)_i`, the un-divided RL numerators
      `N_j = u_j * (h_{0,j} * y_0 * ŷ_1 + h_{1,j} * y_1 * ŷ_0)`
    satisfy the unconditional polynomial identity
      `N_0 + N_1 = (y_0 + y_1) * ŷ_0 * ŷ_1`
    over ANY commutative ring without division or non-zero assumptions! -/
theorem rl_cleared_denominator_2x2_flux_identity
    (h00 h01 h10 h11 u0 u1 y0 y1 : Int) :
    let yhat0 := h00 * u0 + h01 * u1
    let yhat1 := h10 * u0 + h11 * u1
    let N0 := u0 * (h00 * y0 * yhat1 + h10 * y1 * yhat0)
    let N1 := u1 * (h01 * y0 * yhat1 + h11 * y1 * yhat0)
    N0 + N1 = (y0 + y1) * yhat0 * yhat1 := by
  dsimp
  simp only [Int.mul_add, Int.add_mul]
  ac_rfl

/-- Theorem 1e (Division-Free Universal Polynomial Flux Identity for Arbitrary `m × n`):
    For any `m × n` system, if `D` is the common denominator (e.g. `∏_{p=0}^{m-1} (H u)_p`)
    and `C_i` is the cofactor `∏_{p ≠ i} (H u)_p` so that `C_i * (H u)_i = D` for each `i < m`,
    then defining the cleared-denominator ratio `R_i = y_i * C_i` (which uses ZERO divisions!),
    the cleared-denominator RL update `N_j = u_j * ∑_{i=0}^{m-1} H_{i,j} R_i` satisfies:
      `∑_{j=0}^{n-1} N_j = D * ∑_{i=0}^{m-1} y_i`. -/
theorem rl_cleared_denominator_flux_general (H : Nat → Nat → Int) (m n : Nat)
    (u y C : Nat → Int) (D : Int)
    (h_cofactor : ∀ i, i < m → C i * forward H n u i = D) :
    let R := fun i => y i * C i
    sumRange n (rlStep H m u R) = D * sumRange m y := by
  intro R
  apply rl_flux_conservation H m n u y R D
  intro i hi
  dsimp [R]
  have h1 : y i * C i * forward H n u i = y i * (C i * forward H n u i) := by
    ac_rfl
  rw [h1, h_cofactor i hi]
  ac_rfl

-- ============================================================================
-- 2. Non-Negativity, Strict Positivity, and Zero-Locking Invariance
-- ============================================================================

/-- Theorem 2a (One-Step Non-Negativity Invariance):
    If `u_j ≥ 0`, `H_{i,j} ≥ 0`, and `r_i ≥ 0`, then `(rlStep H m u r)_j ≥ 0`. -/
theorem rl_nonneg (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j : Nat)
    (hu : 0 ≤ u j) (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i) :
    0 ≤ rlStep H m u r j := by
  unfold rlStep
  exact Int.mul_nonneg hu (backproject_nonneg H m r j hH hr)

/-- Theorem 2b (Multi-Step Trajectory Non-Negativity Invariance):
    If `u^{(0)} ≥ 0`, `H ≥ 0`, and `r^{(k)} ≥ 0` at each iteration `k`,
    then `u^{(k)}_j ≥ 0` for all iterations `k : Nat` and pixels `j < n`. -/
theorem rl_trajectory_nonneg (H : Nat → Nat → Int) (m n : Nat)
    (u0 : Nat → Int) (rSeq : Nat → Nat → Int)
    (hu0 : ∀ j, j < n → 0 ≤ u0 j)
    (hH : ∀ i j, i < m → j < n → 0 ≤ H i j)
    (hr : ∀ k i, i < m → 0 ≤ rSeq k i) :
    ∀ k j, j < n → 0 ≤ rlTrajectory H m u0 rSeq k j := by
  intro k
  induction k with
  | zero =>
    intro j hj
    exact hu0 j hj
  | succ t ih =>
    intro j hj
    simp only [rlTrajectory]
    exact rl_nonneg H m (rlTrajectory H m u0 rSeq t) (rSeq t) j
      (ih j hj) (fun i hi => hH i j hi hj) (fun i hi => hr t i hi)

/-- Theorem 2c (Strict Positivity Preservation):
    If `u_j > 0`, `H_{i,j} ≥ 0`, `r_i ≥ 0`, and there is at least one detector pixel `i₀ < m`
    with `H_{i₀,j} > 0` and `r_{i₀} > 0`, then `(rlStep H m u r)_j > 0`. -/
theorem rl_strict_pos (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j i₀ : Nat)
    (hi₀ : i₀ < m) (hu : 0 < u j)
    (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i)
    (hH_pos : 0 < H i₀ j) (hr_pos : 0 < r i₀) :
    0 < rlStep H m u r j := by
  unfold rlStep
  exact Int.mul_pos hu (backproject_pos H m r j i₀ hi₀ hH hr hH_pos hr_pos)

/-- Theorem 3a (Multiplicative Zero-Locking / Support Invariance):
    If `u_j = 0`, then `(rlStep H m u r)_j = 0`. -/
theorem rl_zero_locking (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j : Nat)
    (hu : u j = 0) :
    rlStep H m u r j = 0 := by
  unfold rlStep
  rw [hu, Int.zero_mul]

/-- Theorem 3b (Trajectory Zero-Locking):
    If `u^{(0)}_j = 0`, then `u^{(k)}_j = 0` for all `k : Nat`. -/
theorem rl_trajectory_zero_locking (H : Nat → Nat → Int) (m : Nat)
    (u0 : Nat → Int) (rSeq : Nat → Nat → Int) (j : Nat)
    (hu0 : u0 j = 0) :
    ∀ k, rlTrajectory H m u0 rSeq k j = 0 := by
  intro k
  induction k with
  | zero => exact hu0
  | succ t ih =>
    simp only [rlTrajectory]
    exact rl_zero_locking H m (rlTrajectory H m u0 rSeq t) (rSeq t) j ih

-- ============================================================================
-- 3. Fixed-Point Characterization & KKT Optimality Conditions
-- ============================================================================

/-- Lemma: When the detector residual ratio is constant (`r_i = S` for all `i < m`),
    the backprojection `(H^T r)_j` equals `w_j * S`. -/
theorem backproject_const_ratio (H : Nat → Nat → Int) (m : Nat) (r : Nat → Int) (S : Int) (j : Nat)
    (hr : ∀ i, i < m → r i = S) :
    backproject H m r j = sensitivity H m j * S := by
  unfold backproject sensitivity
  rw [← sumRange_mul_right]
  apply sumRange_congr
  intro i hi
  rw [hr i hi]

/-- Theorem 4a (Exact Solution Fixed-Point Theorem):
    If `u*` is an exact solution so that `H u* = y` (meaning the residual ratio is identically 1:
    `r_i = 1` for all `i < m`) and the PSF is unit-normalized (`w_j = 1`),
    then `u*` is an exact fixed point of the Lucy-Richardson operator:
      `rlStep H m u* r j = u* j`. -/
theorem rl_exact_solution_fixed_point (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j : Nat)
    (hr : ∀ i, i < m → r i = 1) (hw : sensitivity H m j = 1) :
    rlStep H m u r j = u j := by
  unfold rlStep
  rw [backproject_const_ratio H m r 1 j hr]
  rw [hw]
  omega

/-- Theorem 4b (Sensitivity-Weighted Exact Solution Fixed-Point):
    For any PSF (even without unit column normalization), if `r_i = S` for all `i < m`,
    then `rlStep H m u r j = u_j * w_j * S`, so dividing by `w_j * S` recovers `u_j` exactly. -/
theorem rl_weighted_exact_solution_fixed_point (H : Nat → Nat → Int) (m : Nat)
    (u r : Nat → Int) (S : Int) (j : Nat)
    (hr : ∀ i, i < m → r i = S) :
    rlStep H m u r j = u j * (sensitivity H m j * S) := by
  unfold rlStep
  rw [backproject_const_ratio H m r S j hr]

/-- Theorem 5 (KKT Complementarity & Stationarity Equivalence):
    A source pixel `j` is a fixed point of the sensitivity-weighted Lucy-Richardson step
      `w_j * u_j = rlStep H m u r j`
    if and only if the KKT complementarity condition holds:
      `u_j = 0  ∨  (H^T r)_j = w_j`. -/
theorem rl_kkt_stationarity_iff (H : Nat → Nat → Int) (m : Nat) (u r : Nat → Int) (j : Nat) :
    rlStep H m u r j = u j * sensitivity H m j ↔
    (u j = 0 ∨ backproject H m r j = sensitivity H m j) := by
  unfold rlStep
  constructor
  · intro h
    have hdiff : u j * (backproject H m r j - sensitivity H m j) = 0 := by
      rw [Int.mul_sub, h, Int.sub_self]
    cases Int.mul_eq_zero.mp hdiff with
    | inl hu => exact Or.inl hu
    | inr hbp =>
      apply Or.inr
      omega
  · intro h
    cases h with
    | inl hu =>
      rw [hu, Int.zero_mul, Int.zero_mul]
    | inr hbp =>
      rw [hbp]

end Lucky
