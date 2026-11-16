/-
  Lucky/VecMat.lean
  Arbitrary-Dimension Discrete Linear Operators, Finite Summations,
  Fubini's Theorem, and Forward/Adjoint Blur Calculus in Lean 4.
-/

namespace Lucky

/-- Finite summation `∑_{i=0}^{n-1} f(i)` over `Int`. -/
def sumRange : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumRange n f + f n

@[simp]
theorem sumRange_zero_len (f : Nat → Int) : sumRange 0 f = 0 := rfl

@[simp]
theorem sumRange_succ (n : Nat) (f : Nat → Int) :
    sumRange (n + 1) f = sumRange n f + f n := rfl

/-- Summation of the identically zero function is zero. -/
theorem sumRange_zero (n : Nat) : sumRange n (fun _ => 0) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp [sumRange, ih]

/-- Pointwise congruence of finite sums on `[0, n)`. -/
theorem sumRange_congr (n : Nat) (f g : Nat → Int) (h : ∀ i, i < n → f i = g i) :
    sumRange n f = sumRange n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    have ih' := ih (fun i hi => h i (by omega))
    have hk := h k (by omega)
    simp [sumRange, ih', hk]

/-- Linearity of finite sums: addition. -/
theorem sumRange_add (n : Nat) (f g : Nat → Int) :
    sumRange n (fun i => f i + g i) = sumRange n f + sumRange n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp [sumRange, ih]
    omega

/-- Linearity of finite sums: subtraction. -/
theorem sumRange_sub (n : Nat) (f g : Nat → Int) :
    sumRange n (fun i => f i - g i) = sumRange n f - sumRange n g := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp [sumRange, ih]
    omega

/-- Linearity of finite sums: negation. -/
theorem sumRange_neg (n : Nat) (f : Nat → Int) :
    sumRange n (fun i => -f i) = -sumRange n f := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp [sumRange, ih]
    omega

/-- Factoring a constant multiplier on the left out of a finite sum. -/
theorem sumRange_mul_left (n : Nat) (c : Int) (f : Nat → Int) :
    sumRange n (fun i => c * f i) = c * sumRange n f := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    simp [sumRange, ih, Int.mul_add]

/-- Factoring a constant multiplier on the right out of a finite sum. -/
theorem sumRange_mul_right (n : Nat) (c : Int) (f : Nat → Int) :
    sumRange n (fun i => f i * c) = sumRange n f * c := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    simp [sumRange, ih, Int.add_mul]

/-- Discrete Fubini's Theorem: exchanging the order of finite double sums for any `m, n : Nat`. -/
theorem sumRange_comm (m n : Nat) (f : Nat → Nat → Int) :
    sumRange m (fun i => sumRange n (fun j => f i j)) =
    sumRange n (fun j => sumRange m (fun i => f i j)) := by
  induction m with
  | zero =>
    simp [sumRange, sumRange_zero]
  | succ k ih =>
    simp only [sumRange_succ]
    rw [ih, ← sumRange_add]

/-- Non-negativity preservation of finite sums. -/
theorem sumRange_nonneg (n : Nat) (f : Nat → Int) (h : ∀ i, i < n → 0 ≤ f i) :
    0 ≤ sumRange n f := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    have ih' := ih (fun i hi => h i (by omega))
    have hk := h k (by omega)
    simp [sumRange]
    omega

/-- Monotonicity of finite sums: `f ≤ g` pointwise implies `∑ f ≤ ∑ g`. -/
theorem sumRange_le (n : Nat) (f g : Nat → Int) (h : ∀ i, i < n → f i ≤ g i) :
    sumRange n f ≤ sumRange n g := by
  induction n with
  | zero => simp [sumRange]
  | succ k ih =>
    have ih' := ih (fun i hi => h i (by omega))
    have hk := h k (by omega)
    simp [sumRange]
    omega

/-- Strict positivity of a finite sum of non-negative terms with at least one positive term. -/
theorem sumRange_pos (n : Nat) (f : Nat → Int) (k : Nat) (hk : k < n)
    (hnonneg : ∀ i, i < n → 0 ≤ f i) (hpos : 0 < f k) :
    0 < sumRange n f := by
  induction n with
  | zero => omega
  | succ m ih =>
    simp only [sumRange_succ]
    by_cases hkm : k = m
    · subst hkm
      have hsum : 0 ≤ sumRange k f := sumRange_nonneg k f (fun i hi => hnonneg i (by omega))
      omega
    · have hlt : k < m := by omega
      have ih' := ih hlt (fun i hi => hnonneg i (by omega))
      have hm : 0 ≤ f m := hnonneg m (by omega)
      omega

/-- A sum of non-negative terms is zero iff every term in the range is zero. -/
theorem sumRange_eq_zero_iff (n : Nat) (f : Nat → Int) (hnonneg : ∀ i, i < n → 0 ≤ f i) :
    sumRange n f = 0 ↔ ∀ i, i < n → f i = 0 := by
  induction n with
  | zero =>
    constructor
    · intro _ i hi; omega
    · intro _; rfl
  | succ k ih =>
    simp only [sumRange_succ]
    have hsub : ∀ i, i < k → 0 ≤ f i := fun i hi => hnonneg i (by omega)
    have hsum_nonneg : 0 ≤ sumRange k f := sumRange_nonneg k f hsub
    have hk_nonneg : 0 ≤ f k := hnonneg k (by omega)
    constructor
    · intro hzero i hi
      have hboth : sumRange k f = 0 ∧ f k = 0 := by omega
      by_cases hik : i = k
      · subst hik; exact hboth.2
      ·exact (ih hsub).mp hboth.1 i (by omega)
    · intro hall
      have hk0 : f k = 0 := hall k (by omega)
      have hsum0 : sumRange k f = 0 := (ih hsub).mpr (fun i hi => hall i (by omega))
      omega

-- ============================================================================
-- Forward Blur Operator (H u) and Adjoint Backprojection Operator (H^T r)
-- ============================================================================

/-- Forward imaging operator `(H u)_i = ∑_{j=0}^{n-1} H_{i,j} u_j`. -/
def forward (H : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (i : Nat) : Int :=
  sumRange n (fun j => H i j * u j)

/-- Adjoint backprojection operator `(H^T r)_j = ∑_{i=0}^{m-1} H_{i,j} r_i`. -/
def backproject (H : Nat → Nat → Int) (m : Nat) (r : Nat → Int) (j : Nat) : Int :=
  sumRange m (fun i => H i j * r i)

/-- Column sensitivity (detector coverage weight) `w_j = ∑_{i=0}^{m-1} H_{i,j}`. -/
def sensitivity (H : Nat → Nat → Int) (m : Nat) (j : Nat) : Int :=
  sumRange m (fun i => H i j)

/-- Fundamental Adjoint Duality Theorem:
    `⟨u, H^T r⟩_n = ∑_{j=0}^{n-1} u_j (H^T r)_j = ∑_{i=0}^{m-1} r_i (H u)_i = ⟨r, H u⟩_m`. -/
theorem adjoint_inner_product (H : Nat → Nat → Int) (m n : Nat) (u r : Nat → Int) :
    sumRange n (fun j => u j * backproject H m r j) =
    sumRange m (fun i => r i * forward H n u i) := by
  unfold backproject forward
  have h1 : sumRange n (fun j => u j * sumRange m (fun i => H i j * r i)) =
            sumRange n (fun j => sumRange m (fun i => u j * (H i j * r i))) := by
    apply sumRange_congr
    intro j _
    rw [← sumRange_mul_left]
  have h2 : sumRange m (fun i => r i * sumRange n (fun j => H i j * u j)) =
            sumRange m (fun i => sumRange n (fun j => r i * (H i j * u j))) := by
    apply sumRange_congr
    intro i _
    rw [← sumRange_mul_left]
  rw [h1, h2, sumRange_comm]
  apply sumRange_congr
  intro i _
  apply sumRange_congr
  intro j _
  ac_rfl

/-- Forward Flux Identity:
    `∑_{i=0}^{m-1} (H u)_i = ∑_{j=0}^{n-1} w_j u_j`. -/
theorem forward_flux_identity (H : Nat → Nat → Int) (m n : Nat) (u : Nat → Int) :
    sumRange m (fun i => forward H n u i) =
    sumRange n (fun j => sensitivity H m j * u j) := by
  unfold forward sensitivity
  rw [sumRange_comm]
  apply sumRange_congr
  intro j _
  rw [← sumRange_mul_right]

/-- Unit-Normalized PSF Forward Flux Conservation:
    When `∑_{i=0}^{m-1} H_{i,j} = 1` for all `j < n`, `∑_{i=0}^{m-1} (H u)_i = ∑_{j=0}^{n-1} u_j`. -/
theorem forward_unit_psf_flux (H : Nat → Nat → Int) (m n : Nat) (u : Nat → Int)
    (h_norm : ∀ j, j < n → sensitivity H m j = 1) :
    sumRange m (fun i => forward H n u i) = sumRange n u := by
  rw [forward_flux_identity]
  apply sumRange_congr
  intro j hj
  rw [h_norm j hj, Int.one_mul]

/-- Non-negativity of the forward operator `H u`. -/
theorem forward_nonneg (H : Nat → Nat → Int) (n : Nat) (u : Nat → Int) (i : Nat)
    (hH : ∀ j, j < n → 0 ≤ H i j) (hu : ∀ j, j < n → 0 ≤ u j) :
    0 ≤ forward H n u i := by
  unfold forward
  apply sumRange_nonneg
  intro j hj
  exact Int.mul_nonneg (hH j hj) (hu j hj)

/-- Non-negativity of the adjoint backprojection operator `H^T r`. -/
theorem backproject_nonneg (H : Nat → Nat → Int) (m : Nat) (r : Nat → Int) (j : Nat)
    (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i) :
    0 ≤ backproject H m r j := by
  unfold backproject
  apply sumRange_nonneg
  intro i hi
  exact Int.mul_nonneg (hH i hi) (hr i hi)

/-- Strict positivity of the adjoint backprojection operator `H^T r` when coupled to a positive ray. -/
theorem backproject_pos (H : Nat → Nat → Int) (m : Nat) (r : Nat → Int) (j i₀ : Nat)
    (hi₀ : i₀ < m) (hH : ∀ i, i < m → 0 ≤ H i j) (hr : ∀ i, i < m → 0 ≤ r i)
    (hH_pos : 0 < H i₀ j) (hr_pos : 0 < r i₀) :
    0 < backproject H m r j := by
  unfold backproject
  apply sumRange_pos m (fun i => H i j * r i) i₀ hi₀
  · intro i hi
    exact Int.mul_nonneg (hH i hi) (hr i hi)
  · exact Int.mul_pos hH_pos hr_pos

end Lucky
