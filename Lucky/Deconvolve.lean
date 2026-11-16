/-
  Lucky/Deconvolve.lean
  Executable 2D Image Deconvolution Engine in Lean 4:
    - 2D Image & PSF Kernel representations (`Image2D`)
    - Finite-Window Forward Convolution, Adjoint Correlation, & Boundary Sensitivity `w_{x,y}`
    - Quantitative Metrics: PSNR (dB), RMSE, Csiszár KL Divergence, Charbonnier TV Energy, Flux Error
    - Four Deconvolution Algorithms:
        1. Classic Lucy-Richardson (1972 / 1974)
        2. Dey et al. Unsplit TV-RL (2006)
        3. Boundary & Readout-Damped RL
        4. PCLR (Our Improved Positivity & Conservation-Preserving Accelerated Split-EM RL)
-/

namespace Lucky

@[inline]
def fmax (a b : Float) : Float := if a < b then b else a

@[inline]
def fmin (a b : Float) : Float := if a < b then a else b

/-- A 2D floating-point image of dimensions `width × height` stored in row-major order. -/
structure Image2D where
  width : Nat
  height : Nat
  pixels : Array Float
deriving Inhabited

namespace Image2D

@[inline]
def get (img : Image2D) (x y : Nat) : Float :=
  if x < img.width && y < img.height then
    img.pixels.getD (y * img.width + x) 0.0
  else
    0.0

@[inline]
def getClamped (img : Image2D) (x y : Int) : Float :=
  let cx : Nat :=
    if x < 0 then 0
    else if x.toNat ≥ img.width then img.width - 1
    else x.toNat
  let cy : Nat :=
    if y < 0 then 0
    else if y.toNat ≥ img.height then img.height - 1
    else y.toNat
  img.get cx cy

def ofFn (w h : Nat) (f : Nat → Nat → Float) : Image2D := Id.run do
  let mut arr := Array.mkEmpty (w * h)
  for y in [0:h] do
    for x in [0:w] do
      arr := arr.push (f x y)
  return ⟨w, h, arr⟩

def const (w h : Nat) (v : Float) : Image2D :=
  ⟨w, h, Array.replicate (w * h) v⟩

def totalFlux (img : Image2D) : Float :=
  img.pixels.foldl (fun acc v => acc + v) 0.0

def minVal (img : Image2D) : Float :=
  if img.pixels.isEmpty then 0.0
  else img.pixels.foldl (fun acc v => if v < acc then v else acc) (img.pixels.getD 0 0.0)

def maxVal (img : Image2D) : Float :=
  if img.pixels.isEmpty then 0.0
  else img.pixels.foldl (fun acc v => if v > acc then v else acc) (img.pixels.getD 0 0.0)

/-- Normalize an image or PSF kernel so its sum is 1.0. -/
def normalizeFlux (img : Image2D) (targetFlux : Float := 1.0) : Image2D :=
  let s := img.totalFlux
  if s <= 1.0e-15 then img
  else
    let scale := targetFlux / s
    ⟨img.width, img.height, img.pixels.map (fun v => v * scale)⟩

end Image2D

-- ============================================================================
-- 2D Forward Blur Convolution, Adjoint Backprojection, and Sensitivity Field
-- ============================================================================

/-- Finite-window 2D forward blur `(H u)_{x,y}` where `psf` has odd dimensions `(2*rx+1) × (2*ry+1)`.
    Only source pixels inside `[0, W) × [0, H)` contribute to the sensor window. -/
def forward2D (psf : Image2D) (u : Image2D) : Image2D :=
  let rx : Int := (psf.width / 2 : Nat)
  let ry : Int := (psf.height / 2 : Nat)
  Image2D.ofFn u.width u.height (fun x y => Id.run do
    let mut sum : Float := 0.0
    for ky in [0:psf.height] do
      let dy : Int := (ky : Int) - ry
      let sy : Int := (y : Int) - dy
      if 0 ≤ sy && sy < (u.height : Int) then
        for kx in [0:psf.width] do
          let dx : Int := (kx : Int) - rx
          let sx : Int := (x : Int) - dx
          if 0 ≤ sx && sx < (u.width : Int) then
            sum := sum + psf.get kx ky * u.get sx.toNat sy.toNat
    return sum)

/-- Adjoint 2D backprojection `(H^T r)_{x,y}` (correlating with flipped PSF). -/
def backproject2D (psf : Image2D) (r : Image2D) : Image2D :=
  let rx : Int := (psf.width / 2 : Nat)
  let ry : Int := (psf.height / 2 : Nat)
  Image2D.ofFn r.width r.height (fun x y => Id.run do
    let mut sum : Float := 0.0
    for ky in [0:psf.height] do
      let dy : Int := (ky : Int) - ry
      let iy : Int := (y : Int) + dy
      if 0 ≤ iy && iy < (r.height : Int) then
        for kx in [0:psf.width] do
          let dx : Int := (kx : Int) - rx
          let ix : Int := (x : Int) + dx
          if 0 ≤ ix && ix < (r.width : Int) then
            sum := sum + psf.get kx ky * r.get ix.toNat iy.toNat
    return sum)

/-- Windowed FOV sensitivity field `w_{x,y} = (H^T 1)_{x,y}`, measuring what fraction
    of photons emitted at `(x, y)` land inside the finite sensor window. -/
def sensitivityField2D (psf : Image2D) (w h : Nat) : Image2D :=
  backproject2D psf (Image2D.const w h 1.0)

-- ============================================================================
-- Quantitative Image Quality & Variational Energy Metrics
-- ============================================================================

/-- Root-Mean-Square Error (RMSE) between two images. -/
def rmse2D (a b : Image2D) : Float := Id.run do
  let n := a.pixels.size
  if n == 0 then return 0.0
  let mut s : Float := 0.0
  for i in [0:n] do
    let d := a.pixels.getD i 0.0 - b.pixels.getD i 0.0
    s := s + d * d
  return Float.sqrt (s / n.toFloat)

/-- Peak Signal-to-Noise Ratio (PSNR in dB) relative to peak intensity `peak`. -/
def psnr2D (truth est : Image2D) (peak : Float := 1.0) : Float :=
  let r := rmse2D truth est
  if r <= 1.0e-12 then 99.99
  else 20.0 * Float.log10 (peak / r)

/-- Generalized Kullback-Leibler Divergence (Csiszár I-Divergence) `D_KL(y ‖ H u)`. -/
def klDivergence2D (y yhat : Image2D) (eps : Float := 1.0e-8) : Float := Id.run do
  let n := y.pixels.size
  let mut sum : Float := 0.0
  for i in [0:n] do
    let yi := fmax eps (y.pixels.getD i 0.0)
    let yhi := fmax eps (yhat.pixels.getD i 0.0)
    sum := sum + (yi * Float.log (yi / yhi) + yhi - yi)
  return sum

/-- Isotropic / Bilateral Charbonnier Edge-Preserving Potential `Φ(u) = ∑_{x,y} ∑_{l ∈ N} ω_l (√(Δu² + ε²) - ε)`. -/
def charbonnierEnergy2D (u : Image2D) (eps : Float) : Float := Id.run do
  let mut sum : Float := 0.0
  let offsets : List (Int × Int × Float) := [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 0.5), (-1, 1, 0.5), (1, -1, 0.5), (-1, -1, 0.5)
  ]
  for y in [0:u.height] do
    for x in [0:u.width] do
      let u0 := u.get x y
      for (dx, dy, w) in offsets do
        let ul := u.getClamped ((x : Int) + dx) ((y : Int) + dy)
        let diff := u0 - ul
        sum := sum + 0.5 * w * (Float.sqrt (diff * diff + eps * eps) - eps)
  return sum

/-- Composite MAP Variational Objective `E(u) = D_KL(y ‖ H u) + λ Φ_Charbonnier(u)`. -/
def mapObjective2D (psf y u : Image2D) (lam eps : Float) : Float :=
  let yhat := forward2D psf u
  klDivergence2D y yhat + lam * charbonnierEnergy2D u eps

-- ============================================================================
-- Split-EM Edge-Preserving Regularizer Components `R⁺(u)` and `R⁻(u)`
-- ============================================================================

/-- Computes `(R⁺_{x,y}(u), R⁻_{x,y}(u))` for the 8-neighborhood Charbonnier / Bilateral TV
    potential, where `c_{j,l}(u) = ω_{j,l} / √((u_j - u_l)² + ε²)` is strictly positive and symmetric. -/
def splitCharbonnierTerms2D (u : Image2D) (eps : Float) : Image2D × Image2D := Id.run do
  let n := u.width * u.height
  let mut rPos := Array.mkEmpty n
  let mut rNeg := Array.mkEmpty n
  let offsets : List (Int × Int × Float) := [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 0.5), (-1, 1, 0.5), (1, -1, 0.5), (-1, -1, 0.5)
  ]
  for y in [0:u.height] do
    for x in [0:u.width] do
      let u0 := fmax 0.0 (u.get x y)
      let mut p : Float := 0.0
      let mut q : Float := 0.0
      for (dx, dy, w) in offsets do
        let ul := fmax 0.0 (u.getClamped ((x : Int) + dx) ((y : Int) + dy))
        let diff := u0 - ul
        let c := w / Float.sqrt (diff * diff + eps * eps)
        p := p + c * u0
        q := q + c * ul
      rPos := rPos.push p
      rNeg := rNeg.push q
  return (⟨u.width, u.height, rPos⟩, ⟨u.width, u.height, rNeg⟩)

/-- Computes Second-Order Midpoint Split-EM components `(R^{+,2}_{x,y}(u), R^{-,2}_{x,y}(u))`
    along the 4 symmetric axes `(1,0), (0,1), (1,1), (1,-1)`.
    By Lean 4 Theorem `second_order_linear_ramp_zero_staircasing` (`Lucky/BregmanTGV.lean`),
    this regularizer vanishes identically on linear intensity ramps (`∇Φ₂(a x + b) = 0`),
    eliminating staircasing on smooth slopes and nebula shells! -/
def splitSecondOrderTerms2D (u : Image2D) (eps : Float) : Image2D × Image2D := Id.run do
  let n := u.width * u.height
  let mut rPos := Array.mkEmpty n
  let mut rNeg := Array.mkEmpty n
  let axes : List (Int × Int × Float) := [
    (1, 0, 1.0), (0, 1, 1.0), (1, 1, 0.5), (1, -1, 0.5)
  ]
  for y in [0:u.height] do
    for x in [0:u.width] do
      let u0 := fmax 0.0 (u.get x y)
      let mut p : Float := 0.0
      let mut q : Float := 0.0
      for (dx, dy, w) in axes do
        let uPlus := fmax 0.0 (u.getClamped ((x : Int) + dx) ((y : Int) + dy))
        let uMinus := fmax 0.0 (u.getClamped ((x : Int) - dx) ((y : Int) - dy))
        let uMid := 0.5 * (uPlus + uMinus)
        let lap := u0 - uMid
        let c := 2.0 * w / Float.sqrt (lap * lap + eps * eps)
        p := p + c * u0
        q := q + c * uMid
      rPos := rPos.push p
      rNeg := rNeg.push q
  return (⟨u.width, u.height, rPos⟩, ⟨u.width, u.height, rNeg⟩)

/-- Local 3×3 Gaussian-weighted smoothing used for structure-adaptive sky/feature gating. -/
def smooth3x3 (u : Image2D) : Image2D :=
  Image2D.ofFn u.width u.height (fun x y => Id.run do
    let ix : Int := x
    let iy : Int := y
    let c0 := 4.0 * u.getClamped ix iy
    let n1 := 2.0 * (u.getClamped (ix + 1) iy + u.getClamped (ix - 1) iy +
                     u.getClamped ix (iy + 1) + u.getClamped ix (iy - 1))
    let n2 := 1.0 * (u.getClamped (ix + 1) (iy + 1) + u.getClamped (ix - 1) (iy + 1) +
                     u.getClamped (ix + 1) (iy - 1) + u.getClamped (ix - 1) (iy - 1))
    return (c0 + n1 + n2) / 16.0)

-- ============================================================================
-- Four Deconvolution Algorithms in Lean 4
-- ============================================================================

/-- 1. Classic Lucy-Richardson (1972 / 1974):
    `u^{(k+1)} = u^{(k)} ⊙ (H^T (y / H u^{(k)}))` without boundary weight, damping, or regularization. -/
def stepClassicRL (psf y u : Image2D) : Image2D :=
  let yhat := forward2D psf u
  let ratio : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty y.pixels.size
    for i in [0:y.pixels.size] do
      let yi := y.pixels.getD i 0.0
      let yhi := fmax 1.0e-7 (yhat.pixels.getD i 0.0)
      arr := arr.push (yi / yhi)
    return arr⟩
  let bproj := backproject2D psf ratio
  ⟨u.width, u.height, Id.run do
    let mut arr := Array.mkEmpty u.pixels.size
    for i in [0:u.pixels.size] do
      arr := arr.push (u.pixels.getD i 0.0 * bproj.pixels.getD i 0.0)
    return arr⟩

/-- 2. Dey et al. (2006) Unsplit TV-Regularized Lucy-Richardson:
    `u^{(k+1)} = u^{(k)} * (H^T (y / H u^{(k)})) / (w + λ (R⁺ - R⁻))`.
    Note: Can produce negative pixels when `w + λ (R⁺ - R⁻) < 0`! -/
def stepDeyTVRL (psf y wField u : Image2D) (lam eps : Float) : Image2D :=
  let yhat := forward2D psf u
  let ratio : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty y.pixels.size
    for i in [0:y.pixels.size] do
      let yi := y.pixels.getD i 0.0
      let yhi := fmax 1.0e-7 (yhat.pixels.getD i 0.0)
      arr := arr.push (yi / yhi)
    return arr⟩
  let bproj := backproject2D psf ratio
  let (rPos, rNeg) := splitCharbonnierTerms2D u eps
  ⟨u.width, u.height, Id.run do
    let mut arr := Array.mkEmpty u.pixels.size
    for i in [0:u.pixels.size] do
      let uj := u.pixels.getD i 0.0
      let wj := wField.pixels.getD i 1.0
      let grad := rPos.pixels.getD i 0.0 - rNeg.pixels.getD i 0.0
      let denom := wj + lam * grad
      let safeDenom := if Float.abs denom < 1.0e-6 then 1.0e-6 else denom
      arr := arr.push (uj * bproj.pixels.getD i 0.0 / safeDenom)
    return arr⟩

/-- 3. Boundary- & Readout-Damped Lucy-Richardson (Single Base Step):
    Uses sensitivity weight `w_{x,y}`, readout-noise offset `b > 0`, and White (1994) noise damping. -/
def stepDampedRL (psf y wField u : Image2D) (bOffset noiseSigma : Float) : Image2D :=
  let yhat := forward2D psf u
  let ratio : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty y.pixels.size
    for i in [0:y.pixels.size] do
      let yi := y.pixels.getD i 0.0
      let yhi := yhat.pixels.getD i 0.0
      let rawRatio := (yi + bOffset) / (yhi + bOffset)
      -- White (1994) adaptive SNR gate: damp noisy background fluctuations within `noiseSigma`
      let resid := Float.abs (yi - yhi)
      let snrWeight := (resid * resid) / (resid * resid + noiseSigma * noiseSigma + 1.0e-12)
      let dampedRatio := 1.0 + snrWeight * (rawRatio - 1.0)
      arr := arr.push dampedRatio
    return arr⟩
  let bproj := backproject2D psf ratio
  let raw : Image2D := ⟨u.width, u.height, Id.run do
    let mut arr := Array.mkEmpty u.pixels.size
    for i in [0:u.pixels.size] do
      let uj := u.pixels.getD i 0.0
      let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
      arr := arr.push (uj * (bproj.pixels.getD i 0.0 / wj))
    return arr⟩
  raw.normalizeFlux y.totalFlux

/-- Reflective boundary-padded 2D forward blur `(H u)_{x,y}`:
    Eliminates Gibbs boundary ringing on extended scenes that cross the image border! -/
def forwardReflect2D (psf : Image2D) (u : Image2D) : Image2D :=
  let rx : Int := (psf.width / 2 : Nat)
  let ry : Int := (psf.height / 2 : Nat)
  Image2D.ofFn u.width u.height (fun x y => Id.run do
    let mut sum : Float := 0.0
    for ky in [0:psf.height] do
      let dy : Int := (ky : Int) - ry
      let sy : Int := (y : Int) - dy
      for kx in [0:psf.width] do
        let dx : Int := (kx : Int) - rx
        let sx : Int := (x : Int) - dx
        sum := sum + psf.get kx ky * u.getClamped sx sy
    return sum)

/-- Adjoint 2D backprojection with clamped/reflective boundary handling. -/
def backprojectReflect2D (psf : Image2D) (r : Image2D) : Image2D :=
  let rx : Int := (psf.width / 2 : Nat)
  let ry : Int := (psf.height / 2 : Nat)
  Image2D.ofFn r.width r.height (fun x y => Id.run do
    let mut sum : Float := 0.0
    for ky in [0:psf.height] do
      let dy : Int := (ky : Int) - ry
      let iy : Int := (y : Int) + dy
      for kx in [0:psf.width] do
        let dx : Int := (kx : Int) - rx
        let ix : Int := (x : Int) + dx
        sum := sum + psf.get kx ky * r.getClamped ix iy
    return sum)

/-- 4. Our Improved Algorithm: PCLR (Positivity & Conservation-Preserving Accelerated Split-EM RL)
    Base Split-EM + Boundary-Weighted + Readout-Damped + Over-Relaxed Step `T_PCLR(u)`:
      `u'_{j} = Z⁻¹ * u_j * ( ((H^T r)_j + λ R⁻_j(u)) / (w_j + λ R⁺_j(u)) )^p`
    where:
      - `R⁺_j(u) ≥ 0, R⁻_j(u) ≥ 0` guarantee UNCONDITIONAL POSITIVITY (`Theorem 3c`),
      - `w_j = (H^T 1)_j` eliminates boundary truncation bias (`Theorem 1b`),
      - `b > 0` + White SNR damping prevent background noise spikes (`Theorem 4a, 4c`),
      - `Z` enforces exact photon conservation `∑ u'_j = ∑ y_i` (`Theorem 1a`). -/
def baseStepPCLR (psf y wField u : Image2D)
    (lam eps bOffset noiseSigma pGain : Float) : Image2D :=
  let yhat := forward2D psf u
  let ratio : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty y.pixels.size
    for i in [0:y.pixels.size] do
      let yi := y.pixels.getD i 0.0
      let yhi := yhat.pixels.getD i 0.0
      let rawRatio := (yi + bOffset) / (yhi + bOffset)
      let resid := Float.abs (yi - yhi)
      let snrWeight := (resid * resid) / (resid * resid + noiseSigma * noiseSigma + 1.0e-12)
      let dampedRatio := 1.0 + snrWeight * (rawRatio - 1.0)
      arr := arr.push dampedRatio
    return arr⟩
  let bproj := backproject2D psf ratio
  let (rPos, rNeg) := splitCharbonnierTerms2D u eps
  let raw : Image2D := ⟨u.width, u.height, Id.run do
    let mut arr := Array.mkEmpty u.pixels.size
    for i in [0:u.pixels.size] do
      let uj := fmax 0.0 (u.pixels.getD i 0.0)
      let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
      let num := bproj.pixels.getD i 0.0 + lam * rNeg.pixels.getD i 0.0
      let den := wj + lam * rPos.pixels.getD i 0.0
      let mult := Float.pow (fmax 1.0e-8 (num / den)) pGain
      arr := arr.push (uj * mult)
    return arr⟩
  raw.normalizeFlux y.totalFlux

/-- Full Safeguarded Accelerated PCLR Iteration (`Theorem 5a-5e`):
    Combines:
      1. Multiplicative log-space Biggs-Andrews / Nesterov extrapolation
         `v_j = u_j * (u_j / uPrev_j)^β_k` (natively preserving strict positivity!),
      2. Homotopy edge-threshold annealing on `ε_k` to sharpen fine edges and point sources,
      3. Monotone Variational Objective Guard `E(u_accel) ≤ E(T_PCLR(u))` (`Theorem 5a-5e`). -/
def runPCLR (psf y : Image2D) (iters : Nat)
    (lam : Float := 0.015) (eps : Float := 0.014)
    (bOffset : Float := 0.015) (noiseSigma : Float := 0.008)
    (pGain : Float := 1.32) : Image2D := Id.run do
  let wField := sensitivityField2D psf y.width y.height
  let targetFlux := y.totalFlux
  let mut u := y.normalizeFlux targetFlux
  let mut uPrev := u
  for k in [0:iters] do
    -- Homotopy schedule: softly ramp regularization after initial low-freq recovery
    let progress := (k + 1).toFloat / (fmax 1.0 iters.toFloat)
    let lamK := lam * (0.45 + 0.55 * progress)
    let epsK := eps * (1.25 - 0.45 * progress)
    let tStep := baseStepPCLR psf y wField u lamK epsK bOffset noiseSigma pGain
    if k == 0 then
      uPrev := u
      u := tStep
    else
      -- Multiplicative Biggs-Andrews / Nesterov momentum in log-intensity space
      let kf := k.toFloat
      let beta := fmin 0.72 (kf / (kf + 2.5))
      let vExtrapRaw : Image2D := ⟨u.width, u.height, Id.run do
        let mut arr := Array.mkEmpty u.pixels.size
        for i in [0:u.pixels.size] do
          let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
          let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
          let ratioStep := fmin 2.5 (fmax 0.4 (uk / up))
          let v := uk * Float.pow ratioStep beta
          arr := arr.push v
        return arr⟩
      let vExtrap := vExtrapRaw.normalizeFlux targetFlux
      let accelProposal := baseStepPCLR psf y wField vExtrap lamK epsK bOffset noiseSigma pGain
      -- Monotone Objective Safeguard (`guardedAccelStep` from `Lucky/Improved.lean`)
      let eBase := mapObjective2D psf y tStep lam eps
      let eAccel := mapObjective2D psf y accelProposal lam eps
      uPrev := u
      if eAccel ≤ eBase then
        u := accelProposal
      else
        u := tStep
  return u

/-- 5. Second-Generation Improved Algorithm: TGV-Bregman PCLR (`Lucky/BregmanTGV.lean`)
    Combines:
      1. First-Order Charbonnier (`R^{+,1}, R^{-,1}`) + Second-Order Midpoint (`R^{+,2}, R^{-,2}`)
         Split-EM Regularization (zero staircasing on linear slopes by `Theorem 6b`),
      2. Dual Bregman Subgradient Debiasing `q = q⁺ - q⁻` accumulated from `(H^T r - w)`
         (restoring 100% star peak contrast with unconditional positivity by `Theorem 7a-7d`),
      3. Structure-Adaptive SNR Damping (full damping in dark sky, full fidelity on features),
      4. Multiplicative Log-Space Nesterov Extrapolation + Exact Photon Conservation. -/
def runTGVBregmanPCLR (psf y : Image2D) (iters : Nat)
    (lam1 : Float := 0.045) (lam2 : Float := 0.004)
    (eps1 : Float := 0.010) (eps2 : Float := 0.008)
    (bOffset : Float := 0.010) (noiseSigma : Float := 0.012)
    (pGain : Float := 1.38) (gammaBreg : Float := 0.35) : Image2D := Id.run do
  let wField := sensitivityField2D psf y.width y.height
  let targetFlux := y.totalFlux
  let n := y.pixels.size
  let mut u := y.normalizeFlux targetFlux
  let mut uPrev := u
  let mut qDual := Array.replicate n 0.0

  for k in [0:iters] do
    let kf := k.toFloat
    let beta := if k == 0 then 0.0 else fmin 0.74 (kf / (kf + 2.3))
    let vRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
        let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
        let ratioStep := fmin 2.6 (fmax 0.38 (uk / up))
        arr := arr.push (uk * Float.pow ratioStep beta)
      return arr⟩
    let v := vRaw.normalizeFlux targetFlux
    let vSmooth := smooth3x3 v
    let yhat := forward2D psf v

    let ratio : Image2D := ⟨y.width, y.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let yi := y.pixels.getD i 0.0
        let yhi := yhat.pixels.getD i 0.0
        let vsi := vSmooth.pixels.getD i 0.04
        let skyFactor := Float.exp (-fmax 0.0 (vsi - 0.045) / 0.045)
        let localSig := noiseSigma * (0.08 + 0.92 * skyFactor)
        let rawRatio := (yi + bOffset) / (yhi + bOffset)
        let resid := Float.abs (yi - yhi)
        let snrGate := (resid * resid) / (resid * resid + localSig * localSig + 1.0e-12)
        arr := arr.push (1.0 + snrGate * (rawRatio - 1.0))
      return arr⟩

    let bproj := backproject2D psf ratio
    if k ≥ 5 then
      let mut qNext := Array.mkEmpty n
      for i in [0:n] do
        let vsi := vSmooth.pixels.getD i 0.04
        let skyFactor := Float.exp (-fmax 0.0 (vsi - 0.045) / 0.045)
        let structGate := 1.0 - skyFactor
        let bpDiff := bproj.pixels.getD i 0.0 - wField.pixels.getD i 1.0
        let qi := 0.92 * qDual.getD i 0.0 + gammaBreg * structGate * bpDiff
        qNext := qNext.push qi
      qDual := qNext

    let (r1Pos, r1Neg) := splitCharbonnierTerms2D v eps1
    let (r2Pos, r2Neg) := splitSecondOrderTerms2D v eps2

    let uNewRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let vi := fmax 0.0 (v.pixels.getD i 0.0)
        let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
        let vsi := vSmooth.pixels.getD i 0.04
        let skyFactor := Float.exp (-fmax 0.0 (vsi - 0.045) / 0.045)
        let w1 := lam1 * (0.25 + 0.75 * skyFactor)
        let w2 := lam2 * (1.0 - 0.5 * skyFactor)
        let qi := qDual.getD i 0.0
        let qPos := fmax 0.0 qi
        let qNeg := fmax 0.0 (-qi)
        let num := fmax 1.0e-8 (bproj.pixels.getD i 0.0 + w1 * r1Neg.pixels.getD i 0.0 + w2 * r2Neg.pixels.getD i 0.0 + qPos)
        let den := fmax 1.0e-8 (wj + w1 * r1Pos.pixels.getD i 0.0 + w2 * r2Pos.pixels.getD i 0.0 + qNeg)
        arr := arr.push (vi * Float.pow (num / den) pGain)
      return arr⟩

    uPrev := u
    u := uNewRaw.normalizeFlux targetFlux
  return u

/-- Pairwise-symmetric spatially-adaptive Charbonnier Split-EM regularizer (`R^{+,1}, R^{-,1}`):
    Uses symmetric edge conductivity `c_{j,l} = 0.5 * (s_j + s_l) * ω_{j,l} / √((u_j - u_l)² + ε²)`.
    Because `c_{j,l} = c_{l,j}`, Lean 4 Theorem `split_reg_flux_balance` (`Lucky/Improved.lean`)
    guarantees exact global flux balance `∑_j R^{+,1}_j = ∑_j R^{-,1}_j` even for spatially varying `s_j`! -/
def splitSymmetricCharbonnier2D (u sWeight : Image2D) (eps : Float) : Image2D × Image2D := Id.run do
  let n := u.width * u.height
  let mut rPos := Array.mkEmpty n
  let mut rNeg := Array.mkEmpty n
  let offsets : List (Int × Int × Float) := [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, 0.5), (-1, 1, 0.5), (1, -1, 0.5), (-1, -1, 0.5)
  ]
  for y in [0:u.height] do
    for x in [0:u.width] do
      let u0 := fmax 0.0 (u.get x y)
      let s0 := sWeight.get x y
      let mut p : Float := 0.0
      let mut q : Float := 0.0
      for (dx, dy, w) in offsets do
        let ul := fmax 0.0 (u.getClamped ((x : Int) + dx) ((y : Int) + dy))
        let sl := sWeight.getClamped ((x : Int) + dx) ((y : Int) + dy)
        let diff := u0 - ul
        let c := 0.5 * (s0 + sl) * w / Float.sqrt (diff * diff + eps * eps)
        p := p + c * u0
        q := q + c * ul
      rPos := rPos.push p
      rNeg := rNeg.push q
  return (⟨u.width, u.height, rPos⟩, ⟨u.width, u.height, rNeg⟩)

/-- 6. Third-Generation Self-Calibrated Symmetric TGV-Bregman PCLR (`runGen3PCLR`):
    Combines:
      1. Automatic background & gradient-flatness self-calibration (no hardcoded sky threshold),
      2. Pairwise-symmetric spatially-adaptive 1st-order Charbonnier + 2nd-order Midpoint Split-EM,
      3. Denoised Dual Bregman subgradient debiasing (`q = q⁺ - q⁻`) + step annealing. -/
def runGen3PCLR (psf y : Image2D) (iters : Nat)
    (lamSky : Float := 0.055) (lamFg : Float := 0.010) (lam2 : Float := 0.004)
    (eps1 : Float := 0.009) (eps2 : Float := 0.007)
    (bOffset : Float := 0.008) (noiseSigma : Float := 0.011)
    (pGain : Float := 1.40) (gammaBreg : Float := 0.38) : Image2D := Id.run do
  let wField := sensitivityField2D psf y.width y.height
  let targetFlux := y.totalFlux
  let n := y.pixels.size
  let ySmooth := smooth3x3 y
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)
  let bgScale := fmax 0.025 (0.075 * dynRange)

  let mut u := y.normalizeFlux targetFlux
  let mut uPrev := u
  let mut qDual := Array.replicate n 0.0

  for k in [0:iters] do
    let kf := k.toFloat
    let damp := 1.0 / (1.0 + fmax 0.0 (kf - 20.0) * 0.22)
    let lateDamp := 1.0 / (1.0 + fmax 0.0 (kf - 24.0) * 0.25)
    let beta := if k == 0 then 0.0 else (fmin 0.75 (kf / (kf + 2.2))) * damp
    let pK := (1.0 + (pGain - 1.0) * damp) * lateDamp

    let vRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
        let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
        let ratioStep := fmin 2.6 (fmax 0.38 (uk / up))
        arr := arr.push (uk * Float.pow ratioStep beta)
      return arr⟩
    let v := vRaw.normalizeFlux targetFlux
    let vSmooth := smooth3x3 v
    let yhat := forward2D psf v

    -- Dual-criterion flatness gate: intensity near background OR ultra-low local gradient
    let flatGate : Image2D := Image2D.ofFn v.width v.height (fun x y =>
      let vsi := vSmooth.get x y
      let gx := 0.5 * (vSmooth.getClamped ((x : Int) + 1) y - vSmooth.getClamped ((x : Int) - 1) y)
      let gy := 0.5 * (vSmooth.getClamped x ((y : Int) + 1) - vSmooth.getClamped x ((y : Int) - 1))
      let gmag := Float.sqrt (gx * gx + gy * gy)
      let bgG := Float.exp (-fmax 0.0 (vsi - bgEst) / bgScale)
      let grG := Float.exp (-(gmag / (0.015 * dynRange)) * (gmag / (0.015 * dynRange)))
      fmax bgG (0.45 * grG))

    let ratio : Image2D := ⟨y.width, y.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let yi := y.pixels.getD i 0.0
        let yhi := yhat.pixels.getD i 0.0
        let skyF := flatGate.pixels.getD i 0.5
        let localSig := noiseSigma * (0.06 + 0.94 * skyF)
        let rawRatio := (yi + bOffset) / (yhi + bOffset)
        let resid := Float.abs (yi - yhi)
        let snrGate := (resid * resid) / (resid * resid + localSig * localSig + 1.0e-12)
        arr := arr.push (1.0 + snrGate * (rawRatio - 1.0))
      return arr⟩

    let bproj := backproject2D psf ratio
    if 5 ≤ k && k ≤ 22 then
      let bpDiffImg : Image2D := ⟨y.width, y.height, Id.run do
        let mut arr := Array.mkEmpty n
        for i in [0:n] do
          arr := arr.push (bproj.pixels.getD i 0.0 - wField.pixels.getD i 1.0)
        return arr⟩
      let bpSmooth := smooth3x3 bpDiffImg
      let mut qNext := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        let structG := 1.0 - skyF
        let bpd := 0.82 * bpDiffImg.pixels.getD i 0.0 + 0.18 * bpSmooth.pixels.getD i 0.0
        let qi := 0.92 * qDual.getD i 0.0 + gammaBreg * structG * bpd
        qNext := qNext.push qi
      qDual := qNext

    let sWeight : Image2D := ⟨v.width, v.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        arr := arr.push (lamFg + (lamSky - lamFg) * skyF)
      return arr⟩

    let (r1Pos, r1Neg) := splitSymmetricCharbonnier2D v sWeight eps1
    let (r2Pos, r2Neg) := splitSecondOrderTerms2D v eps2

    let uNewRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let vi := fmax 0.0 (v.pixels.getD i 0.0)
        let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
        let skyF := flatGate.pixels.getD i 0.5
        let w2 := lam2 * (1.0 - 0.5 * skyF)
        let qi := qDual.getD i 0.0
        let qPos := fmax 0.0 qi
        let qNeg := fmax 0.0 (-qi)
        let num := fmax 1.0e-8 (bproj.pixels.getD i 0.0 + r1Neg.pixels.getD i 0.0 + w2 * r2Neg.pixels.getD i 0.0 + qPos)
        let den := fmax 1.0e-8 (wj + r1Pos.pixels.getD i 0.0 + w2 * r2Pos.pixels.getD i 0.0 + qNeg)
        arr := arr.push (vi * Float.pow (num / den) pK)
      return arr⟩

    uPrev := u
    u := uNewRaw.normalizeFlux targetFlux
  return u

/-- 7. Joint Multi-Frame "Lucky Imaging" TGV-Bregman Split-EM Deconvolution (`Lucky/Multiframe.lean`):
    Simultaneously deconvolves a burst of `F` short-exposure frames `frames[f]` with distinct
    anisotropic turbulence PSFs `psfs[f]`, weighting each frame by its Lucky sharpness `γ_f`. -/
def runMultiFrameLuckyPCLR (psfs frames : Array Image2D) (iters : Nat)
    (lamSky : Float := 0.055) (lamFg : Float := 0.009) (lam2 : Float := 0.004)
    (eps1 : Float := 0.009) (eps2 : Float := 0.007)
    (bOffset : Float := 0.008) (noiseSigma : Float := 0.013)
    (pGain : Float := 1.40) (gammaBreg : Float := 0.38) : Image2D := Id.run do
  if frames.isEmpty then return Image2D.const 1 1 0.0
  let f0 := frames.getD 0 (Image2D.const 1 1 0.0)
  let w := f0.width
  let h := f0.height
  let n := w * h
  let numF := frames.size

  -- Compute Lucky sharpness weights γ_f = (∑ (psf_f)²)^0.75 normalized to unit sum
  let mut rawGammas := Array.mkEmpty numF
  let mut gammaSum : Float := 0.0
  for f in [0:numF] do
    let p := psfs.getD f f0
    let mut sq : Float := 0.0
    for v in p.pixels do
      sq := sq + v * v
    let gf := Float.pow (fmax 1.0e-8 sq) 0.75
    rawGammas := rawGammas.push gf
    gammaSum := gammaSum + gf
  let gammas := rawGammas.map (fun g => g / (fmax 1.0e-12 gammaSum))

  -- Precompute joint sensitivity w̄_j = ∑_f γ_f w_j^{(f)} and weighted average frame ȳ
  let mut wBarArr := Array.replicate n 0.0
  let mut yBarArr := Array.replicate n 0.0
  for f in [0:numF] do
    let gf := gammas.getD f 0.0
    let wf := sensitivityField2D (psfs.getD f f0) w h
    let yf := frames.getD f f0
    let mut nextW := Array.mkEmpty n
    let mut nextY := Array.mkEmpty n
    for i in [0:n] do
      nextW := nextW.push (wBarArr.getD i 0.0 + gf * wf.pixels.getD i 1.0)
      nextY := nextY.push (yBarArr.getD i 0.0 + gf * yf.pixels.getD i 0.0)
    wBarArr := nextW
    yBarArr := nextY

  let wBar : Image2D := ⟨w, h, wBarArr⟩
  let yBar : Image2D := ⟨w, h, yBarArr⟩
  let targetFlux := yBar.totalFlux
  let ySmooth := smooth3x3 yBar
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)
  let bgScale := fmax 0.025 (0.075 * dynRange)

  let mut u := yBar.normalizeFlux targetFlux
  let mut uPrev := u
  let mut qDual := Array.replicate n 0.0

  for k in [0:iters] do
    let kf := k.toFloat
    let damp := 1.0 / (1.0 + fmax 0.0 (kf - 20.0) * 0.22)
    let lateDamp := 1.0 / (1.0 + fmax 0.0 (kf - 24.0) * 0.25)
    let beta := if k == 0 then 0.0 else (fmin 0.75 (kf / (kf + 2.2))) * damp
    let pK := (1.0 + (pGain - 1.0) * damp) * lateDamp

    let vRaw : Image2D := ⟨w, h, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
        let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
        let ratioStep := fmin 2.6 (fmax 0.38 (uk / up))
        arr := arr.push (uk * Float.pow ratioStep beta)
      return arr⟩
    let v := vRaw.normalizeFlux targetFlux
    let vSmooth := smooth3x3 v

    let flatGate : Image2D := Image2D.ofFn w h (fun x y =>
      let vsi := vSmooth.get x y
      let gx := 0.5 * (vSmooth.getClamped ((x : Int) + 1) y - vSmooth.getClamped ((x : Int) - 1) y)
      let gy := 0.5 * (vSmooth.getClamped x ((y : Int) + 1) - vSmooth.getClamped x ((y : Int) - 1))
      let gmag := Float.sqrt (gx * gx + gy * gy)
      let bgG := Float.exp (-fmax 0.0 (vsi - bgEst) / bgScale)
      let grG := Float.exp (-(gmag / (0.015 * dynRange)) * (gmag / (0.015 * dynRange)))
      fmax bgG (0.45 * grG))

    -- Joint Multi-Frame Backprojection B_j = ∑_f γ_f (H_f^T r_f)_j
    let mut bprojBarArr := Array.replicate n 0.0
    for f in [0:numF] do
      let gf := gammas.getD f 0.0
      let psfF := psfs.getD f f0
      let yF := frames.getD f f0
      let yhatF := forward2D psfF v
      let ratioF : Image2D := ⟨w, h, Id.run do
        let mut arr := Array.mkEmpty n
        for i in [0:n] do
          let yi := yF.pixels.getD i 0.0
          let yhi := yhatF.pixels.getD i 0.0
          let skyF := flatGate.pixels.getD i 0.5
          let localSig := noiseSigma * (0.06 + 0.94 * skyF)
          let rawRatio := (yi + bOffset) / (yhi + bOffset)
          let resid := Float.abs (yi - yhi)
          let snrGate := (resid * resid) / (resid * resid + localSig * localSig + 1.0e-12)
          arr := arr.push (1.0 + snrGate * (rawRatio - 1.0))
        return arr⟩
      let bpF := backproject2D psfF ratioF
      let mut nextBp := Array.mkEmpty n
      for i in [0:n] do
        nextBp := nextBp.push (bprojBarArr.getD i 0.0 + gf * bpF.pixels.getD i 0.0)
      bprojBarArr := nextBp

    let bprojBar : Image2D := ⟨w, h, bprojBarArr⟩
    if 5 ≤ k && k ≤ 22 then
      let bpDiffImg : Image2D := ⟨w, h, Id.run do
        let mut arr := Array.mkEmpty n
        for i in [0:n] do
          arr := arr.push (bprojBar.pixels.getD i 0.0 - wBar.pixels.getD i 1.0)
        return arr⟩
      let bpSmooth := smooth3x3 bpDiffImg
      let mut qNext := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        let structG := 1.0 - skyF
        let bpd := 0.82 * bpDiffImg.pixels.getD i 0.0 + 0.18 * bpSmooth.pixels.getD i 0.0
        let qi := 0.92 * qDual.getD i 0.0 + gammaBreg * structG * bpd
        qNext := qNext.push qi
      qDual := qNext

    let sWeight : Image2D := ⟨w, h, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        arr := arr.push (lamFg + (lamSky - lamFg) * skyF)
      return arr⟩

    let (r1Pos, r1Neg) := splitSymmetricCharbonnier2D v sWeight eps1
    let (r2Pos, r2Neg) := splitSecondOrderTerms2D v eps2

    let uNewRaw : Image2D := ⟨w, h, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let vi := fmax 0.0 (v.pixels.getD i 0.0)
        let wj := fmax 1.0e-6 (wBar.pixels.getD i 1.0)
        let skyF := flatGate.pixels.getD i 0.5
        let w2 := lam2 * (1.0 - 0.5 * skyF)
        let qi := qDual.getD i 0.0
        let qPos := fmax 0.0 qi
        let qNeg := fmax 0.0 (-qi)
        let num := fmax 1.0e-8 (bprojBar.pixels.getD i 0.0 + r1Neg.pixels.getD i 0.0 + w2 * r2Neg.pixels.getD i 0.0 + qPos)
        let den := fmax 1.0e-8 (wj + r1Pos.pixels.getD i 0.0 + w2 * r2Pos.pixels.getD i 0.0 + qNeg)
        arr := arr.push (vi * Float.pow (num / den) pK)
      return arr⟩

    uPrev := u
    u := uNewRaw.normalizeFlux targetFlux
  return u

-- ============================================================================
-- Generation-4: Blind Deconvolution (Single-Frame & Multi-Frame Blind Lucky PCLR)
-- Formalized in `Lucky/Blind.lean` (Theorems 11a–11d & 12a–12f)
-- ============================================================================

/-- Dual PSF-space adjoint backprojection `(U(u)^T r)_{kx, ky}` for a `kw × kh` PSF
    (Theorem 11b `blind_joint_adjoint_identity`). -/
def backprojectPSF2D (u : Image2D) (r : Image2D) (kw kh : Nat) : Image2D :=
  let rx : Int := (kw / 2 : Nat)
  let ry : Int := (kh / 2 : Nat)
  Image2D.ofFn kw kh (fun kx ky => Id.run do
    let dx : Int := (kx : Int) - rx
    let dy : Int := (ky : Int) - ry
    let mut sum : Float := 0.0
    for y in [0:r.height] do
      let sy : Int := (y : Int) - dy
      if 0 ≤ sy && sy < (u.height : Int) then
        for x in [0:r.width] do
          let sx : Int := (x : Int) - dx
          if 0 ≤ sx && sx < (u.width : Int) then
            sum := sum + u.get sx.toNat sy.toNat * r.get x y
    return sum)

/-- Locks the barycentric first-moment centroid `(∑ kx h, ∑ ky h)` of a PSF to its
    geometric center `(kw/2, kh/2)` via bilinear subpixel shifting, eliminating
    translational shift ambiguity `(u(x - δ), h(x + δ))`. -/
def centerPSF2D (psf : Image2D) : Image2D := Id.run do
  let kw := psf.width
  let kh := psf.height
  let rx := (kw / 2).toFloat
  let ry := (kh / 2).toFloat
  let tot := fmax 1.0e-12 psf.totalFlux
  let mut cx : Float := 0.0
  let mut cy : Float := 0.0
  for ky in [0:kh] do
    for kx in [0:kw] do
      let v := psf.get kx ky
      cx := cx + (kx.toFloat - rx) * v
      cy := cy + (ky.toFloat - ry) * v
  cx := cx / tot
  cy := cy / tot
  let shifted := Image2D.ofFn kw kh (fun kx ky =>
    let sx := kx.toFloat + cx
    let sy := ky.toFloat + cy
    let x0 := Float.floor sx
    let y0 := Float.floor sy
    let fx := sx - x0
    let fy := sy - y0
    let ix0 : Int := ((x0 + 100.0).toUInt64.toNat : Int) - 100
    let iy0 : Int := ((y0 + 100.0).toUInt64.toNat : Int) - 100
    let sample (ix iy : Int) : Float :=
      if 0 ≤ ix && ix < (kw : Int) && 0 ≤ iy && iy < (kh : Int) then
        psf.get ix.toNat iy.toNat
      else 0.0
    let v00 := sample ix0 iy0
    let v10 := sample (ix0 + 1) iy0
    let v01 := sample ix0 (iy0 + 1)
    let v11 := sample (ix0 + 1) (iy0 + 1)
    (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 +
    (1.0 - fx) * fy * v01 + fx * fy * v11)
  return shifted.normalizeFlux 1.0

/-- Classic Alternating Blind Lucy-Richardson (Holmes 1992 / Fish et al. 1995):
    Alternates unregularized RL image steps and RL PSF simplex steps.
    Demonstrates the Levin et al. (2009) no-blur / noise-amplification degeneracy
    (Theorem 12a `blind_no_blur_degeneracy_theorem`). -/
def runClassicBlindRL (psfInit y : Image2D) (outerCycles : Nat := 10)
    (uIters : Nat := 4) (hIters : Nat := 4) : Image2D × Image2D := Id.run do
  let kw := psfInit.width
  let kh := psfInit.height
  let targetFlux := y.totalFlux
  let onesY := Image2D.const y.width y.height 1.0
  let mut u := y.normalizeFlux targetFlux
  let mut h := psfInit.normalizeFlux 1.0
  for _ in [0:outerCycles] do
    for _ in [0:uIters] do
      let yhat := forward2D h u
      let ratio : Image2D := ⟨y.width, y.height, Id.run do
        let mut arr := Array.mkEmpty y.pixels.size
        for i in [0:y.pixels.size] do
          let yi := y.pixels.getD i 0.0
          let yhi := fmax 1.0e-8 (yhat.pixels.getD i 0.0)
          arr := arr.push (yi / yhi)
        return arr⟩
      let bproj := backproject2D h ratio
      let uNext : Image2D := ⟨u.width, u.height, Id.run do
        let mut arr := Array.mkEmpty u.pixels.size
        for i in [0:u.pixels.size] do
          arr := arr.push (u.pixels.getD i 0.0 * bproj.pixels.getD i 0.0)
        return arr⟩
      u := uNext.normalizeFlux targetFlux
    let sensH := backprojectPSF2D u onesY kw kh
    for _ in [0:hIters] do
      let yhat := forward2D h u
      let ratio : Image2D := ⟨y.width, y.height, Id.run do
        let mut arr := Array.mkEmpty y.pixels.size
        for i in [0:y.pixels.size] do
          let yi := y.pixels.getD i 0.0
          let yhi := fmax 1.0e-8 (yhat.pixels.getD i 0.0)
          arr := arr.push (yi / yhi)
        return arr⟩
      let bprojH := backprojectPSF2D u ratio kw kh
      let hNext : Image2D := ⟨kw, kh, Id.run do
        let mut arr := Array.mkEmpty h.pixels.size
        for k in [0:h.pixels.size] do
          let hk := h.pixels.getD k 0.0
          let bpk := bprojH.pixels.getD k 0.0
          let sk := fmax 1.0e-8 (sensH.pixels.getD k 1.0)
          arr := arr.push (hk * (bpk / sk))
        return arr⟩
      h := hNext.normalizeFlux 1.0
  return (u, h)

/-- Shock-sharpened salient structure guide `u_sal` that restores 1-pixel step edges
    and point sources while gating out low-SNR background fluctuations, preventing
    PSF width underestimation (`σ_h² = σ_y² - σ_u²`). -/
def buildSalientGuide2D (u : Image2D) (bgEst dynRange : Float)
    (shockSteps : Nat := 3) (alphaShock : Float := 0.52) : Image2D := Id.run do
  let w := u.width
  let h := u.height
  let mut uCurr := u
  for _ in [0:shockSteps] do
    let uS := smooth3x3 uCurr
    let uS2 := smooth3x3 uS
    let nextU := Image2D.ofFn w h (fun x y =>
      let ui := uCurr.get x y
      let gx := 0.5 * (uS.getClamped ((x : Int) + 1) y - uS.getClamped ((x : Int) - 1) y)
      let gy := 0.5 * (uS.getClamped x ((y : Int) + 1) - uS.getClamped x ((y : Int) - 1))
      let gmag := Float.sqrt (gx * gx + gy * gy)
      let lap := uS.get x y - uS2.get x y
      let scaledLap := lap / (0.015 * dynRange + 1.0e-8)
      let ePos := Float.exp (fmin 12.0 scaledLap)
      let eNeg := Float.exp (fmax (-12.0) (-scaledLap))
      let tanhLap := (ePos - eNeg) / (ePos + eNeg)
      fmin 1.0 (fmax 0.0 (ui + alphaShock * tanhLap * gmag)))
    uCurr := nextU
  return Image2D.ofFn w h (fun x y =>
    let ui := uCurr.get x y
    let fgEx := fmax 0.0 (ui - bgEst)
    let thr := 0.035 * dynRange
    let gate := (fgEx * fgEx) / (fgEx * fgEx + thr * thr + 1.0e-12)
    fmin 1.0 (fmax 0.0 (bgEst + gate * (ui - bgEst))))

/-- Multi-Channel Salient High-Pass + Jordan Signed-Gradient Split-EM PSF Solver
    (Formalized in `Lucky/Blind.lean`, Theorems 11c & 12c–12e):
    Solves for the unknown PSF `h` on the probability simplex using Nesterov-accelerated
    Split-EM across high-pass point-source, foreground, and directional Jordan-split
    signed-gradient (`g_x⁺, g_x⁻, g_y⁺, g_y⁻`) channels, with Charbonnier smoothness
    and soft-thresholded barycentric centering. -/
def solveBlindPSF2D (y uSal psfPrior : Image2D) (bgEst : Float)
    (innerIters : Nat := 7) (tauSparse : Float := 0.008) : Image2D := Id.run do
  let w := y.width
  let h := y.height
  let kw := psfPrior.width
  let kh := psfPrior.height
  let onesY := Image2D.const w h 1.0

  let yBg := smooth3x3 (smooth3x3 (smooth3x3 y))
  let uBg := smooth3x3 (smooth3x3 (smooth3x3 uSal))

  let yHp := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (y.get x yIdx - yBg.get x yIdx - 0.003))
  let uHp := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (uSal.get x yIdx - uBg.get x yIdx - 0.003))

  let yFg := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (y.get x yIdx - bgEst - 0.004))
  let uFg := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (uSal.get x yIdx - bgEst - 0.004))

  let gxY := Image2D.ofFn w h (fun x yIdx => 0.5 * (y.getClamped ((x : Int) + 1) yIdx - y.getClamped ((x : Int) - 1) yIdx))
  let gyY := Image2D.ofFn w h (fun x yIdx => 0.5 * (y.getClamped x ((yIdx : Int) + 1) - y.getClamped x ((yIdx : Int) - 1)))
  let gxU := Image2D.ofFn w h (fun x yIdx => 0.5 * (uSal.getClamped ((x : Int) + 1) yIdx - uSal.getClamped ((x : Int) - 1) yIdx))
  let gyU := Image2D.ofFn w h (fun x yIdx => 0.5 * (uSal.getClamped x ((yIdx : Int) + 1) - uSal.getClamped x ((yIdx : Int) - 1)))

  let gxpY := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (gxY.get x yIdx - 0.002))
  let gxnY := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (-gxY.get x yIdx - 0.002))
  let gypY := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (gyY.get x yIdx - 0.002))
  let gynY := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (-gyY.get x yIdx - 0.002))

  let gxpU := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (gxU.get x yIdx - 0.002))
  let gxnU := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (-gxU.get x yIdx - 0.002))
  let gypU := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (gyU.get x yIdx - 0.002))
  let gynU := Image2D.ofFn w h (fun x yIdx => fmax 0.0 (-gyU.get x yIdx - 0.002))

  let channels : Array (Float × Image2D × Image2D) := #[
    (1.30, yHp, uHp),
    (0.65, yFg, uFg),
    (0.85, gxpY, gxpU),
    (0.85, gxnY, gxnU),
    (0.85, gypY, gypU),
    (0.85, gynY, gynU)
  ]

  let mut sensArr := Array.replicate (kw * kh) 0.0
  for (wCh, _, uCh) in channels do
    let sCh := backprojectPSF2D uCh onesY kw kh
    let mut nextS := Array.mkEmpty (kw * kh)
    for k in [0:kw * kh] do
      nextS := nextS.push (sensArr.getD k 0.0 + wCh * sCh.pixels.getD k 0.0)
    sensArr := nextS
  let sensTotal : Image2D := ⟨kw, kh, sensArr⟩
  let meanSens := fmax 1.0e-8 (sensTotal.totalFlux / (kw * kh).toFloat)

  let mut hCurr := psfPrior.normalizeFlux 1.0
  let mut hPrev := hCurr
  for it in [0:innerIters] do
    let kf := it.toFloat
    let beta := if it == 0 then 0.0 else fmin 0.55 (kf / (kf + 3.0))
    let vRaw : Image2D := ⟨kw, kh, Id.run do
      let mut arr := Array.mkEmpty (kw * kh)
      for k in [0:kw * kh] do
        let hk := fmax 1.0e-8 (hCurr.pixels.getD k 0.0)
        let hp := fmax 1.0e-8 (hPrev.pixels.getD k 0.0)
        let rStep := fmin 2.0 (fmax 0.5 (hk / hp))
        arr := arr.push (hk * Float.pow rStep beta)
      return arr⟩
    let vH := vRaw.normalizeFlux 1.0

    let mut bprojArr := Array.replicate (kw * kh) 0.0
    for (wCh, yCh, uCh) in channels do
      let yhatCh := forward2D vH uCh
      let ratioCh : Image2D := ⟨w, h, Id.run do
        let mut arr := Array.mkEmpty (w * h)
        for i in [0:w * h] do
          let yi := yCh.pixels.getD i 0.0
          let yhi := yhatCh.pixels.getD i 0.0
          arr := arr.push ((yi + 0.002) / (yhi + 0.002))
        return arr⟩
      let bpCh := backprojectPSF2D uCh ratioCh kw kh
      let mut nextBp := Array.mkEmpty (kw * kh)
      for k in [0:kw * kh] do
        nextBp := nextBp.push (bprojArr.getD k 0.0 + wCh * bpCh.pixels.getD k 0.0)
      bprojArr := nextBp

    let (rPos, rNeg) := splitCharbonnierTerms2D vH 0.003
    let lamH := 0.002 * meanSens
    let hNext : Image2D := ⟨kw, kh, Id.run do
      let mut arr := Array.mkEmpty (kw * kh)
      for k in [0:kw * kh] do
        let hk := vH.pixels.getD k 0.0
        let bpk := bprojArr.getD k 0.0
        let sk := fmax 1.0e-8 (sensTotal.pixels.getD k 1.0)
        let rpk := rPos.pixels.getD k 0.0
        let rnk := rNeg.pixels.getD k 0.0
        let num := fmax 1.0e-10 (bpk + lamH * rnk)
        let den := fmax 1.0e-10 (sk + lamH * rpk)
        arr := arr.push (hk * Float.pow (num / den) 1.15)
      return arr⟩
    hPrev := hCurr
    hCurr := hNext.normalizeFlux 1.0

  let maxH := hCurr.maxVal
  let hThresh : Image2D := ⟨kw, kh, hCurr.pixels.map (fun v => fmax 0.0 (v - tauSparse * maxH))⟩
  return centerPSF2D (hThresh.normalizeFlux 1.0)

/-- Single-Frame Blind Self-Calibrated Gen-3 PCLR (`runBlindPCLR`):
    Starting ONLY from a generic isotropic Gaussian PSF guess `psfInit` and a single
    blurred+noisy observation `y`, jointly recovers the sharp image `u` and the unknown
    anisotropic PSF `h` via 4 outer stages of Salient Guide extraction + Split-EM PSF estimation. -/
def runBlindPCLR (psfInit y : Image2D) (iters : Nat := 35) : Image2D × Image2D := Id.run do
  let ySmooth := smooth3x3 y
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)
  let mut hEst := psfInit.normalizeFlux 1.0

  for stage in [0:4] do
    let hSmooth := smooth3x3 hEst
    let hGuide : Image2D := ⟨psfInit.width, psfInit.height, Id.run do
      let mut arr := Array.mkEmpty psfInit.pixels.size
      for k in [0:psfInit.pixels.size] do
        arr := arr.push (0.80 * hEst.pixels.getD k 0.0 + 0.20 * hSmooth.pixels.getD k 0.0)
      return arr⟩
    let hGuideNorm := hGuide.normalizeFlux 1.0
    let uGuide := runGen3PCLR hGuideNorm y (20 + 5 * stage) 0.055 0.0085 0.004 0.009 0.007 0.008 0.011 1.38 0.36
    let uSal := buildSalientGuide2D uGuide bgEst dynRange 3 0.54

    let priorBlend : Image2D := ⟨psfInit.width, psfInit.height, Id.run do
      let mut arr := Array.mkEmpty psfInit.pixels.size
      for k in [0:psfInit.pixels.size] do
        arr := arr.push (0.50 * psfInit.pixels.getD k 0.0 + 0.50 * hSmooth.pixels.getD k 0.0)
      return arr⟩
    hEst := solveBlindPSF2D y uSal (priorBlend.normalizeFlux 1.0) bgEst (6 + stage) 0.008

  let uFinal := runGen3PCLR hEst y iters 0.060 0.0085 0.004 0.009 0.007 0.008 0.011 1.40 0.36
  return (uFinal, hEst)

/-- Multi-Frame Blind "Lucky Imaging" PCLR (`runMultiFrameBlindLuckyPCLR`):
    Given `F` short-exposure frames `frames` blurred by `F` UNKNOWN, distinct atmospheric
    turbulence PSFs and a single generic isotropic Gaussian initial guess `psfInit`,
    jointly recovers all `F` unknown PSFs `psfsEst` and the diffraction-limited image `uJoint`
    (Theorem 12f `blind_multiframe_cross_convolution_identifiability`). -/
def runMultiFrameBlindLuckyPCLR (psfInit : Image2D) (frames : Array Image2D)
    (iters : Nat := 35) : Image2D × Array Image2D := Id.run do
  let numF := frames.size
  if numF == 0 then return (Image2D.const 1 1 0.0, #[])
  let f0 := frames.getD 0 (Image2D.const 1 1 0.0)
  let ySmooth := smooth3x3 f0
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)

  let mut psfsEst := Array.replicate numF (psfInit.normalizeFlux 1.0)
  for stage in [0:3] do
    let psfsGuide := psfsEst.map (fun h =>
      let hs := smooth3x3 h
      let g : Image2D := ⟨h.width, h.height, Id.run do
        let mut arr := Array.mkEmpty h.pixels.size
        for k in [0:h.pixels.size] do
          arr := arr.push (0.80 * h.pixels.getD k 0.0 + 0.20 * hs.pixels.getD k 0.0)
        return arr⟩
      g.normalizeFlux 1.0)
    let uJointGuide := runMultiFrameLuckyPCLR psfsGuide frames (20 + 5 * stage)
    let uSal := buildSalientGuide2D uJointGuide bgEst dynRange 3 0.54
    let mut nextPsfs := Array.mkEmpty numF
    for f in [0:numF] do
      let hf := psfsEst.getD f psfInit
      let hSmooth := smooth3x3 hf
      let priorF : Image2D := ⟨psfInit.width, psfInit.height, Id.run do
        let mut arr := Array.mkEmpty psfInit.pixels.size
        for k in [0:psfInit.pixels.size] do
          arr := arr.push (0.45 * psfInit.pixels.getD k 0.0 + 0.55 * hSmooth.pixels.getD k 0.0)
        return arr⟩
      let hNew := solveBlindPSF2D (frames.getD f f0) uSal (priorF.normalizeFlux 1.0) bgEst (6 + stage) 0.008
      nextPsfs := nextPsfs.push hNew
    psfsEst := nextPsfs

  let uFinal := runMultiFrameLuckyPCLR psfsEst frames iters
  return (uFinal, psfsEst)

-- ============================================================================
-- Generation-4 Lunar Deconvolution: Preconditioned Compact Adjoint +
-- Anisotropic Structure-Tensor Coherence & Hessian Rille Split-EM
-- Formalized in `Lucky/Lunar.lean` (Theorems 13a–13g)
-- ============================================================================

/-- Pairwise-symmetrized Anisotropic Structure-Tensor Coherence Split-EM (`R^{+,aniso}, R^{-,aniso}`):
    Computes local structure tensor `(J_xx, J_xy, J_yy)`, coherence `C_j ∈ [0, 1)`, and
    unit gradient normal `(n_x, n_y)`, then attenuates cross-feature smoothing while enhancing
    along-rille tangent smoothing with exact pairwise symmetry `c_{j,l}^{aniso} = c_{l,j}^{aniso}`
    (Theorem 13e `anisotropic_coherence_split_reg_flux_balance`). -/
def splitAnisotropicCharbonnier2D (u sWeight : Image2D) (eps : Float) : Image2D × Image2D := Id.run do
  let w := u.width
  let h := u.height
  let n := w * h
  let uS := smooth3x3 u
  let gx := Image2D.ofFn w h (fun x y =>
    0.5 * (uS.getClamped ((x : Int) + 1) y - uS.getClamped ((x : Int) - 1) y))
  let gy := Image2D.ofFn w h (fun x y =>
    0.5 * (uS.getClamped x ((y : Int) + 1) - uS.getClamped x ((y : Int) - 1)))
  let jxx := smooth3x3 (Image2D.ofFn w h (fun x y => let v := gx.get x y; v * v))
  let jxy := smooth3x3 (Image2D.ofFn w h (fun x y => gx.get x y * gy.get x y))
  let jyy := smooth3x3 (Image2D.ofFn w h (fun x y => let v := gy.get x y; v * v))

  let mut cohArr := Array.mkEmpty n
  let mut nxArr := Array.mkEmpty n
  let mut nyArr := Array.mkEmpty n
  for y in [0:h] do
    for x in [0:w] do
      let a := jxx.get x y
      let b := jxy.get x y
      let c := jyy.get x y
      let tr := a + c
      let d := Float.sqrt ((a - c) * (a - c) + 4.0 * b * b + 1.0e-14)
      let coh := (d / (tr + 1.0e-6)) * (d / (tr + 1.0e-6))
      let gxi := gx.get x y
      let gyi := gy.get x y
      let gn := Float.sqrt (gxi * gxi + gyi * gyi + 1.0e-12)
      cohArr := cohArr.push coh
      nxArr := nxArr.push (gxi / gn)
      nyArr := nyArr.push (gyi / gn)
  let cohImg : Image2D := ⟨w, h, cohArr⟩
  let nxImg : Image2D := ⟨w, h, nxArr⟩
  let nyImg : Image2D := ⟨w, h, nyArr⟩

  let mut rPos := Array.mkEmpty n
  let mut rNeg := Array.mkEmpty n
  let offsets : List (Int × Int × Float × Float × Float) := [
    (1, 0, 1.0, 1.0, 0.0), (-1, 0, 1.0, -1.0, 0.0),
    (0, 1, 1.0, 0.0, 1.0), (0, -1, 1.0, 0.0, -1.0),
    (1, 1, 0.5, 0.70710678, 0.70710678), (-1, 1, 0.5, -0.70710678, 0.70710678),
    (1, -1, 0.5, 0.70710678, -0.70710678), (-1, -1, 0.5, -0.70710678, -0.70710678)
  ]
  for y in [0:h] do
    for x in [0:w] do
      let u0 := fmax 0.0 (u.get x y)
      let s0 := sWeight.get x y
      let c0 := cohImg.get x y
      let nx0 := nxImg.get x y
      let ny0 := nyImg.get x y
      let mut p : Float := 0.0
      let mut q : Float := 0.0
      for (dx, dy, wBase, ex, ey) in offsets do
        let sx : Int := (x : Int) + dx
        let sy : Int := (y : Int) + dy
        let ul := fmax 0.0 (u.getClamped sx sy)
        let sl := sWeight.getClamped sx sy
        let cl := cohImg.getClamped sx sy
        let nxl := nxImg.getClamped sx sy
        let nyl := nyImg.getClamped sx sy
        let dot0 := (nx0 * ex + ny0 * ey) * (nx0 * ex + ny0 * ey)
        let dotl := (nxl * ex + nyl * ey) * (nxl * ex + nyl * ey)
        let d0 := s0 * (1.0 + c0 * (0.30 * (1.0 - dot0) - 0.65 * dot0))
        let dl := sl * (1.0 + cl * (0.30 * (1.0 - dotl) - 0.65 * dotl))
        let diff := u0 - ul
        let cSym := 0.5 * (d0 + dl) * wBase / Float.sqrt (diff * diff + eps * eps)
        p := p + cSym * u0
        q := q + cSym * ul
      rPos := rPos.push p
      rNeg := rNeg.push q
  return (⟨w, h, rPos⟩, ⟨w, h, rNeg⟩)

/-- Generation-4 Lunar Preconditioned Compact Adjoint + Anisotropic Coherence Split-EM (`runLunarPrecondAnisoPCLR`):
    Combines:
      1. Compact Adjoint Preconditioner `psfAdj` (`K_adj^T (y / H_fwd u)`) with exact data-fidelity
         fixed-point invariance (Theorem 13a `preconditioned_split_em_exact_fixed_point`),
      2. Second-order Hessian rille/ridge curvature gating (Theorem 13c `rille_centerline_first_order_blindness_vs_hessian_detection`),
      3. Pairwise-symmetrized Anisotropic Structure-Tensor Coherence Split-EM (Theorem 13e). -/
def runLunarPrecondAnisoPCLR (psfFwd psfAdj y : Image2D) (iters : Nat := 35)
    (lamSky : Float := 0.055) (lamFg : Float := 0.008) (lam2 : Float := 0.004)
    (eps1 : Float := 0.009) (eps2 : Float := 0.007)
    (bOffset : Float := 0.008) (noiseSigma : Float := 0.011)
    (pGain : Float := 1.42) (gammaBreg : Float := 0.40) : Image2D := Id.run do
  let wField := sensitivityField2D psfAdj y.width y.height
  let targetFlux := y.totalFlux
  let n := y.pixels.size
  let ySmooth := smooth3x3 y
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)

  let mut u := y.normalizeFlux targetFlux
  let mut uPrev := u
  let mut qDual := Array.replicate n 0.0

  for k in [0:iters] do
    let kf := k.toFloat
    let damp := 1.0 / (1.0 + fmax 0.0 (kf - 20.0) * 0.22)
    let lateDamp := 1.0 / (1.0 + fmax 0.0 (kf - 24.0) * 0.25)
    let beta := if k == 0 then 0.0 else (fmin 0.76 (kf / (kf + 2.1))) * damp
    let pK := (1.0 + (pGain - 1.0) * damp) * lateDamp

    let vRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
        let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
        let ratioStep := fmin 2.6 (fmax 0.38 (uk / up))
        arr := arr.push (uk * Float.pow ratioStep beta)
      return arr⟩
    let v := vRaw.normalizeFlux targetFlux
    let vSmooth := smooth3x3 v
    let yhat := forward2D psfFwd v

    -- Combined 1st-order gradient + 2nd-order Hessian rille curvature flatness gate (Theorem 13c-13d)
    let flatGate : Image2D := Image2D.ofFn v.width v.height (fun x yIdx =>
      let gx := 0.5 * (vSmooth.getClamped ((x : Int) + 1) yIdx - vSmooth.getClamped ((x : Int) - 1) yIdx)
      let gy := 0.5 * (vSmooth.getClamped x ((yIdx : Int) + 1) - vSmooth.getClamped x ((yIdx : Int) - 1))
      let gmag := Float.sqrt (gx * gx + gy * gy)
      let u0 := vSmooth.get x yIdx
      let dxx := Float.abs (vSmooth.getClamped ((x : Int) + 1) yIdx - 2.0 * u0 + vSmooth.getClamped ((x : Int) - 1) yIdx)
      let dyy := Float.abs (vSmooth.getClamped x ((yIdx : Int) + 1) - 2.0 * u0 + vSmooth.getClamped x ((yIdx : Int) - 1))
      let hCurv := fmax dxx dyy
      let featScore := gmag / (0.015 * dynRange) + 0.85 * hCurv / (0.012 * dynRange)
      Float.exp (-featScore * featScore))

    let ratio : Image2D := ⟨y.width, y.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let yi := y.pixels.getD i 0.0
        let yhi := yhat.pixels.getD i 0.0
        let skyF := flatGate.pixels.getD i 0.5
        let localSig := noiseSigma * (0.05 + 0.95 * skyF)
        let rawRatio := (yi + bOffset) / (yhi + bOffset)
        let resid := Float.abs (yi - yhi)
        let snrGate := (resid * resid) / (resid * resid + localSig * localSig + 1.0e-12)
        arr := arr.push (1.0 + snrGate * (rawRatio - 1.0))
      return arr⟩

    -- Backproject through compact adjoint preconditioner `psfAdj` (Theorem 13a)
    let bproj := backproject2D psfAdj ratio
    if 4 ≤ k && k ≤ 22 then
      let bpDiffImg : Image2D := ⟨y.width, y.height, Id.run do
        let mut arr := Array.mkEmpty n
        for i in [0:n] do
          arr := arr.push (bproj.pixels.getD i 0.0 - wField.pixels.getD i 1.0)
        return arr⟩
      let bpSmooth := smooth3x3 bpDiffImg
      let mut qNext := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        let structG := 1.0 - skyF
        let bpd := 0.82 * bpDiffImg.pixels.getD i 0.0 + 0.18 * bpSmooth.pixels.getD i 0.0
        let qi := 0.92 * qDual.getD i 0.0 + gammaBreg * structG * bpd
        qNext := qNext.push qi
      qDual := qNext

    let sWeight : Image2D := ⟨v.width, v.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        arr := arr.push (lamFg + (lamSky - lamFg) * skyF)
      return arr⟩

    let (r1Pos, r1Neg) := splitAnisotropicCharbonnier2D v sWeight eps1
    let (r2Pos, r2Neg) := splitSecondOrderTerms2D v eps2

    let uNewRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let vi := fmax 0.0 (v.pixels.getD i 0.0)
        let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
        let skyF := flatGate.pixels.getD i 0.5
        let w2 := lam2 * (1.0 - 0.45 * skyF)
        let qi := qDual.getD i 0.0
        let qPos := fmax 0.0 qi
        let qNeg := fmax 0.0 (-qi)
        let num := fmax 1.0e-8 (bproj.pixels.getD i 0.0 + r1Neg.pixels.getD i 0.0 + w2 * r2Neg.pixels.getD i 0.0 + qPos)
        let den := fmax 1.0e-8 (wj + r1Pos.pixels.getD i 0.0 + w2 * r2Pos.pixels.getD i 0.0 + qNeg)
        arr := arr.push (vi * Float.pow (num / den) pK)
      return arr⟩

    uPrev := u
    u := uNewRaw.normalizeFlux targetFlux
  return u

-- ============================================================================
-- Generation-6 Rényi-Escort Adjoint + Censored-Saturation Split-EM &
-- Theoretical Limit / Stein-SURE Evaluators (`Lucky/Lunar.lean` Thms 14a-14c,
-- `Lucky/Limits.lean` Thms 15a-15l)
-- ============================================================================

/-- Constructs the Rényi-Escort fractional-power adjoint preconditioner `K_{adj} ∝ H^α`
    (`Lucky/Lunar.lean`, Theorem 14b `renyi_escort_adjoint_properties`), which narrows
    the adjoint backprojector FWHM by `1 / √α` while preserving exact 2D PSF anisotropy,
    centroid, non-negativity, and the exact data-fidelity fixed point `H u* = y`. -/
def makeRenyiEscortAdjointPSF (psf : Image2D) (alpha : Float := 1.38) : Image2D :=
  let raw : Image2D := ⟨psf.width, psf.height, psf.pixels.map (fun v => Float.pow (fmax 0.0 v) alpha)⟩
  raw.normalizeFlux 1.0

/-- Generation-6 Lunar Rényi-Escort Adjoint + Censored-Saturation + Anisotropic Hessian Split-EM
    (`Lucky/Lunar.lean` Theorems 14a–14c & `Lucky/Limits.lean` Theorems 15d–15k). -/
def runLunarGen6RenyiPCLR (psfFwd y : Image2D) (iters : Nat := 35)
    (alphaAdj : Float := 1.38) (satCeiling : Float := 0.995)
    (lamSky : Float := 0.055) (lamFg : Float := 0.0072) (lam2 : Float := 0.0036)
    (eps1 : Float := 0.009) (eps2 : Float := 0.007)
    (bOffset : Float := 0.008) (noiseSigma : Float := 0.011)
    (pGain : Float := 1.44) (gammaBreg : Float := 0.42) : Image2D := Id.run do
  let psfAdj := makeRenyiEscortAdjointPSF psfFwd alphaAdj
  let wField := sensitivityField2D psfAdj y.width y.height
  let targetFlux := y.totalFlux
  let n := y.pixels.size
  let ySmooth := smooth3x3 y
  let bgEst := fmax 0.01 (ySmooth.minVal + 0.005)
  let dynRange := fmax 0.10 (ySmooth.maxVal - bgEst)

  let mut u := y.normalizeFlux targetFlux
  let mut uPrev := u
  let mut qDual := Array.replicate n 0.0

  for k in [0:iters] do
    let kf := k.toFloat
    let damp := 1.0 / (1.0 + fmax 0.0 (kf - 19.0) * 0.24)
    let lateDamp := 1.0 / (1.0 + fmax 0.0 (kf - 23.0) * 0.26)
    let beta := if k == 0 then 0.0 else (fmin 0.76 (kf / (kf + 2.1))) * damp
    let pK := (1.0 + (pGain - 1.0) * damp) * lateDamp

    let vRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let uk := fmax 1.0e-7 (u.pixels.getD i 0.0)
        let up := fmax 1.0e-7 (uPrev.pixels.getD i 0.0)
        let ratioStep := fmin 2.6 (fmax 0.38 (uk / up))
        arr := arr.push (uk * Float.pow ratioStep beta)
      return arr⟩
    let v := vRaw.normalizeFlux targetFlux
    let vSmooth := smooth3x3 v
    let yhat := forward2D psfFwd v

    let flatGate : Image2D := Image2D.ofFn v.width v.height (fun x yIdx =>
      let gx := 0.5 * (vSmooth.getClamped ((x : Int) + 1) yIdx - vSmooth.getClamped ((x : Int) - 1) yIdx)
      let gy := 0.5 * (vSmooth.getClamped x ((yIdx : Int) + 1) - vSmooth.getClamped x ((yIdx : Int) - 1))
      let gmag := Float.sqrt (gx * gx + gy * gy)
      let u0 := vSmooth.get x yIdx
      let dxx := Float.abs (vSmooth.getClamped ((x : Int) + 1) yIdx - 2.0 * u0 + vSmooth.getClamped ((x : Int) - 1) yIdx)
      let dyy := Float.abs (vSmooth.getClamped x ((yIdx : Int) + 1) - 2.0 * u0 + vSmooth.getClamped x ((yIdx : Int) - 1))
      let hCurv := fmax dxx dyy
      let featScore := gmag / (0.015 * dynRange) + 0.88 * hCurv / (0.012 * dynRange)
      Float.exp (-featScore * featScore))

    -- Censored-Saturation Split-EM Ratio (Theorem 14a `censored_saturation_split_em_restores_fixed_point`)
    let ratio : Image2D := ⟨y.width, y.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let yi := y.pixels.getD i 0.0
        let yhi := yhat.pixels.getD i 0.0
        if yi ≥ satCeiling then
          arr := arr.push 1.0
        else
          let skyF := flatGate.pixels.getD i 0.5
          let localSig := noiseSigma * (0.05 + 0.95 * skyF)
          let rawRatio := (yi + bOffset) / (yhi + bOffset)
          let resid := Float.abs (yi - yhi)
          let snrGate := (resid * resid) / (resid * resid + localSig * localSig + 1.0e-12)
          arr := arr.push (1.0 + snrGate * (rawRatio - 1.0))
      return arr⟩

    -- Backproject through Rényi-Escort fractional-power adjoint preconditioner (Theorem 14b)
    let bproj := backproject2D psfAdj ratio
    if 4 ≤ k && k ≤ 21 then
      let bpDiffImg : Image2D := ⟨y.width, y.height, Id.run do
        let mut arr := Array.mkEmpty n
        for i in [0:n] do
          arr := arr.push (bproj.pixels.getD i 0.0 - wField.pixels.getD i 1.0)
        return arr⟩
      let bpSmooth := smooth3x3 bpDiffImg
      let mut qNext := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        let structG := 1.0 - skyF
        let bpd := 0.82 * bpDiffImg.pixels.getD i 0.0 + 0.18 * bpSmooth.pixels.getD i 0.0
        let qi := 0.92 * qDual.getD i 0.0 + gammaBreg * structG * bpd
        qNext := qNext.push qi
      qDual := qNext

    let sWeight : Image2D := ⟨v.width, v.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let skyF := flatGate.pixels.getD i 0.5
        arr := arr.push (lamFg + (lamSky - lamFg) * skyF)
      return arr⟩

    let (r1Pos, r1Neg) := splitAnisotropicCharbonnier2D v sWeight eps1
    let (r2Pos, r2Neg) := splitSecondOrderTerms2D v eps2

    let uNewRaw : Image2D := ⟨u.width, u.height, Id.run do
      let mut arr := Array.mkEmpty n
      for i in [0:n] do
        let vi := fmax 0.0 (v.pixels.getD i 0.0)
        let wj := fmax 1.0e-6 (wField.pixels.getD i 1.0)
        let skyF := flatGate.pixels.getD i 0.5
        let w2 := lam2 * (1.0 - 0.45 * skyF)
        let qi := qDual.getD i 0.0
        let qPos := fmax 0.0 qi
        let qNeg := fmax 0.0 (-qi)
        let num := fmax 1.0e-8 (bproj.pixels.getD i 0.0 + r1Neg.pixels.getD i 0.0 + w2 * r2Neg.pixels.getD i 0.0 + qPos)
        let den := fmax 1.0e-8 (wj + r1Pos.pixels.getD i 0.0 + w2 * r2Pos.pixels.getD i 0.0 + qNeg)
        arr := arr.push (vi * Float.pow (num / den) pK)
      return arr⟩

    uPrev := u
    u := uNewRaw.normalizeFlux targetFlux
  return u

/-- Evaluates Both Sides of Stein's Unbiased Risk Identity (`Lucky/Limits.lean`, Theorem 15l
    `stein_sure_unbiased_risk_identity`) on 2D images:
      `LHS = ‖H uHat - H u*‖₂² + ‖η‖₂²`,
      `RHS = ‖y - H uHat‖₂² + 2 ⟨η, H uHat - H u*⟩`.
    Returns `(lhs, rhs, |lhs - rhs|)`. -/
def verifySteinSUREIdentity2D (mu y muHat : Image2D) : Float × Float × Float := Id.run do
  let n := mu.pixels.size
  let mut sqPredErr : Float := 0.0
  let mut sqNoise : Float := 0.0
  let mut sqObsResid : Float := 0.0
  let mut covDof : Float := 0.0
  for i in [0:n] do
    let mui := mu.pixels.getD i 0.0
    let yi := y.pixels.getD i 0.0
    let mhi := muHat.pixels.getD i 0.0
    let etai := yi - mui
    let dPred := mhi - mui
    let dObs := yi - mhi
    sqPredErr := sqPredErr + dPred * dPred
    sqNoise := sqNoise + etai * etai
    sqObsResid := sqObsResid + dObs * dObs
    covDof := covDof + etai * dPred
  let lhs := sqPredErr + sqNoise
  let rhs := sqObsResid + 2.0 * covDof
  return (lhs, rhs, Float.abs (lhs - rhs))

/-- Evaluates the Two-Regime Spatially-Adaptive Non-Linear Risk Advantage over the Stationary
    Linear Wiener Oracle (`Lucky/Limits.lean`, Theorem 15d `adaptive_nonlinear_strictly_beats_stationary_wiener`):
    Partitions pixels into Mare/Background (`A = Var_mare`) and Rim/Foreground (`B = Var_rim`)
    and returns `(R_stationary_wiener, R_adaptive_nonlinear, R_stationary - R_adaptive)`. -/
def computeTwoRegimeOracleGap2D (truth : Image2D) (sigmaNoise : Float) : Float × Float × Float := Id.run do
  let w := truth.width
  let h := truth.height
  let mut sumMare : Float := 0.0
  let mut cntMare : Float := 0.0
  let mut sumRim : Float := 0.0
  let mut cntRim : Float := 0.0
  for y in [0:h] do
    for x in [0:w] do
      let u0 := truth.get x y
      let gx := truth.getClamped ((x : Int) + 1) y - u0
      let gy := truth.getClamped x ((y : Int) + 1) - u0
      let locVar := gx * gx + gy * gy
      if locVar < 0.002 then
        sumMare := sumMare + locVar
        cntMare := cntMare + 1.0
      else
        sumRim := sumRim + locVar
        cntRim := cntRim + 1.0
  let A := 1000.0 * (sumMare / fmax 1.0 cntMare)
  let B := 1000.0 * (sumRim / fmax 1.0 cntRim)
  let N := 1000.0 * (sigmaNoise * sigmaNoise)
  let rStat := ((A + B) * (2.0 * N)) / (A + B + 2.0 * N)
  let rAdapt := (A * N) / (A + N) + (B * N) / (B + N)
  return (rStat, rAdapt, rStat - rAdapt)

/-- Generation-7 Parseval-SURE Two-Band Spatial-Spectral Lunar Deconvolution
    (`Lucky/Limits.lean`, Theorems 15m, 15n, and 15o `parseval_sure_two_band_mixing_risk_identity`):
    Combines:
      1. High-SNR Crater-Rim / Stripe Band `uHi` (aggressive Rényi-Escort Split-EM),
      2. Low-SNR Mare / Background Band `uLo` (conservative Rényi-Escort Split-EM),
      3. Balanced Pilot Band `uMid` (`runLunarGen6RenyiPCLR`),
      4. Self-Calibrating Parseval-SURE Spatial Mixing Weight
         `α_j = clip((D_j - c_SURE * σ_Δ²) / (D_j + ε), 0, 1)` where `D_j = smooth3x3((uHi - uLo)²)_j`
         and `σ_Δ²` is the empirical mare difference-band noise variance. -/
def runLunarGen7ParsevalSUREPCLR (psfFwd y : Image2D) (iters : Nat := 35)
    (alphaAdj : Float := 1.40) (satCeiling : Float := 0.995) : Image2D := Id.run do
  let targetFlux := y.totalFlux
  let n := y.pixels.size
  let uMid := runLunarGen6RenyiPCLR psfFwd y iters 1.38 satCeiling 0.055 0.0072 0.0035 0.009 0.007 0.008 0.011 1.44 0.42
  let uHi  := runLunarGen6RenyiPCLR psfFwd y iters alphaAdj satCeiling 0.052 0.0066 0.0032 0.009 0.007 0.008 0.0105 1.46 0.44
  let uLo  := runLunarGen6RenyiPCLR psfFwd y iters 1.35 satCeiling 0.062 0.0076 0.0038 0.009 0.007 0.008 0.0115 1.42 0.40
  let uSmooth := smooth3x3 uMid

  -- Band difference energy `d_j² = (uHi_j - uLo_j)²` and 3x3 window average `D_j` (Theorem 15o)
  let diffSq : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty n
    for i in [0:n] do
      let d := uHi.pixels.getD i 0.0 - uLo.pixels.getD i 0.0
      arr := arr.push (d * d)
    return arr⟩
  let locDiffSq := smooth3x3 diffSq

  -- Estimate Parseval difference-band noise floor `σ_Δ²` from flat background pixels
  let mut sumNoiseDiff : Float := 0.0
  let mut cntNoiseDiff : Float := 0.0
  for yIdx in [0:y.height] do
    for xIdx in [0:y.width] do
      let u0 := uSmooth.get xIdx yIdx
      let gx := uSmooth.getClamped ((xIdx : Int) + 1) yIdx - u0
      let gy := uSmooth.getClamped xIdx ((yIdx : Int) + 1) - u0
      if gx * gx + gy * gy < 0.0012 then
        sumNoiseDiff := sumNoiseDiff + locDiffSq.get xIdx yIdx
        cntNoiseDiff := cntNoiseDiff + 1.0
  let sigDeltaSq := sumNoiseDiff / fmax 1.0 cntNoiseDiff

  -- Parseval-SURE Two-Band Spatial Mixing + Pilot Fusion (Theorem 15o)
  let rawOut : Image2D := ⟨y.width, y.height, Id.run do
    let mut arr := Array.mkEmpty n
    for i in [0:n] do
      let dj2 := locDiffSq.pixels.getD i 0.0
      let alphaSURE := fmin 1.0 (fmax 0.0 ((dj2 - 0.85 * sigDeltaSq) / (dj2 + 1.0e-10)))
      let uhi := uHi.pixels.getD i 0.0
      let ulo := uLo.pixels.getD i 0.0
      let umid := uMid.pixels.getD i 0.0
      let uSure := alphaSURE * uhi + (1.0 - alphaSURE) * ulo
      arr := arr.push (0.35 * umid + 0.65 * uSure)
    return arr⟩
  return rawOut.normalizeFlux targetFlux

/-- Verifies the Law-of-Total-Variance Pilot Spectrum Restoration Identity (`Theorem 15m`)
    and Stein-Morozov Effective-DoF Discrepancy Target (`Theorem 15n`):
    Returns `(sPilRatio, sEmRestoredRatio, rvTarget)`. -/
def verifyLawOfTotalVarianceAndSteinDoF2D (psf : Image2D) (h2Mid s0Mid nVar : Float) : Float × Float × Float := Id.run do
  let denom := h2Mid * s0Mid + nVar
  let sPil := (h2Mid * s0Mid * s0Mid) / denom
  let vPost := (s0Mid * nVar) / denom
  let sEm := sPil + vPost
  let mut sumH2 : Float := 0.0
  for v in psf.pixels do
    sumH2 := sumH2 + v * v
  let rvTarget := 1.0 - 3.35 * sumH2
  return (sPil / s0Mid, sEm / s0Mid, rvTarget)

end Lucky
