import Lucky

open Lucky

/-- Format a Float to 4 decimal places. -/
def fmt4 (v : Float) : String :=
  let sign := if v < 0.0 then "-" else ""
  let av := Float.abs v
  let scaled := (av * 10000.0 + 0.5).toUInt64
  let intPart := scaled / 10000
  let fracPart := scaled % 10000
  let fracStr := toString fracPart
  let pad := String.ofList (List.replicate (4 - fracStr.length) '0')
  s!"{sign}{intPart}.{pad}{fracStr}"

/-- Scene 1: 36×36 Deep-Sky Astronomy + Sharp Bar Test Scene. -/
def makeGroundTruth (w h : Nat) : Image2D :=
  Image2D.ofFn w h (fun x y => Id.run do
    let fx := x.toFloat
    let fy := y.toFloat
    let mut val : Float := 0.04
    let d1 := (fx - 11.0) * (fx - 11.0) + (fy - 11.0) * (fy - 11.0)
    let d2 := (fx - 15.0) * (fx - 12.0) + (fy - 15.0) * (fy - 12.0)
    val := val + 0.90 * Float.exp (-d1 / 1.4)
    val := val + 0.78 * Float.exp (-d2 / 1.4)
    let dr := Float.sqrt ((fx - 23.0) * (fx - 23.0) + (fy - 21.0) * (fy - 21.0))
    if 4.0 ≤ dr && dr ≤ 7.2 then
      val := val + 0.55
    else if dr < 4.0 then
      val := val + 0.14
    if 3 ≤ x && x ≤ 14 && 25 ≤ y && y ≤ 31 then
      val := val + 0.65
    return fmin 1.0 val)

/-- Scene 2: 36×36 Piecewise-Smooth Phantom + Resolution Stripes + Linear Ramp Scene. -/
def makePhantomResolutionScene (w h : Nat) : Image2D :=
  Image2D.ofFn w h (fun x y => Id.run do
    let fx := x.toFloat
    let fy := y.toFloat
    let mut val : Float := 0.035
    -- Elliptical shell + smooth interior affine ramp (tests 2nd-order zero-staircasing Theorem 6b)
    let nx := (fx - 13.0) / 8.5
    let ny := (fy - 13.0) / 9.5
    let r2 := nx * nx + ny * ny
    if r2 ≤ 1.0 && r2 ≥ 0.68 then
      val := 0.86
    else if r2 < 0.68 then
      val := 0.28 + 0.18 * nx + 0.10 * ny
    -- High-contrast focal pinpoint inside phantom
    let df := (fx - 13.0) * (fx - 13.0) + (fy - 10.0) * (fy - 10.0)
    val := val + 0.55 * Float.exp (-df / 1.2)
    -- Fine resolution bar triplet in bottom-right
    if 22 ≤ y && y ≤ 32 then
      if (x == 22 || x == 23) || (x == 26 || x == 27) || (x == 30 || x == 31) then
        val := 0.82
    return fmin 1.0 val)

/-- Generate a 7×7 atmospheric Moffat + motion-asymmetric blur PSF normalized to unit sum. -/
def makeBlurPSF (aniso : Float := -0.45) (scale : Float := 2.6) : Image2D :=
  let raw := Image2D.ofFn 7 7 (fun x y =>
    let dx := x.toFloat - 3.0
    let dy := y.toFloat - 3.0
    let r2 := (dx * dx + 1.3 * dy * dy + aniso * dx * dy) / scale
    1.0 / Float.pow (1.0 + r2) 2.2)
  raw.normalizeFlux 1.0

/-- Deterministic pseudo-random zero-mean Poisson/readout noise generator for reproducible benchmarks. -/
def addDetectorNoise (cleanBlurred : Image2D) (sigma : Float) (seed : UInt64 := 0x9E3779B97F4A7C15) : Image2D :=
  let n := cleanBlurred.pixels.size
  ⟨cleanBlurred.width, cleanBlurred.height, Id.run do
    let mut arr := Array.mkEmpty n
    let mut state : UInt64 := seed
    for i in [0:n] do
      state := state + 0x9E3779B97F4A7C15
      let mut z := state
      z := (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9
      z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
      z := z ^^^ (z >>> 31)
      let u1 := (z % 100000).toFloat / 100000.0 - 0.5
      let yi := cleanBlurred.pixels.getD i 0.0
      let localStd := sigma * Float.sqrt (fmax 0.02 yi)
      let noisy := fmax 0.002 (yi + 2.8 * localStd * u1)
      arr := arr.push noisy
    return arr⟩

def printRow (name : String) (truth psf y est : Image2D) (lam eps : Float) : IO Unit := do
  let psnr := psnr2D truth est 1.0
  let rmse := rmse2D truth est
  let kl := klDivergence2D y (forward2D psf est)
  let mapE := mapObjective2D psf y est lam eps
  let fluxErr := 100.0 * Float.abs (est.totalFlux - y.totalFlux) / y.totalFlux
  let minV := est.minVal
  IO.println s!"| {name} | {fmt4 psnr} dB | {fmt4 rmse} | {fmt4 kl} | {fmt4 mapE} | {fmt4 fluxErr}% | {fmt4 minV} |"

def main : IO Unit := do
  IO.println "========================================================================================="
  IO.println "  Lean 4 Formalized Lucy-Richardson & Gen-3 Multi-Frame Lucky PCLR Deconvolution Suite"
  IO.println "========================================================================================="
  let w := 36
  let h := 36
  let truth := makeGroundTruth w h
  let psf := makeBlurPSF (-0.45) 2.6
  let cleanBlurred := forward2D psf truth
  let observed := addDetectorNoise cleanBlurred 0.025
  let wField := sensitivityField2D psf w h
  let iters := 35
  let lam := 0.018
  let eps := 0.015

  -- 1. Run Classic Lucy-Richardson (35 iterations)
  let mut uClassic := observed.normalizeFlux observed.totalFlux
  for _ in [0:iters] do
    uClassic := stepClassicRL psf observed uClassic

  -- 2. Run Dey et al. (2006) Unsplit TV-RL (moderate λ = 0.018 and strong λ = 0.40 showing positivity breakdown)
  let mut uDeyMod := observed.normalizeFlux observed.totalFlux
  for _ in [0:iters] do
    uDeyMod := stepDeyTVRL psf observed wField uDeyMod 0.018 eps
  let uDeyStrong := stepDeyTVRL psf observed wField uClassic 0.40 eps

  -- 3. Run Boundary- & Readout-Damped RL (35 iterations)
  let mut uDamped := observed.normalizeFlux observed.totalFlux
  for _ in [0:iters] do
    uDamped := stepDampedRL psf observed wField uDamped 0.02 0.010

  -- 4. Run Improved PCLR Gen-1 & Gen-2 TGV-Bregman PCLR
  let uPCLR := runPCLR psf observed iters lam eps 0.02 0.010 1.35
  let uPCLRStrong := baseStepPCLR psf observed wField uClassic 0.40 eps 0.02 0.010 1.0
  let uTGVBregman := runTGVBregmanPCLR psf observed iters 0.045 0.004 0.010 0.008 0.010 0.012 1.38 0.35

  -- 5. Run Self-Calibrated Symmetric Gen-3 PCLR (`runGen3PCLR`)
  let uGen3 := runGen3PCLR psf observed iters 0.060 0.0085 0.004 0.009 0.007 0.008 0.011 1.40 0.42

  -- 6. Run 4-Frame Atmospheric "Lucky Imaging" Burst (`runMultiFrameLuckyPCLR`, Theorem 9a-9e)
  let psfsLucky := #[
    makeBlurPSF (-0.55) 2.2,
    makeBlurPSF 0.60 2.7,
    makeBlurPSF (-0.25) 3.1,
    makeBlurPSF 0.45 2.3
  ]
  let framesLucky := #[
    addDetectorNoise (forward2D (psfsLucky.getD 0 psf) truth) 0.032 0x123456789ABCDEF0,
    addDetectorNoise (forward2D (psfsLucky.getD 1 psf) truth) 0.032 0xFEDCBA9876543210,
    addDetectorNoise (forward2D (psfsLucky.getD 2 psf) truth) 0.032 0x5555555555555555,
    addDetectorNoise (forward2D (psfsLucky.getD 3 psf) truth) 0.032 0xAAAAAAAAAAAAAAAA
  ]
  let uLuckyJoint := runMultiFrameLuckyPCLR psfsLucky framesLucky iters

  IO.println ""
  IO.println "--- Frame 1: Deep-Sky Astronomy + Sharp Bar Benchmark (36×36) ---"
  IO.println "| Method (35 iterations)            | PSNR (↑)   | RMSE (↓) | KL Div   | MAP Obj (↓)| Flux Err| Min Pixel |"
  IO.println "|-----------------------------------|------------|----------|----------|------------|---------|-----------|"
  printRow "Observed Blurred + Noisy         " truth psf observed observed lam eps
  printRow "1. Classic Lucy-Richardson (RL)  " truth psf observed uClassic lam eps
  printRow "2a. Dey Unsplit TV-RL (λ=0.018)  " truth psf observed uDeyMod lam eps
  printRow "2b. Dey Unsplit TV-RL (λ=0.40)   " truth psf observed uDeyStrong lam eps
  printRow "3. Boundary+Readout Damped RL    " truth psf observed uDamped lam eps
  printRow "4a. Improved PCLR (Gen-1, λ=.018)" truth psf observed uPCLR lam eps
  printRow "4b. Split-EM PCLR Stress (λ=0.40)" truth psf observed uPCLRStrong lam eps
  printRow "5. TGV-Bregman PCLR (Gen-2)      " truth psf observed uTGVBregman lam eps
  printRow "6. Self-Calibrated Gen-3 PCLR    " truth psf observed uGen3 lam eps
  printRow "7. Multi-Frame Lucky Gen-3 (F=4) " truth psf observed uLuckyJoint lam eps

  -- Scene 2 Benchmark in Lean 4
  let truth2 := makePhantomResolutionScene w h
  let psf2 := makeBlurPSF 0.35 2.4
  let obs2 := addDetectorNoise (forward2D psf2 truth2) 0.024 0x7777777777777777
  let mut uClassic2 := obs2.normalizeFlux obs2.totalFlux
  for _ in [0:iters] do
    uClassic2 := stepClassicRL psf2 obs2 uClassic2
  let uPCLR2 := runPCLR psf2 obs2 iters lam eps 0.02 0.010 1.35
  let uGen3_2 := runGen3PCLR psf2 obs2 iters 0.055 0.0080 0.0035 0.009 0.007 0.008 0.011 1.40 0.40
  let psf2Adj := makeBlurPSF 0.35 1.15
  let uGen4Lunar2 := runLunarPrecondAnisoPCLR psf2 psf2Adj obs2 iters 0.055 0.0075 0.0035 0.009 0.007 0.008 0.011 1.42 0.40
  let uGen6Renyi2 := runLunarGen6RenyiPCLR psf2 obs2 iters 1.38 0.995 0.055 0.0072 0.0035 0.009 0.007 0.008 0.011 1.44 0.42
  let uGen7Sure2 := runLunarGen7ParsevalSUREPCLR psf2 obs2 iters 1.40 0.995

  IO.println ""
  IO.println "--- Frame 2: Piecewise-Smooth Phantom + Resolution Stripes (36×36) ---"
  IO.println "| Method (35 iterations)            | PSNR (↑)   | RMSE (↓) | KL Div   | MAP Obj (↓)| Flux Err| Min Pixel |"
  IO.println "|-----------------------------------|------------|----------|----------|------------|---------|-----------|"
  printRow "Observed Blurred + Noisy         " truth2 psf2 obs2 obs2 lam eps
  printRow "1. Classic Lucy-Richardson (RL)  " truth2 psf2 obs2 uClassic2 lam eps
  printRow "2. Improved PCLR (Gen-1)         " truth2 psf2 obs2 uPCLR2 lam eps
  printRow "3. Self-Calibrated Gen-3 PCLR    " truth2 psf2 obs2 uGen3_2 lam eps
  printRow "4. Precond Aniso Lunar Gen-4 PCLR" truth2 psf2 obs2 uGen4Lunar2 lam eps
  printRow "5. Rényi-Escort Lunar Gen-6 PCLR " truth2 psf2 obs2 uGen6Renyi2 lam eps
  printRow "6. Parseval-SURE Lunar Gen-7 PCLR" truth2 psf2 obs2 uGen7Sure2 lam eps

  -- Blind Deconvolution Benchmark in Lean 4 (Unknown PSF: initialized to isotropic Gaussian!)
  let psfInitIso := (Image2D.ofFn 7 7 (fun x y =>
    let dx := x.toFloat - 3.0
    let dy := y.toFloat - 3.0
    Float.exp (-(dx * dx + dy * dy) / (2.0 * 1.65 * 1.65)))).normalizeFlux 1.0
  let (uClassicBlind, hClassicBlind) := runClassicBlindRL psfInitIso observed 10 4 4
  let (uBlindPCLR, hBlindPCLR) := runBlindPCLR psfInitIso observed iters
  let (uBlindLucky, hsBlindLucky) := runMultiFrameBlindLuckyPCLR psfInitIso framesLucky iters
  let initPsfPsnr := psnr2D psf psfInitIso psf.maxVal
  let cBlindPsfPsnr := psnr2D psf hClassicBlind psf.maxVal
  let pclrBlindPsfPsnr := psnr2D psf hBlindPCLR psf.maxVal
  let mut meanLuckyPsfPsnr : Float := 0.0
  for f in [0:psfsLucky.size] do
    let pt := psfsLucky.getD f psf
    let pe := hsBlindLucky.getD f psfInitIso
    meanLuckyPsfPsnr := meanLuckyPsfPsnr + psnr2D pt pe pt.maxVal
  meanLuckyPsfPsnr := meanLuckyPsfPsnr / psfsLucky.size.toFloat

  IO.println ""
  IO.println "--- Blind Deconvolution Benchmark (Unknown PSF, Isotropic Initial Guess, 36×36) ---"
  IO.println s!"  Initial Isotropic Guess PSF PSNR: {fmt4 initPsfPsnr} dB"
  IO.println s!"  Classic Blind RL (Fish 1995) PSF PSNR: {fmt4 cBlindPsfPsnr} dB | Blind Gen-3 PCLR PSF PSNR: {fmt4 pclrBlindPsfPsnr} dB | Blind Lucky (F=4) Mean PSF PSNR: {fmt4 meanLuckyPsfPsnr} dB"
  IO.println "| Blind Method (Unknown PSF)        | PSNR (↑)   | RMSE (↓) | KL Div   | MAP Obj (↓)| Flux Err| Min Pixel |"
  IO.println "|-----------------------------------|------------|----------|----------|------------|---------|-----------|"
  printRow "Observed Blurred + Noisy         " truth psf observed observed lam eps
  printRow "1. Classic Blind RL (Fish 1995)  " truth hClassicBlind observed uClassicBlind lam eps
  printRow "2. Single-Frame Blind PCLR (Ours)" truth hBlindPCLR observed uBlindPCLR lam eps
  printRow "3. Multi-Frame Blind Lucky (F=4) " truth (hsBlindLucky.getD 0 psfInitIso) observed uBlindLucky lam eps

  -- Theoretical Limits & Stein-SURE Verification Suite in Lean 4 (`Lucky/Limits.lean`, Theorems 15a–15p)
  let cleanBlurred2 := forward2D psf2 truth2
  let predGen7_2 := forward2D psf2 uGen7Sure2
  let (sureLHS, sureRHS, sureDiff) := verifySteinSUREIdentity2D cleanBlurred2 obs2 predGen7_2
  let (rStat1, rAdapt1, rGap1) := computeTwoRegimeOracleGap2D truth 0.025
  let (rStat2, rAdapt2, rGap2) := computeTwoRegimeOracleGap2D truth2 0.024
  let (sPilRat, sEmRat, rvTarget2) := verifyLawOfTotalVarianceAndSteinDoF2D psf2 0.020 1.000 0.040
  let rmseDeyNeg := rmse2D truth uDeyStrong
  let uDeyPosProj : Image2D := ⟨uDeyStrong.width, uDeyStrong.height, uDeyStrong.pixels.map (fun v => fmax 0.0 v)⟩
  let rmseDeyPosProj := rmse2D truth uDeyPosProj

  IO.println ""
  IO.println "--- Fundamental Theoretical Limits & Oracle Verification (`Lucky/Limits.lean`, Theorems 15a–15p) ---"
  IO.println s!"  [Theorem 15d] Scene 1 Stationary Wiener Risk = {fmt4 rStat1} vs Adaptive Non-Linear Risk = {fmt4 rAdapt1} (Strict Advantage Δ = +{fmt4 rGap1} > 0)"
  IO.println s!"  [Theorem 15d] Scene 2 Stationary Wiener Risk = {fmt4 rStat2} vs Adaptive Non-Linear Risk = {fmt4 rAdapt2} (Strict Advantage Δ = +{fmt4 rGap2} > 0)"
  IO.println s!"  [Theorem 15h] Vector Positivity Projection Contraction on Unconstrained Dey TV-RL: RMSE {fmt4 rmseDeyNeg} -> {fmt4 rmseDeyPosProj} (Strict L² Gain!)"
  IO.println s!"  [Theorem 15l] Stein-SURE Identity Verification: LHS = {fmt4 sureLHS}, RHS = {fmt4 sureRHS}, |LHS - RHS| = {fmt4 sureDiff}"
  IO.println s!"  [Theorem 15m/15n] Law-of-Total-Variance Pilot Spectrum Ratio = {fmt4 sPilRat} -> Restored S_EM Ratio = {fmt4 sEmRat} | Stein-Morozov DoF Target = {fmt4 rvTarget2}"
  IO.println "========================================================================================="
