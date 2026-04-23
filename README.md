# CHOAXIAL

CHOAXIAL is a compact Python toolkit for first-pass neutrino-scattering
feasibility studies with light nuclei and gas targets. In its current form the
repository is centered on $CF_4$, with built-in Standard-Model calculations for
coherent elastic neutrino-nucleus scattering (CEvNS) on $^{12}C$ and $^{19}F$,
neutrino-electron elastic scattering on the `42` electrons in a $CF_4$ molecule,
stopped-pion decay-at-rest (DAR) source models, and a simple detector-scaling
layer.

The code is meant for a preliminary physics evaluation, not for a final
experiment design report. A good way to think about it is: can this source,
target, and detector concept produce rates in the right ballpark to justify a
stronger ERC Starting Grant proposal? The repository helps answer that question
quickly by making the source assumptions, cross-section ingredients, target
composition, and detector normalization explicit and easy to inspect.

## Quick Start

Install the minimal dependencies:

```bash
pip install numpy matplotlib
```

Typical workflow:

```bash
# 1. Inspect a single CEvNS differential cross section
python cevens.py --target 19F --enu-mev 30 --er-kev 5 --json

# 2. Generate an ESS-like or J-PARC-like DAR flux benchmark
python ESS_flux.py

# 3. Fold the flux with the interaction model to get CF4 rates per molecule
python rate_estimation.py

# 4. Convert molecule-normalized rates into detector-level spectra and yearly counts
python detector_estimation.py --config configs/detector_config.json
```

Alternative source benchmark:

```bash
python JPARK_flux.py
```

This produces a standalone J-PARC MLF flux benchmark, but it is not yet wired
into the main rate pipeline in the same automatic way as the ESS-like source.

## What The Repository Does

At the moment the codebase provides four main layers:

1. `cevens.py`
   Interaction kernels for CEvNS and neutrino-electron scattering, including
   vector and axial pieces and basic $CF_4$ bookkeeping.
2. `ESS_flux.py` and `JPARK_flux.py`
   DAR source models for ESS-like and J-PARC MLF-like benchmarks, with CSV and
   plot export.
3. `rate_estimation.py`
   Flux-folded nuclear and electron recoil spectra per $CF_4$ molecule.
4. `detector_estimation.py`
   Detector-normalized rates for a cylindrical ideal-gas $CF_4$ target specified
   through JSON.

What it is good for:

- checking order-of-magnitude event rates
- comparing prompt and delayed source components
- separating $^{12}C$ and $^{19}F$ contributions inside $CF_4$
- estimating how relevant the built-in $^{19}F$ axial term is
- seeing how pressure, fiducial fraction, threshold, and geometry affect yearly
  yields

What it does not yet do:

- detector response, smearing, or efficiency modeling beyond a hard threshold
- beam timing cuts or pulse-window optimization
- background modeling
- uncertainty propagation
- automated parameter scans
- higher-order and two-body axial corrections for $^{19}F$

## Intended Role For An ERC Starting Proposal

This repository is best used as an early-stage support tool for a proposal
narrative. It lets you show that the source-target-detector concept has a
credible physics basis, that the relevant recoil scales are understood, and that
the expected rates are not obviously too small before investing in a full
sensitivity study.

In that sense, CHOAXIAL is not the final analysis stack. It is the layer that
helps decide whether the next steps are worth funding effort:

- add higher-order nuclear-response corrections and uncertainty estimates
- add detector response and backgrounds
- run systematic scans over pressure, threshold, geometry, and baseline
- translate rates into discovery reach, exclusion contours, or design trade-offs

## Workflow In One View

$\text{source model} \to \phi(E_\nu) \to \text{fold with } \frac{d\sigma}{dE} \to \text{rate per CF}_4\text{ molecule} \to \text{detector-scaled yearly spectrum}$

The implementation follows exactly this chain:

- `cevens.py` computes $\frac{d\sigma}{dE_r}$ and $\frac{d\sigma}{dT_e}$

- `ESS_flux.py` or `JPARK_flux.py` provides the DAR flux model
- `rate_estimation.py` computes $\frac{dR}{dE}$ per $CF_4$ molecule
- `detector_estimation.py` multiplies by the number of fiducial molecules and by
  live time

## Physics Model

### 1. CEvNS kernel

The CEvNS differential cross section in `cevens.py` is implemented as a vector
piece plus a pure axial piece [Sierra, candela 2026](https://arxiv.org/pdf/2603.05281):

$$\frac{d\sigma}{dE_r} = \frac{G_F^2 m_N}{4\pi} \left[ \left(1 - \frac{m_N E_r}{2E_\nu^2} - \frac{E_r}{E_\nu}\right) Q_W^2 |F_W(q^2)|^2 + \left(1 + \frac{m_N E_r}{2E_\nu^2} - \frac{E_r}{E_\nu}\right) F_A(q^2) \right]$$

with

$$q^2 = 2 m_N E_r$$
$$Q_W = Z(1 - 4\sin^2\theta_W) - N$$

Key implementation choices:

- $^{12}C$ is treated as spin zero, so the axial term is set to zero.
- $^{19}F$ is the only nucleus with a built-in axial contribution.
- the code keeps total, vector-only, and axial-only differential cross sections separate

#### Vector form factor

The vector form factor has two built-in options in `cevens.py`:

  **Helm (default):**

  $$F_W(q) = 3\,\frac{j_1(qR_n)}{qR_n}\,\exp\left[-\frac{(qs)^2}{2}\right]$$

  This is the standard Helm analytic form-factor parametrization introduced in [Helm 1956](https://link.aps.org/doi/10.1103/PhysRev.104.1466).

  with

  $$R_n^2 = c^2 + \frac{7}{3}\pi^2 a^2 - 5s^2$$

  and default parameters

  $$c = 1.23\,A^{1/3} - 0.60\,\text{fm},\quad a = 0.52\,\text{fm},\quad s = 0.90\,\text{fm}$$

  These default values follow the standard phenomenological choice widely used in recoil calculations. The charge-density systematics underlying the usual $c,\ a,\ s$ parameters are associated with [Friedrich & Vögler 1982](https://inspirehep.net/literature/1448782), while their practical use in recoil phenomenology was popularized by [Lewin & Smith 1996](https://www.sciencedirect.com/science/article/pii/S0927650596000473). The same Helm parametrization is also standard in CEvNS phenomenology; for example, it is used explicitly in [Chatterjee et al. 2023](https://link.aps.org/doi/10.1103/PhysRevD.107.055019).

  Unless the parameter $c$ is user-specified, the code uses the default values above. In the implementation, for $q^2 \le 0$ the code returns $F_W = 1$

  **Point-like (`--pointlike`):**

  $F_W(q^2) = 1$ for all $q^2$ (unity form factor).

#### Axial form factor

The generic axial structure is written as

$$F_A(q^2) = \frac{8\pi}{2J+1} \left[ (g_A^s)^2 S_{00}(q^2) - g_A g_A^s S_{01}(q^2) + g_A^2 S_{11}(q^2) \right]$$

For $^{19}F$, two Hoferichter-Menéndez-Schwenk axial levels are available:
[arXiv:2007.08529](https://arxiv.org/abs/2007.08529).

Both use the Appendix E / Table VIII $^{19}F$ shell-model polynomial-fit
responses for the proton and neutron $L = 1$ transverse operators
$F^{\Sigma_0}$ and $F^{\Sigma_{00}}$:

$$F(u) = \exp(-u/2)\sum_i c_i u^i,\qquad u = \frac{q^2 b^2}{2},\qquad b = 1.7623\,\text{fm}$$

- **Fast**: `axial_model="hoferichter_19f_fast"` uses the polynomial-fit
  transverse shell-model response with $g_A = 1.27641$, $g_A^{s,N} = 0$, and
  $\delta_0(q^2)=\delta_{00}(q^2)=0$. This is intended for fast feasibility
  scans.
- **Central detailed [Default]**: `axial_model="hoferichter_19f_central"` uses the same
  response basis, $g_A^{s,N}=-0.085$, and one central prescription for the
  Eq. (86) correction functions $\delta_0(q^2)$ and $\delta_{00}(q^2)$ using
  Table I and Table V inputs.

Both levels form $S_{00}^T$, $S_{01}^T$, and $S_{11}^T$ following
Eqs. (80)-(85), then feed them into $F_A(q^2)$ above. Neither level is an
uncertainty-band treatment; scans over $\rho$, $c_D$, and related nuclear
inputs are a later step.

The old dipole-like spin-expectation suppression is no longer the default,
because it is not directly supported as a standard $^{19}F$ CEvNS nuclear-response
model. A toy/testing axial model remains available for debugging and
benchmarking, but it is not proposal-grade physics.

In the current scripts, the default $^{19}F$ axial model is
`hoferichter_19f_fast`. To change it:

- for one-point checks, pass `--axial-model hoferichter_19f_central`, `none`, or
  `toy` to `python cevens.py ...`
- for the full $CF_4$ rate scan, edit `fluorine_axial_model` in
  `RateConfig` inside `rate_estimation.py`
- in Python code, call `fluorine19_target(axial_model="hoferichter_19f_central")`
  or another supported option

### 2. Neutrino-electron scattering

`cevens.py` also includes the Standard-Model free-electron elastic-scattering kernel for electron recoils. In the low-energy limit used here, the differential cross section is written as

$$ \frac{d\sigma}{dT_e} = \frac{2 G_F^2 m_e}{\pi} \left[ g_{\mathrm{lead}}^2 + g_{\mathrm{sub}}^2 (1-y)^2 - \frac{g_L g_R m_e T_e}{E_\nu^2} \right], $$

for neutrinos, with

$$
y = \frac{T_e}{E_\nu}.
$$

For antineutrinos, the same expression is used with the standard interchange $g_L \leftrightarrow g_R,$



This is the standard tree-level $\nu$--$e$ elastic-scattering form used in low-energy neutrino phenomenology. The notation is fully consistent: $g_L$ and $g_R$ always denote the underlying left- and right-handed electron couplings, while the difference between neutrinos and antineutrinos is implemented only through which coupling multiplies the leading term and which one multiplies the $(1-y)^2$ term. In other words, the couplings themselves do not change meaning; only their placement in the cross section changes between $\nu e$ and $\bar{\nu} e$ scattering.

For $\nu_\mu e$ and $\nu_\tau e$ scattering, the code uses the Standard-Model neutral-current couplings

$$
g_L = -\frac{1}{2} + \sin^2\theta_W,
\qquad
g_R = \sin^2\theta_W.
$$

For $\nu_e e$ scattering, the charged-current contribution is included through the standard replacement

$$
g_L \to g_L + 1,
$$

so that

$$
g_L = \frac{1}{2} + \sin^2\theta_W,
\qquad
g_R = \sin^2\theta_W.
$$

Accordingly, the larger $\nu_e e$ cross section does not come from a different formula, but from the same formula evaluated with the charged-current-enhanced left-handed coupling. This is the standard low-energy treatment; see, for example, Vogel and Engel, *Phys. Rev. D* **39** (1989) 3378, and the PDG review of neutrino cross sections:
[https://users.physics.unc.edu/~engelj/papers/ve.pdf](https://users.physics.unc.edu/~engelj/papers/ve.pdf),
[https://pdg.lbl.gov/2023/reviews/rpp2023-rev-nu-cross-sections.pdf](https://pdg.lbl.gov/2023/reviews/rpp2023-rev-nu-cross-sections.pdf).

For $CF_4$, the molecule-level rate is obtained by multiplying the single-electron cross section by the total number of target electrons in one molecule,

$$
Z_{\mathrm{tot}} = 6 + 4 \times 9 = 42.
$$

Accordingly, the repository keeps both the per-electron and per-molecule normalizations explicit when building electron-recoil spectra.

### 3. DAR source model

Both source modules implement the standard stopped-pion decay-at-rest (DAR) chain,

$$
\pi^+ \to \mu^+ + \nu_\mu,
$$

followed by

$$
\mu^+ \to e^+ + \nu_e + \bar{\nu}_\mu.
$$

This is the standard source model used for spallation-based low-energy neutrino beams such as J-PARC MLF, and it is also the basis of the ESS-style benchmark adopted here. In this approximation, the prompt $\nu_\mu$ component comes from $\pi^+$ decay at rest and is therefore monochromatic, while the delayed $\nu_e$ and $\bar{\nu}_\mu$ components come from $\mu^+$ decay at rest and follow the usual Michel spectra. :contentReference[oaicite:0]{index=0}

For the prompt component, the neutrino energy is fixed by two-body pion decay at rest,

$$
E_{\nu_\mu}^{\mathrm{prompt}} = \frac{m_\pi^2-m_\mu^2}{2m_\pi} \simeq 29.79\ \mathrm{MeV},
$$

while the delayed components extend up to the muon-decay endpoint

$$
E_{\max} = \frac{m_\mu}{2} \simeq 52.83\ \mathrm{MeV}.
$$

In both `ESS_flux.py` and `JPARK_flux.py`, these values are computed directly from the charged-pion and muon masses and used as the prompt line energy and delayed-spectrum endpoint. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

The delayed $\nu_e$ and $\bar{\nu}_\mu$ spectra are implemented as normalized Michel distributions,

$$
f_{\nu_e}(E) = \frac{192}{m_\mu} \left(\frac{E}{m_\mu}\right)^2 \left(\frac{1}{2}-\frac{E}{m_\mu}\right),
$$

$$
f_{\bar{\nu}_\mu}(E) = \frac{64}{m_\mu} \left(\frac{E}{m_\mu}\right)^2 \left(\frac{3}{4}-\frac{E}{m_\mu}\right),
$$

with support only in the interval $0 \le E \le m_\mu/2$. The code defines these as normalized differential shapes in units of $\mathrm{MeV}^{-1}$ and then multiplies them by the source normalization and geometry factor to obtain physical fluxes. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

The source normalization is handled in a deliberately simple benchmark form. In both scripts, the average proton rate is obtained from beam power divided by proton kinetic energy,

$$
\dot N_p = \frac{P}{E_p},
$$

after converting the proton energy to joules. A neutrino yield per proton per flavor, $y_\nu$, is then assumed, so that the source rate per flavor is

$$
\dot N_\nu = \dot N_p \, y_\nu.
$$

For the ESS benchmark, the current script uses a design-style configuration with $P=5\ \mathrm{MW}$, $E_p=2\ \mathrm{GeV}$, repetition rate $f=14\ \mathrm{Hz}$, pulse length $\tau=2.86\ \mathrm{ms}$, and yield $y_\nu=0.3$ neutrinos per proton per flavor. For the J-PARC MLF benchmark, the script uses $P=1\ \mathrm{MW}$, $E_p=3\ \mathrm{GeV}$, repetition rate $f=25\ \mathrm{Hz}$, two bunches per spill, bunch width $100\ \mathrm{ns}$, and yield $y_\nu=0.48$ neutrinos per proton per flavor. These are benchmark inputs used by the repository, not first-principles hadron-production calculations. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

The transport from source to detector is modeled with a simple isotropic point-source geometry factor,

$$
\phi \propto \frac{1}{4\pi L^2},
$$

implemented explicitly as

$$
\frac{1}{4\pi r^2},
$$

with $r$ converted to centimeters so that the flux is returned in $\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$ or $\mathrm{cm}^{-2}\,\mathrm{POT}^{-1}$ units as appropriate. Delayed components are therefore computed as

$$ \phi_{\mathrm{delayed}}(E) = \dot N_\nu \, \frac{1}{4\pi L^2}\, f(E),
$$

while the prompt $\nu_\mu$ line is first computed as an integrated line intensity and then, when needed for histogrammed outputs, deposited into the energy bin containing $E_{\nu_\mu}^{\mathrm{prompt}}$ by dividing the line intensity by that bin width. This is how the scripts build both smooth delayed spectra and binned total spectra including the monochromatic prompt contribution. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

`ESS_flux.py` produces average differential fluxes in units of neutrinos$/(\mathrm{cm}^2\,\mathrm{s}\,\mathrm{MeV})$ for an ESS-style DAR benchmark. `JPARK_flux.py` provides both average fluxes and per-POT fluences, so that the J-PARC benchmark can be used either as a time-averaged source model or as a proton-normalized source description. In the current repository structure, the ESS benchmark is the one directly wired into the main rate-estimation chain, while the J-PARC script is kept as a separate benchmark generator. :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

### 4. Flux folding and detector scaling

`rate_estimation.py` computes the differential event-rate spectrum as

$$\frac{dR}{dE_r} = \int dE_\nu \, \phi(E_\nu) \cdot \frac{d\sigma}{dE_r}$$

with an explicit prompt contribution for the monochromatic $\nu_\mu$ line and separate delayed contributions from $\nu_e$ and $\bar{\nu}_\mu$ For $CF_4$, the code keeps the molecule composition explicit:

$$\text{CF}_4 = 1 \times {}^{12}\text{C} + 4 \times {}^{19}\text{F}$$

The detector layer then applies ideal-gas bookkeeping:
$V = \pi R^2 L$,
$n_{\text{moles}} = \frac{PV}{R_{\text{gas}} T}$,
$N_{\text{molecules}} = n_{\text{moles}} \cdot N_A \cdot f_{\text{fiducial}}$,
and converts the molecule-normalized differential rate into detector-level
spectra and yearly counts:

$$\frac{dN}{dE} = N_{\text{molecules}} \cdot \frac{dR}{dE} \cdot t_{\text{exposure}}$$

The only analysis cut currently implemented is a hard recoil-energy threshold.

## Repository Structure

```text
.
├── cevens.py
├── ESS_flux.py
├── JPARK_flux.py
├── rate_estimation.py
├── detector_estimation.py
├── configs/
│   └── detector_config.json
├── GMO_exam/
│   └── simple_estimates.ipynb
├── ess_flux_output/             # autogenerated
├── jparc_mlf_flux_output/       # autogenerated
├── cevens_rate_output/          # autogenerated
├── detector_rate_output/        # autogenerated
└── README.md
```

Module-by-module summary:

- `cevens.py`
  Core physics module. Defines nuclear targets, form factors, axial models, the
  CEvNS calculator, the neutrino-electron calculator, and a small CLI for
  single-point cross-section checks.
- `ESS_flux.py`
  ESS-like DAR source benchmark. Writes point-grid and binned flux CSV/PNG
  outputs in `ess_flux_output/`.
- `JPARK_flux.py`
  J-PARC MLF flux benchmark. Writes average-flux and per-POT fluence outputs in
  `jparc_mlf_flux_output/`.
- `rate_estimation.py`
  Builds $^{12}C$, $^{19}F$, and $CF_4$ electron targets, folds the ESS-like flux with
  the interaction kernels, and writes per-molecule recoil spectra and summary
  plots to `cevens_rate_output/`.
- `detector_estimation.py`
  Reads the rate CSVs, parses the detector JSON, computes fiducial $CF_4$
  inventory, writes detector-level spectra, and exports a compact summary JSON.
- `configs/detector_config.json`
  Example detector benchmark with cylindrical geometry, fiducial fraction, gas
  pressure, temperature, and threshold.
- `GMO_exam/simple_estimates.ipynb`
  Exploratory notebook outside the main scripted pipeline.

## How To Use The Code

### Single-point interaction checks

Use `cevens.py` when you want to inspect a specific CEvNS differential cross
section value before running the full chain:

```bash
python cevens.py --target 19F --enu-mev 30 --er-kev 5
python cevens.py --target 19F --enu-mev 30 --er-kev 5 --json
python cevens.py --target 19F --enu-mev 30 --er-kev 5 --axial-model none
python cevens.py --target 19F --enu-mev 30 --er-kev 5 --axial-model toy
python cevens.py --target 12C --enu-mev 30 --er-kev 5 --pointlike
```

This is the quickest way to check kinematics, weak charge, form-factor value,
and the relative size of vector and axial pieces.

### Source benchmarks

Run:

```bash
python ESS_flux.py
python JPARK_flux.py
```

Use `ESS_flux.py` for the current end-to-end rate chain. Use `JPARK_flux.py` as
an alternative source benchmark or as a starting point for a future J-PARC rate
driver.

### Molecule-normalized rates

Run:

```bash
python rate_estimation.py
```

This produces:

- CEvNS differential rates per $CF_4$ molecule
- neutrino-electron differential rates per electron and per $CF_4$ molecule
- separate prompt, delayed, and total components
- decomposition into $^{12}C$ and $^{19}F$
- vector and axial separation for $^{19}F$

Important normalization:

- outputs are rates per molecule, not per detector
- the ESS-like flux is the source model used in the current main rate script

### Detector-level spectra

Run:

```bash
python detector_estimation.py --config configs/detector_config.json
```

The detector config currently supports:

- `radius_m` or `diameter_m`
- `length_m` or `height_m`
- `fiducial_fraction`
- one gas-pressure field among `pressure_pa`, `pressure_kpa`, `pressure_mbar`,
  `pressure_bar`, `pressure_torr`, or `pressure_atm`
- `temperature_K`
- `analysis.energy_threshold_kev`

The script writes detector-level nuclear and electron recoil CSVs, plots, and a
summary JSON with integrated rates above threshold.

## Output Files

The most relevant autogenerated products are:

- `ess_flux_output/`
  ESS-like point-grid and binned flux tables and plots
- `jparc_mlf_flux_output/`
  J-PARC average-flux and per-POT fluence tables and plots
- `cevens_rate_output/cf4_differential_rate_per_molecule.csv`
  CEvNS spectra per molecule, including the $^{12}C$/$^{19}F$ decomposition
- `cevens_rate_output/cf4_electron_differential_rate_per_molecule.csv`
  neutrino-electron recoil spectra per electron and per molecule
- `detector_rate_output/cf4_detector_summary.json`
  compact detector-level summary for proposal notes and quick comparisons

## Units And Conventions

The code uses practical phenomenology units throughout:

- neutrino energy: $\mathrm{MeV}$

- nuclear recoil energy: $\mathrm{keV}$

- electron recoil energy: $\mathrm{keV}$

- nuclear masses: $\mathrm{GeV}$ internally

- CEvNS differential cross section: $\mathrm{cm}^2/\mathrm{keV}$

- delayed flux: $\mathrm{neutrinos}/(\mathrm{cm}^2\,\mathrm{s}\,\mathrm{MeV})$

- prompt line flux: $\mathrm{neutrinos}/(\mathrm{cm}^2\,\mathrm{s})$

- molecule-normalized rates: $\mathrm{s}^{-1}\,\mathrm{keV}^{-1}\,\mathrm{molecule}^{-1}$

- detector-level spectra: $\mathrm{events}/(\mathrm{keV}\,\mathrm{year})$

## Current Assumptions And Limitations

- The physics scope is Standard Model only.
- $^{19}F$ uses the fast Hoferichter-Menéndez-Schwenk polynomial-fit transverse
  axial response by default; the central detailed option includes one
  correction prescription but not uncertainty bands.
- The main rate chain is presently tied to the ESS-like source benchmark.
- The detector model assumes ideal-gas $CF_4$.
- Thresholding is a hard cut with no efficiency curve.
- No backgrounds, acceptance effects, live-time losses, timing cuts, or energy
  smearing are included.
- No automated tests or package metadata are included yet.

These limitations are acceptable for a first-pass feasibility study, but they
become the main items to upgrade before turning this into a proposal-grade
forecast tool.

## Suggested Next Steps

If the preliminary rates look encouraging, the natural follow-up work is:

1. Add the higher-order, radius, and two-body axial correction machinery for
   $^{19}F$, and attach nuclear-response uncertainties.
2. Extend `rate_estimation.py` so the same pipeline can run directly with the
   J-PARC flux benchmark as well as the ESS one.
3. Add detector response: threshold efficiency, quenching choices if needed,
   energy resolution, and acceptance.
4. Add beam timing and background models so prompt and delayed windows can be
   exploited realistically.
5. Introduce scan drivers over pressure, threshold, baseline, and detector size
   to map useful design space for the proposal.
6. Add uncertainty bookkeeping and a lightweight test suite so benchmark numbers
   are reproducible and easier to defend.

## References And Context

These are the main theory and source-model references that match the structure
already implemented in the code:

1. D. Z. Freedman, ["Coherent effects of a weak neutral current"](
   https://www.osti.gov/biblio/4288911), Phys. Rev. D 9 (1974) 1389.
2. R. H. Helm, ["Inelastic and Elastic Scattering of 187-Mev Electrons from
   Selected Even-Even Nuclei"](
   https://journals.aps.org/pr/abstract/10.1103/PhysRev.104.1466), Phys. Rev.
   104 (1956) 1466.
3. P. Vogel and J. Engel, ["Neutrino electromagnetic form factors"](
   https://journals.aps.org/prd/abstract/10.1103/PhysRevD.39.3378), Phys. Rev.
   D 39 (1989) 3378.
   For the Standard-Model neutrino-electron scattering conventions commonly used
   in low-energy phenomenology.
4. JSNS2 Collaboration, ["Technical Design Report: Searching for a Sterile
   Neutrino at J-PARC MLF"](https://arxiv.org/abs/1705.08629),
   arXiv:1705.08629.
5. Standard stopped-pion and muon-DAR neutrino spectra as encoded in the Michel
   formulas implemented in `ESS_flux.py` and `JPARK_flux.py`.

For proposal work, these references should be supplemented with whatever nuclear
structure and detector-performance inputs are chosen for the final benchmark.
