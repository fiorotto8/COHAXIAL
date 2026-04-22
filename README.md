*# CHOAXIAL

CHOAXIAL is a compact Python toolkit for first-pass neutrino-scattering
feasibility studies with light nuclei and gas targets. In its current form the
repository is centered on `CF4`, with built-in Standard-Model calculations for
coherent elastic neutrino-nucleus scattering (CEvNS) on `12C` and `19F`,
neutrino-electron elastic scattering on the `42` electrons in a `CF4` molecule,
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
   vector and axial pieces and basic `CF4` bookkeeping.
2. `ESS_flux.py` and `JPARK_flux.py`
   DAR source models for ESS-like and J-PARC MLF-like benchmarks, with CSV and
   plot export.
3. `rate_estimation.py`
   Flux-folded nuclear and electron recoil spectra per `CF4` molecule.
4. `detector_estimation.py`
   Detector-normalized rates for a cylindrical ideal-gas `CF4` target specified
   through JSON.

What it is good for:

- checking order-of-magnitude event rates
- comparing prompt and delayed source components
- separating `12C` and `19F` contributions inside `CF4`
- estimating how relevant the approximate `19F` axial term is
- seeing how pressure, fiducial fraction, threshold, and geometry affect yearly
  yields

What it does not yet do:

- detector response, smearing, or efficiency modeling beyond a hard threshold
- beam timing cuts or pulse-window optimization
- background modeling
- uncertainty propagation
- automated parameter scans
- proposal-grade nuclear-structure input for `19F`

## Intended Role For An ERC Starting Proposal

This repository is best used as an early-stage support tool for a proposal
narrative. It lets you show that the source-target-detector concept has a
credible physics basis, that the relevant recoil scales are understood, and that
the expected rates are not obviously too small before investing in a full
sensitivity study.

In that sense, CHOAXIAL is not the final analysis stack. It is the layer that
helps decide whether the next steps are worth funding effort:

- replace the approximate nuclear inputs with proposal-grade structure functions
- add detector response and backgrounds
- run systematic scans over pressure, threshold, geometry, and baseline
- translate rates into discovery reach, exclusion contours, or design trade-offs

## Workflow In One View

```text
source model -> flux phi(E_nu)
             -> fold with d sigma / dE
             -> rate per CF4 molecule
             -> scale by detector gas inventory
             -> yearly event spectrum above threshold
```

The implementation follows exactly this chain:

- `cevens.py` computes `d sigma / dE_r` and `d sigma / dT_e`
- `ESS_flux.py` or `JPARK_flux.py` provides the DAR flux model
- `rate_estimation.py` computes `dR/dE` per `CF4` molecule
- `detector_estimation.py` multiplies by the number of fiducial molecules and by
  live time

## Physics Model

### 1. CEvNS kernel

The CEvNS differential cross section in `cevens.py` is implemented as a vector
piece plus a pure axial piece:

$$\frac{d\sigma}{dE_r} = \frac{G_F^2 m_N}{4\pi} \left[ \left(1 - \frac{m_N E_r}{2E_\nu^2} - \frac{E_r}{E_\nu}\right) Q_W^2 |F_W(q^2)|^2 + \left(1 + \frac{m_N E_r}{2E_\nu^2} - \frac{E_r}{E_\nu}\right) F_A(q^2) \right]$$

with

$$q^2 = 2 m_N E_r$$
$$Q_W = Z(1 - 4\sin^2\theta_W) - N$$

Key implementation choices:

- `12C` is treated as spin zero, so the axial term is set to zero.
- `19F` is the only nucleus with a built-in axial contribution.
- the vector form factor can be either Helm or point-*like*
- the code keeps total, vector-only, and axial-only differential cross sections
  separate

The generic axial structure is written as

$$F_A(q^2) = \frac{8\pi}{2J+1} \left[ (g_A^s)^2 S_{00}(q^2) - g_A g_A^s S_{01}(q^2) + g_A^2 S_{11}(q^2) \right]$$

and the repository already exposes a lightweight interface for plugging in
tabulated `S_00`, `S_01`, and `S_11` if better nuclear-structure inputs become
available.

For fast feasibility scans the default `19F` treatment is a simplified
spin-expectation model:

$$F_A(0) \sim g_A^2 \cdot \frac{32\pi}{3} \cdot \frac{J+1}{J(2J+1)} \cdot (S_p - S_n)^2$$
$$F_A(q^2) = F_A(0) \cdot \left(1 + \frac{q^2}{\Lambda_A^2}\right)^{-2 \cdot \text{power}}$$

with the current defaults:

- `J = 1/2`
- `S_p = 0.475`
- `S_n = -0.009`
- `Lambda_A = 0.35 GeV`

This is useful for early proposal work, but it should not be treated as the
final nuclear-structure input for a mature submission.

### 2. Neutrino-electron scattering

`cevens.py` also contains the Standard-Model free-electron differential cross
section used for electron recoils:

$$\frac{d\sigma}{dT_e} = \frac{2G_F^2 m_e}{\pi} \left[ g_{\text{lead}}^2 + g_{\text{sub}}^2(1-y)^2 - \frac{g_L g_R m_e T_e}{E_\nu^2} \right]$$

where `y = T_e / E_nu`, and the chiral couplings are assigned by flavor. For
`nu_e e` scattering the code includes the expected charged-current enhancement in
addition to the neutral-current term. For `CF4`, the molecule-level rate uses
all `42` target electrons.

### 3. DAR source model

Both source modules assume the standard stopped-pion chain:

$$\pi^+ \to \mu^+ + \nu_\mu \quad\text{(prompt, monochromatic)}$$
$$\mu^+ \to e^+ + \nu_e + \bar{\nu}_\mu \quad\text{(delayed, Michel spectra)}$$

The prompt neutrino energy is

$$E_{\nu\mu} = \frac{m_\pi^2 - m_\mu^2}{2m_\pi} \sim 29.79\text{ MeV}$$

The delayed components extend up to

$$E_{\max} = \frac{m_\mu}{2} \sim 52.83\text{ MeV}$$

The normalized Michel spectra used in the code are

$$f_{\nu_e}(E) = \frac{192}{m_\mu} \left(\frac{E}{m_\mu}\right)^2 \left(\frac{1}{2} - \frac{E}{m_\mu}\right)$$
$$f_{\bar{\nu}_\mu}(E) = \frac{64}{m_\mu} \left(\frac{E}{m_\mu}\right)^2 \left(\frac{3}{4} - \frac{E}{m_\mu}\right)$$

Both `ESS_flux.py` and `JPARK_flux.py` use a simple isotropic geometry factor:

$$\phi \propto \frac{1}{4\pi L^2}$$

The ESS benchmark is a design-style flux model. The J-PARC script is a separate
MLF benchmark that exports both average fluxes and per-POT fluences.

### 4. Flux folding and detector scaling

`rate_estimation.py` computes the differential event-rate spectrum as

$$\frac{dR}{dE_r} = \int dE_\nu \, \phi(E_\nu) \cdot \frac{d\sigma}{dE_r}$$

with an explicit prompt contribution for the monochromatic `nu_mu` line and
separate delayed contributions from `nu_e` and `anti-nu_mu`.

For `CF4`, the code keeps the molecule composition explicit:

$$\text{CF}_4 = 1 \times {}^{12}\text{C} + 4 \times {}^{19}\text{F}$$

The detector layer then applies ideal-gas bookkeeping:

$$V = \pi R^2 L$$
$$n_{\text{moles}} = \frac{PV}{R_{\text{gas}} T}$$
$$N_{\text{molecules}} = n_{\text{moles}} \cdot N_A \cdot f_{\text{fiducial}}$$

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
  Builds `12C`, `19F`, and `CF4` electron targets, folds the ESS-like flux with
  the interaction kernels, and writes per-molecule recoil spectra and summary
  plots to `cevens_rate_output/`.
- `detector_estimation.py`
  Reads the rate CSVs, parses the detector JSON, computes fiducial `CF4`
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

- CEvNS differential rates per `CF4` molecule
- neutrino-electron differential rates per electron and per `CF4` molecule
- separate prompt, delayed, and total components
- decomposition into `12C` and `19F`
- vector and axial separation for `19F`

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
  CEvNS spectra per molecule, including the `12C`/`19F` decomposition
- `cevens_rate_output/cf4_electron_differential_rate_per_molecule.csv`
  neutrino-electron recoil spectra per electron and per molecule
- `detector_rate_output/cf4_detector_summary.json`
  compact detector-level summary for proposal notes and quick comparisons

## Units And Conventions

The code uses practical phenomenology units throughout:

- neutrino energy: `MeV`
- nuclear recoil energy: `keV`
- electron recoil energy: `keV`
- nuclear masses: `GeV` internally
- CEvNS differential cross section: `cm^2 / keV`
- delayed flux: `neutrinos / (cm^2 s MeV)`
- prompt line flux: `neutrinos / (cm^2 s)`
- molecule-normalized rates: `s^-1 keV^-1 molecule^-1`
- detector-level spectra: events per `keV` per `year`

## Current Assumptions And Limitations

- The physics scope is Standard Model only.
- `19F` uses an approximate axial model unless the user replaces it with tabulated
  structure functions.
- The main rate chain is presently tied to the ESS-like source benchmark.
- The detector model assumes ideal-gas `CF4`.
- Thresholding is a hard cut with no efficiency curve.
- No backgrounds, acceptance effects, live-time losses, timing cuts, or energy
  smearing are included.
- No automated tests or package metadata are included yet.

These limitations are acceptable for a first-pass feasibility study, but they
become the main items to upgrade before turning this into a proposal-grade
forecast tool.

## Suggested Next Steps

If the preliminary rates look encouraging, the natural follow-up work is:

1. Replace the approximate `19F` axial term with shell-model or other validated
   tabulated `S_ij(q^2)` inputs.
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
*