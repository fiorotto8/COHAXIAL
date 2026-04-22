import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# ============================================================
# J-PARC MLF pion-DAR neutrino flux model
#
# Benchmark point:
#   proton beam power   P = 1 MW
#   proton kinetic E    Ep = 3 GeV
#   repetition rate     f = 25 Hz
#   bunch structure     2 bunches / spill
#   yield benchmark     y = 0.48 neutrinos / proton / flavor
#
# Flux model:
#   pi+ -> mu+ + nu_mu         (prompt, monochromatic)
#   mu+ -> e+ + nu_e + anti-nu_mu  (delayed, Michel spectra)
#
# Units:
#   Energies in MeV unless stated otherwise
#   Average fluxes in neutrinos / (cm^2 s MeV)
#   Per-POT fluences in neutrinos / (cm^2 POT MeV)
# ============================================================

# ---------- Physical constants ----------
ELEM_CHARGE_J = 1.602176634e-19  # J/eV
M_PI_MEV = 139.57039
M_MU_MEV = 105.6583755

# prompt nu_mu energy from pion decay at rest
E_NU_MU_PROMPT_MEV = (M_PI_MEV**2 - M_MU_MEV**2) / (2.0 * M_PI_MEV)

# endpoint of Michel spectra from mu+ decay at rest
E_NU_MAX_MEV = M_MU_MEV / 2.0

OUTPUT_DIR = "jparc_mlf_flux_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class JPARCMLFBeamConfig:
    """
    J-PARC MLF beam configuration for DAR neutrino flux estimates.
    """
    def __init__(
        self,
        beam_power_MW=1.0,
        proton_energy_GeV=3.0,
        repetition_rate_Hz=25.0,
        bunches_per_spill=2,
        bunch_width_s=100e-9,
        neutrino_yield_per_proton_per_flavor=0.48,
    ):
        self.beam_power_MW = beam_power_MW
        self.proton_energy_GeV = proton_energy_GeV
        self.repetition_rate_Hz = repetition_rate_Hz
        self.bunches_per_spill = bunches_per_spill
        self.bunch_width_s = bunch_width_s
        self.neutrino_yield_per_proton_per_flavor = neutrino_yield_per_proton_per_flavor

    @property
    def beam_power_W(self):
        return self.beam_power_MW * 1e6

    @property
    def proton_energy_J(self):
        return self.proton_energy_GeV * 1e9 * ELEM_CHARGE_J

    @property
    def protons_per_second(self):
        return self.beam_power_W / self.proton_energy_J

    @property
    def protons_per_pulse(self):
        return self.protons_per_second / self.repetition_rate_Hz

    @property
    def duty_factor(self):
        return self.repetition_rate_Hz * self.bunches_per_spill * self.bunch_width_s

    @property
    def neutrinos_per_second_per_flavor(self):
        return self.protons_per_second * self.neutrino_yield_per_proton_per_flavor

    @property
    def neutrinos_per_year_per_flavor(self):
        seconds_per_year = 365.25 * 24 * 3600
        return self.neutrinos_per_second_per_flavor * seconds_per_year


def michel_spectrum_nue(E_MeV):
    """
    Normalized differential spectrum f_nue(E) for mu+ DAR:
    integral_0^{m_mu/2} f(E) dE = 1
    Units: 1 / MeV
    Formula:
        f_nue(E) = (192/m_mu) * (E/m_mu)^2 * (1/2 - E/m_mu)
    """
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (192.0 / M_MU_MEV) * x**2 * (0.5 - x)
    return out


def michel_spectrum_numubar(E_MeV):
    """
    Normalized differential spectrum f_numubar(E) for mu+ DAR:
    integral_0^{m_mu/2} f(E) dE = 1
    Units: 1 / MeV
    Formula:
        f_numubar(E) = (64/m_mu) * (E/m_mu)^2 * (3/4 - E/m_mu)
    """
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (64.0 / M_MU_MEV) * x**2 * (0.75 - x)
    return out


def isotropic_geometry_factor_cm2(distance_m):
    """
    1 / (4 pi r^2), with r in cm.
    Result units: 1 / cm^2
    """
    r_cm = distance_m * 100.0
    return 1.0 / (4.0 * np.pi * r_cm**2)


# =========================
# Per-POT fluence functions
# =========================

def prompt_numu_line_fluence_per_pot(distance_m, beam=None):
    """
    Integrated prompt nu_mu line fluence per POT at distance_m.
    Units: neutrinos / (cm^2 POT)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    geom = isotropic_geometry_factor_cm2(distance_m)
    return beam.neutrino_yield_per_proton_per_flavor * geom


def differential_fluence_delayed_per_pot(E_MeV, distance_m, flavor, beam=None):
    """
    Delayed differential fluence per POT at distance_m.
    flavor = 'nue' or 'numubar'
    Units: neutrinos / (cm^2 POT MeV)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    geom = isotropic_geometry_factor_cm2(distance_m)
    source_yield = beam.neutrino_yield_per_proton_per_flavor

    if flavor == "nue":
        shape = michel_spectrum_nue(E_MeV)
    elif flavor == "numubar":
        shape = michel_spectrum_numubar(E_MeV)
    else:
        raise ValueError("flavor must be 'nue' or 'numubar'")

    return source_yield * geom * shape


def binned_prompt_numu_fluence_per_pot(E_edges_MeV, distance_m, beam=None):
    """
    Put the monochromatic prompt nu_mu line into an energy histogram.
    Returns bin-averaged differential fluence in each bin:
      neutrinos / (cm^2 POT MeV)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("E_edges_MeV must be strictly increasing")

    vals = np.zeros(len(edges) - 1, dtype=float)
    line_fluence = prompt_numu_line_fluence_per_pot(distance_m, beam=beam)
    E0 = E_NU_MU_PROMPT_MEV

    idx = np.searchsorted(edges, E0, side="right") - 1
    if 0 <= idx < len(vals):
        width = edges[idx + 1] - edges[idx]
        vals[idx] = line_fluence / width

    return vals


# =========================
# Average flux functions
# =========================

def differential_flux_delayed(E_MeV, distance_m, flavor, beam=None):
    """
    Differential average flux at distance_m for delayed components:
      flavor = 'nue' or 'numubar'
    Returns neutrinos / (cm^2 s MeV)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    geom = isotropic_geometry_factor_cm2(distance_m)
    source_rate = beam.neutrinos_per_second_per_flavor

    if flavor == "nue":
        shape = michel_spectrum_nue(E_MeV)
    elif flavor == "numubar":
        shape = michel_spectrum_numubar(E_MeV)
    else:
        raise ValueError("flavor must be 'nue' or 'numubar'")

    return source_rate * geom * shape


def prompt_numu_line_flux(distance_m, beam=None):
    """
    Integrated prompt nu_mu line average flux at distance_m.
    Units: neutrinos / (cm^2 s)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    geom = isotropic_geometry_factor_cm2(distance_m)
    source_rate = beam.neutrinos_per_second_per_flavor
    return source_rate * geom


def binned_prompt_numu_flux(E_edges_MeV, distance_m, beam=None):
    """
    Put the monochromatic prompt nu_mu line into an energy histogram.
    Returns bin-averaged differential flux in each bin:
      neutrinos / (cm^2 s MeV)
    """
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("E_edges_MeV must be strictly increasing")

    vals = np.zeros(len(edges) - 1, dtype=float)
    line_flux = prompt_numu_line_flux(distance_m, beam=beam)
    E0 = E_NU_MU_PROMPT_MEV

    idx = np.searchsorted(edges, E0, side="right") - 1
    if 0 <= idx < len(vals):
        width = edges[idx + 1] - edges[idx]
        vals[idx] = line_flux / width

    return vals


# =========================
# CSV helpers
# =========================

def save_point_flux_csv(filename, E_MeV, phi_nue, phi_numubar, phi_delayed_sum):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_MeV",
            "phi_nue_per_cm2_s_MeV",
            "phi_numubar_per_cm2_s_MeV",
            "phi_delayed_sum_per_cm2_s_MeV",
        ])
        for row in zip(E_MeV, phi_nue, phi_numubar, phi_delayed_sum):
            writer.writerow(row)


def save_binned_flux_csv(filename, E_edges_MeV, phi_numu_prompt_binned, beam=None, distance_m=24.0):
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    phi_nue = differential_flux_delayed(centers, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(centers, distance_m, "numubar", beam=beam)
    phi_total = phi_numu_prompt_binned + phi_nue + phi_numubar

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_low_MeV",
            "E_high_MeV",
            "E_center_MeV",
            "phi_numu_prompt_per_cm2_s_MeV",
            "phi_nue_per_cm2_s_MeV",
            "phi_numubar_per_cm2_s_MeV",
            "phi_total_per_cm2_s_MeV",
        ])
        for row in zip(edges[:-1], edges[1:], centers, phi_numu_prompt_binned, phi_nue, phi_numubar, phi_total):
            writer.writerow(row)


def save_point_fluence_per_pot_csv(filename, E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_MeV",
            "Phi_nue_per_cm2_POT_MeV",
            "Phi_numubar_per_cm2_POT_MeV",
            "Phi_delayed_sum_per_cm2_POT_MeV",
        ])
        for row in zip(E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum):
            writer.writerow(row)


def save_binned_fluence_per_pot_csv(filename, E_edges_MeV, Phi_numu_prompt_binned, beam=None, distance_m=24.0):
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Phi_nue = differential_fluence_delayed_per_pot(centers, distance_m, "nue", beam=beam)
    Phi_numubar = differential_fluence_delayed_per_pot(centers, distance_m, "numubar", beam=beam)
    Phi_total = Phi_numu_prompt_binned + Phi_nue + Phi_numubar

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "E_low_MeV",
            "E_high_MeV",
            "E_center_MeV",
            "Phi_numu_prompt_per_cm2_POT_MeV",
            "Phi_nue_per_cm2_POT_MeV",
            "Phi_numubar_per_cm2_POT_MeV",
            "Phi_total_per_cm2_POT_MeV",
        ])
        for row in zip(edges[:-1], edges[1:], centers, Phi_numu_prompt_binned, Phi_nue, Phi_numubar, Phi_total):
            writer.writerow(row)


# =========================
# Plot helpers
# =========================

def plot_point_fluxes(filename, E_MeV, phi_nue, phi_numubar, phi_delayed_sum):
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, phi_delayed_sum, label="delayed sum", linestyle="--")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF delayed neutrino differential flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_flux(filename, E_edges_MeV, phi_numu_prompt_binned, beam=None, distance_m=24.0):
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    phi_nue = differential_flux_delayed(centers, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(centers, distance_m, "numubar", beam=beam)
    phi_total = phi_numu_prompt_binned + phi_nue + phi_numubar

    plt.figure(figsize=(8, 5))
    plt.step(centers, phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, phi_nue, label=r"$\nu_e$")
    plt.plot(centers, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, phi_total, label="total", linewidth=2)
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF total neutrino differential flux (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    positive = np.concatenate([phi_numu_prompt_binned, phi_nue, phi_numubar, phi_total])
    positive = positive[positive > 0]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_point_fluence_per_pot(filename, E_MeV, Phi_nue, Phi_numubar, Phi_delayed_sum):
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, Phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, Phi_delayed_sum, label="delayed sum", linestyle="--")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF delayed neutrino differential fluence per POT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_fluence_per_pot(filename, E_edges_MeV, Phi_numu_prompt_binned, beam=None, distance_m=24.0):
    if beam is None:
        beam = JPARCMLFBeamConfig()

    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Phi_nue = differential_fluence_delayed_per_pot(centers, distance_m, "nue", beam=beam)
    Phi_numubar = differential_fluence_delayed_per_pot(centers, distance_m, "numubar", beam=beam)
    Phi_total = Phi_numu_prompt_binned + Phi_nue + Phi_numubar

    plt.figure(figsize=(8, 5))
    plt.step(centers, Phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, Phi_nue, label=r"$\nu_e$")
    plt.plot(centers, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, Phi_total, label="total", linewidth=2)
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF total neutrino differential fluence per POT (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    positive = np.concatenate([Phi_numu_prompt_binned, Phi_nue, Phi_numubar, Phi_total])
    positive = positive[positive > 0]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    beam = JPARCMLFBeamConfig()

    print("=== J-PARC MLF beam configuration ===")
    print(f"Beam power               : {beam.beam_power_MW:.3f} MW")
    print(f"Proton kinetic energy    : {beam.proton_energy_GeV:.3f} GeV")
    print(f"Repetition rate          : {beam.repetition_rate_Hz:.3f} Hz")
    print(f"Bunches per spill        : {beam.bunches_per_spill:d}")
    print(f"Bunch width             : {beam.bunch_width_s*1e9:.1f} ns")
    print(f"Duty factor              : {beam.duty_factor:.6e}")
    print(f"Protons/s                : {beam.protons_per_second:.6e}")
    print(f"Protons/pulse            : {beam.protons_per_pulse:.6e}")
    print(f"Yield per proton/flavor  : {beam.neutrino_yield_per_proton_per_flavor:.3f}")
    print(f"Nu/s per flavor          : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Nu/year per flavor       : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"Prompt nu_mu energy      : {E_NU_MU_PROMPT_MEV:.6f} MeV")
    print(f"Delayed endpoint         : {E_NU_MAX_MEV:.6f} MeV")

    distance_m = 24.0

    line_flux = prompt_numu_line_flux(distance_m, beam=beam)
    line_fluence_per_pot = prompt_numu_line_fluence_per_pot(distance_m, beam=beam)

    print(f"\nPrompt nu_mu line flux at {distance_m:.1f} m: {line_flux:.6e} /cm^2/s")
    print(f"Prompt nu_mu line fluence per POT at {distance_m:.1f} m: {line_fluence_per_pot:.6e} /cm^2/POT")

    # ----- fine grid for smooth delayed spectra -----
    E = np.linspace(0.0, E_NU_MAX_MEV, 500)

    # Average fluxes
    phi_nue = differential_flux_delayed(E, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(E, distance_m, "numubar", beam=beam)
    phi_delayed_sum = phi_nue + phi_numubar

    print(f"Peak nue average differential flux      : {phi_nue.max():.6e} /cm^2/s/MeV")
    print(f"Peak numubar average differential flux  : {phi_numubar.max():.6e} /cm^2/s/MeV")

    # Per-POT fluences
    Phi_nue = differential_fluence_delayed_per_pot(E, distance_m, "nue", beam=beam)
    Phi_numubar = differential_fluence_delayed_per_pot(E, distance_m, "numubar", beam=beam)
    Phi_delayed_sum = Phi_nue + Phi_numubar

    print(f"Peak nue differential fluence per POT      : {Phi_nue.max():.6e} /cm^2/POT/MeV")
    print(f"Peak numubar differential fluence per POT  : {Phi_numubar.max():.6e} /cm^2/POT/MeV")

    # ----- histogram grid for total spectrum including prompt line -----
    E_edges = np.linspace(0.0, E_NU_MAX_MEV, 265)  # ~0.2 MeV bins

    phi_numu_prompt_binned = binned_prompt_numu_flux(E_edges, distance_m, beam=beam)
    Phi_numu_prompt_binned = binned_prompt_numu_fluence_per_pot(E_edges, distance_m, beam=beam)

    # ----- save average flux outputs -----
    save_point_flux_csv(
        f"{OUTPUT_DIR}/jparc_delayed_flux_point_grid.csv",
        E,
        phi_nue,
        phi_numubar,
        phi_delayed_sum,
    )

    save_binned_flux_csv(
        f"{OUTPUT_DIR}/jparc_total_flux_binned.csv",
        E_edges,
        phi_numu_prompt_binned,
        beam=beam,
        distance_m=distance_m,
    )

    plot_point_fluxes(
        f"{OUTPUT_DIR}/jparc_delayed_flux_point_grid.png",
        E,
        phi_nue,
        phi_numubar,
        phi_delayed_sum,
    )

    plot_binned_total_flux(
        f"{OUTPUT_DIR}/jparc_total_flux_binned.png",
        E_edges,
        phi_numu_prompt_binned,
        beam=beam,
        distance_m=distance_m,
    )

    # ----- save per-POT fluence outputs -----
    save_point_fluence_per_pot_csv(
        f"{OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.csv",
        E,
        Phi_nue,
        Phi_numubar,
        Phi_delayed_sum,
    )

    save_binned_fluence_per_pot_csv(
        f"{OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.csv",
        E_edges,
        Phi_numu_prompt_binned,
        beam=beam,
        distance_m=distance_m,
    )

    plot_point_fluence_per_pot(
        f"{OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.png",
        E,
        Phi_nue,
        Phi_numubar,
        Phi_delayed_sum,
    )

    plot_binned_total_fluence_per_pot(
        f"{OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.png",
        E_edges,
        Phi_numu_prompt_binned,
        beam=beam,
        distance_m=distance_m,
    )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR}/jparc_delayed_flux_point_grid.csv")
    print(f"  {OUTPUT_DIR}/jparc_total_flux_binned.csv")
    print(f"  {OUTPUT_DIR}/jparc_delayed_flux_point_grid.png")
    print(f"  {OUTPUT_DIR}/jparc_total_flux_binned.png")
    print(f"  {OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.csv")
    print(f"  {OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.csv")
    print(f"  {OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.png")
    print(f"  {OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.png")