from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# J-PARC MLF pion-DAR benchmark flux model
#
# This module is a lightweight semi-analytic stopped-pion DAR source model.
# It exposes:
#   - average delayed differential fluxes          [cm^-2 s^-1 MeV^-1]
#   - average prompt line flux                     [cm^-2 s^-1]
#   - binned total average flux                   [cm^-2 s^-1 MeV^-1]
#   - per-POT delayed differential fluences       [cm^-2 POT^-1 MeV^-1]
#   - per-POT prompt line fluence                 [cm^-2 POT^-1]
#   - binned total per-POT fluence                [cm^-2 POT^-1 MeV^-1]
#
# Physics scope:
#   pi+ -> mu+ + nu_mu         (prompt, monochromatic)
#   mu+ -> e+ + nu_e + anti-nu_mu  (delayed Michel spectra)
#
# Normalization:
#   source yield per proton per flavor is treated as an effective benchmark
#   input, not a first-principles hadron-production calculation.
#
# Geometry:
#   point source with 1 / (4 pi L^2) dilution only.
#
# Timing:
#   two bunches per spill are exposed as metadata. They are not yet folded into
#   the flux shape or a detailed time-distribution model.
# ============================================================

ELEM_CHARGE_J = 1.602176634e-19  # J / eV
M_PI_MEV = 139.57039
M_MU_MEV = 105.6583755

E_NU_MU_PROMPT_MEV = (M_PI_MEV**2 - M_MU_MEV**2) / (2.0 * M_PI_MEV)
E_NU_MAX_MEV = M_MU_MEV / 2.0

OUTPUT_DIR = "jparc_mlf_flux_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class JPARCMLFBeamConfig:
    """J-PARC MLF benchmark beam metadata for semi-analytic DAR source estimates.

    Parameters
    ----------
    beam_power_MW
        Average proton-beam power in MW.
    proton_energy_GeV
        Proton kinetic energy in GeV.
    repetition_rate_Hz
        Spill repetition rate in Hz.
    bunches_per_spill
        Number of narrow bunches per spill. Kept as metadata for timing studies.
    bunch_width_s
        Approximate RMS-scale bunch width metadata. Not used in the flux shape.
    bunch_spacing_s
        Approximate spacing between the two bunches in a spill. Metadata only.
    neutrino_yield_per_proton_per_flavor
        Effective source yield per incoming proton and per neutrino flavor.
        The default 0.48 is a practical benchmark normalization for J-PARC MLF.
    yield_fractional_uncertainty
        Optional fractional normalization uncertainty on the source yield.
        It is exposed in metadata and summaries, but not propagated further.
    """

    def __init__(
        self,
        beam_power_MW: float = 1.0,
        proton_energy_GeV: float = 3.0,
        repetition_rate_Hz: float = 25.0,
        bunches_per_spill: int = 2,
        bunch_width_s: float = 100e-9,
        bunch_spacing_s: float = 600e-9,
        neutrino_yield_per_proton_per_flavor: float = 0.48,
        yield_fractional_uncertainty: float = 0.0,
    ) -> None:
        self.beam_power_MW = beam_power_MW
        self.proton_energy_GeV = proton_energy_GeV
        self.repetition_rate_Hz = repetition_rate_Hz
        self.bunches_per_spill = bunches_per_spill
        self.bunch_width_s = bunch_width_s
        self.bunch_spacing_s = bunch_spacing_s
        self.neutrino_yield_per_proton_per_flavor = neutrino_yield_per_proton_per_flavor
        self.yield_fractional_uncertainty = yield_fractional_uncertainty

    @property
    def beam_power_W(self) -> float:
        return self.beam_power_MW * 1e6

    @property
    def proton_energy_J(self) -> float:
        return self.proton_energy_GeV * 1e9 * ELEM_CHARGE_J

    @property
    def protons_per_second(self) -> float:
        return self.beam_power_W / self.proton_energy_J

    @property
    def protons_per_pulse(self) -> float:
        return self.protons_per_second / self.repetition_rate_Hz

    @property
    def protons_per_bunch(self) -> float:
        return self.protons_per_pulse / self.bunches_per_spill

    @property
    def duty_factor(self) -> float:
        return self.repetition_rate_Hz * self.bunches_per_spill * self.bunch_width_s

    @property
    def spill_timing_window_s(self) -> float:
        if self.bunches_per_spill <= 1:
            return self.bunch_width_s
        return self.bunches_per_spill * self.bunch_width_s + (self.bunches_per_spill - 1) * self.bunch_spacing_s

    @property
    def neutrinos_per_second_per_flavor(self) -> float:
        return self.protons_per_second * self.neutrino_yield_per_proton_per_flavor

    @property
    def neutrinos_per_pulse_per_flavor(self) -> float:
        return self.protons_per_pulse * self.neutrino_yield_per_proton_per_flavor

    @property
    def neutrinos_per_year_per_flavor(self) -> float:
        seconds_per_year = 365.25 * 24.0 * 3600.0
        return self.neutrinos_per_second_per_flavor * seconds_per_year


def michel_spectrum_nue(E_MeV: np.ndarray | float) -> np.ndarray:
    """Normalized delayed nu_e spectrum from mu+ DAR in units of MeV^-1."""
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (192.0 / M_MU_MEV) * x**2 * (0.5 - x)
    return out


def michel_spectrum_numubar(E_MeV: np.ndarray | float) -> np.ndarray:
    """Normalized delayed anti-nu_mu spectrum from mu+ DAR in units of MeV^-1."""
    E = np.asarray(E_MeV, dtype=float)
    out = np.zeros_like(E)
    mask = (E >= 0.0) & (E <= E_NU_MAX_MEV)
    x = E[mask] / M_MU_MEV
    out[mask] = (64.0 / M_MU_MEV) * x**2 * (0.75 - x)
    return out


def isotropic_geometry_factor_cm2(distance_m: float) -> float:
    """Return the point-source dilution factor 1 / (4 pi L^2) in cm^-2."""
    r_cm = distance_m * 100.0
    return 1.0 / (4.0 * np.pi * r_cm**2)


def _delayed_shape(E_MeV: np.ndarray | float, flavor: str) -> np.ndarray:
    if flavor == "nue":
        return michel_spectrum_nue(E_MeV)
    if flavor == "numubar":
        return michel_spectrum_numubar(E_MeV)
    raise ValueError("flavor must be 'nue' or 'numubar'")


def _binned_line_density(E_edges_MeV: np.ndarray, line_energy_MeV: float, line_intensity: float) -> np.ndarray:
    """Represent a monochromatic line as a histogram-bin average density."""
    edges = np.asarray(E_edges_MeV, dtype=float)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("E_edges_MeV must be strictly increasing")

    vals = np.zeros(len(edges) - 1, dtype=float)
    idx = np.searchsorted(edges, line_energy_MeV, side="right") - 1
    if 0 <= idx < len(vals):
        width = edges[idx + 1] - edges[idx]
        vals[idx] = line_intensity / width
    return vals


def prompt_numu_line_flux(distance_m: float, beam: JPARCMLFBeamConfig | None = None) -> float:
    """Integrated prompt nu_mu average line flux in neutrinos / (cm^2 s)."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return beam.neutrinos_per_second_per_flavor * isotropic_geometry_factor_cm2(distance_m)


def prompt_numu_line_fluence_per_pot(distance_m: float, beam: JPARCMLFBeamConfig | None = None) -> float:
    """Integrated prompt nu_mu fluence per POT in neutrinos / (cm^2 POT)."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return beam.neutrino_yield_per_proton_per_flavor * isotropic_geometry_factor_cm2(distance_m)


def differential_flux_delayed(
    E_MeV: np.ndarray | float,
    distance_m: float,
    flavor: str,
    beam: JPARCMLFBeamConfig | None = None,
) -> np.ndarray:
    """Delayed differential average flux in neutrinos / (cm^2 s MeV)."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return (
        beam.neutrinos_per_second_per_flavor
        * isotropic_geometry_factor_cm2(distance_m)
        * _delayed_shape(E_MeV, flavor)
    )


def differential_fluence_delayed_per_pot(
    E_MeV: np.ndarray | float,
    distance_m: float,
    flavor: str,
    beam: JPARCMLFBeamConfig | None = None,
) -> np.ndarray:
    """Delayed differential fluence per POT in neutrinos / (cm^2 POT MeV)."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return (
        beam.neutrino_yield_per_proton_per_flavor
        * isotropic_geometry_factor_cm2(distance_m)
        * _delayed_shape(E_MeV, flavor)
    )


def binned_prompt_numu_flux(
    E_edges_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> np.ndarray:
    """Histogrammed prompt nu_mu line as average flux density in cm^-2 s^-1 MeV^-1."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return _binned_line_density(E_edges_MeV, E_NU_MU_PROMPT_MEV, prompt_numu_line_flux(distance_m, beam=beam))


def binned_prompt_numu_fluence_per_pot(
    E_edges_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> np.ndarray:
    """Histogrammed prompt nu_mu line as fluence density in cm^-2 POT^-1 MeV^-1."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    return _binned_line_density(
        E_edges_MeV,
        E_NU_MU_PROMPT_MEV,
        prompt_numu_line_fluence_per_pot(distance_m, beam=beam),
    )


def total_differential_flux(
    E_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return delayed average differential flux components on a point grid."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    E = np.asarray(E_MeV, dtype=float)
    phi_nue = differential_flux_delayed(E, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(E, distance_m, "numubar", beam=beam)
    return {
        "E_MeV": E,
        "phi_nue": phi_nue,
        "phi_numubar": phi_numubar,
        "phi_delayed_sum": phi_nue + phi_numubar,
    }


def total_differential_fluence_per_pot(
    E_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return delayed differential fluence-per-POT components on a point grid."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    E = np.asarray(E_MeV, dtype=float)
    Phi_nue = differential_fluence_delayed_per_pot(E, distance_m, "nue", beam=beam)
    Phi_numubar = differential_fluence_delayed_per_pot(E, distance_m, "numubar", beam=beam)
    return {
        "E_MeV": E,
        "Phi_nue": Phi_nue,
        "Phi_numubar": Phi_numubar,
        "Phi_delayed_sum": Phi_nue + Phi_numubar,
    }


def binned_total_flux(
    E_edges_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return the binned total average flux, including the prompt line bin."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    phi_numu_prompt_binned = binned_prompt_numu_flux(edges, distance_m, beam=beam)
    phi_nue = differential_flux_delayed(centers, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(centers, distance_m, "numubar", beam=beam)
    return {
        "E_low_MeV": edges[:-1],
        "E_high_MeV": edges[1:],
        "E_center_MeV": centers,
        "phi_numu_prompt_binned": phi_numu_prompt_binned,
        "phi_nue": phi_nue,
        "phi_numubar": phi_numubar,
        "phi_total": phi_numu_prompt_binned + phi_nue + phi_numubar,
    }


def binned_total_fluence_per_pot(
    E_edges_MeV: np.ndarray,
    distance_m: float,
    beam: JPARCMLFBeamConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return the binned total fluence per POT, including the prompt line bin."""
    if beam is None:
        beam = JPARCMLFBeamConfig()
    edges = np.asarray(E_edges_MeV, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Phi_numu_prompt_binned = binned_prompt_numu_fluence_per_pot(edges, distance_m, beam=beam)
    Phi_nue = differential_fluence_delayed_per_pot(centers, distance_m, "nue", beam=beam)
    Phi_numubar = differential_fluence_delayed_per_pot(centers, distance_m, "numubar", beam=beam)
    return {
        "E_low_MeV": edges[:-1],
        "E_high_MeV": edges[1:],
        "E_center_MeV": centers,
        "Phi_numu_prompt_binned": Phi_numu_prompt_binned,
        "Phi_nue": Phi_nue,
        "Phi_numubar": Phi_numubar,
        "Phi_total": Phi_numu_prompt_binned + Phi_nue + Phi_numubar,
    }


def save_point_flux_csv(filename: str, E_MeV: np.ndarray, phi_nue: np.ndarray, phi_numubar: np.ndarray, phi_delayed_sum: np.ndarray) -> None:
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


def save_binned_flux_csv(filename: str, binned_flux: dict[str, np.ndarray]) -> None:
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
        for row in zip(
            binned_flux["E_low_MeV"],
            binned_flux["E_high_MeV"],
            binned_flux["E_center_MeV"],
            binned_flux["phi_numu_prompt_binned"],
            binned_flux["phi_nue"],
            binned_flux["phi_numubar"],
            binned_flux["phi_total"],
        ):
            writer.writerow(row)


def save_point_fluence_per_pot_csv(
    filename: str,
    E_MeV: np.ndarray,
    Phi_nue: np.ndarray,
    Phi_numubar: np.ndarray,
    Phi_delayed_sum: np.ndarray,
) -> None:
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


def save_binned_fluence_per_pot_csv(filename: str, binned_fluence: dict[str, np.ndarray]) -> None:
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
        for row in zip(
            binned_fluence["E_low_MeV"],
            binned_fluence["E_high_MeV"],
            binned_fluence["E_center_MeV"],
            binned_fluence["Phi_numu_prompt_binned"],
            binned_fluence["Phi_nue"],
            binned_fluence["Phi_numubar"],
            binned_fluence["Phi_total"],
        ):
            writer.writerow(row)


def plot_point_fluxes(filename: str, E_MeV: np.ndarray, phi_nue: np.ndarray, phi_numubar: np.ndarray, phi_delayed_sum: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, phi_delayed_sum, linestyle="--", label="delayed sum")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF delayed neutrino differential flux")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_flux(filename: str, binned_flux: dict[str, np.ndarray]) -> None:
    centers = binned_flux["E_center_MeV"]
    phi_numu_prompt_binned = binned_flux["phi_numu_prompt_binned"]
    phi_nue = binned_flux["phi_nue"]
    phi_numubar = binned_flux["phi_numubar"]
    phi_total = binned_flux["phi_total"]

    plt.figure(figsize=(8, 5))
    plt.step(centers, phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, phi_nue, label=r"$\nu_e$")
    plt.plot(centers, phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, phi_total, linewidth=2, label="total")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential flux [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF total neutrino differential flux (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    positive = np.concatenate([phi_numu_prompt_binned, phi_nue, phi_numubar, phi_total])
    positive = positive[positive > 0.0]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_point_fluence_per_pot(filename: str, E_MeV: np.ndarray, Phi_nue: np.ndarray, Phi_numubar: np.ndarray, Phi_delayed_sum: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(E_MeV, Phi_nue, label=r"$\nu_e$")
    plt.plot(E_MeV, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(E_MeV, Phi_delayed_sum, linestyle="--", label="delayed sum")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF delayed neutrino differential fluence per POT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_binned_total_fluence_per_pot(filename: str, binned_fluence: dict[str, np.ndarray]) -> None:
    centers = binned_fluence["E_center_MeV"]
    Phi_numu_prompt_binned = binned_fluence["Phi_numu_prompt_binned"]
    Phi_nue = binned_fluence["Phi_nue"]
    Phi_numubar = binned_fluence["Phi_numubar"]
    Phi_total = binned_fluence["Phi_total"]

    plt.figure(figsize=(8, 5))
    plt.step(centers, Phi_numu_prompt_binned, where="mid", label=r"prompt $\nu_\mu$")
    plt.plot(centers, Phi_nue, label=r"$\nu_e$")
    plt.plot(centers, Phi_numubar, label=r"$\bar{\nu}_\mu$")
    plt.plot(centers, Phi_total, linewidth=2, label="total")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel(r"Differential fluence [cm$^{-2}$ POT$^{-1}$ MeV$^{-1}$]")
    plt.title("J-PARC MLF total neutrino differential fluence per POT (binned)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    positive = np.concatenate([Phi_numu_prompt_binned, Phi_nue, Phi_numubar, Phi_total])
    positive = positive[positive > 0.0]
    if positive.size:
        plt.ylim(positive.min() * 0.8, positive.max() * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    beam = JPARCMLFBeamConfig()
    distance_m = 24.0

    print("=== J-PARC MLF DAR benchmark summary ===")
    print(f"Beam power                       : {beam.beam_power_MW:.3f} MW")
    print(f"Proton kinetic energy            : {beam.proton_energy_GeV:.3f} GeV")
    print(f"Repetition rate                  : {beam.repetition_rate_Hz:.3f} Hz")
    print(f"Bunches per spill                : {beam.bunches_per_spill:d}")
    print(f"Bunch width                      : {beam.bunch_width_s * 1e9:.1f} ns")
    print(f"Bunch spacing                    : {beam.bunch_spacing_s * 1e9:.1f} ns")
    print(f"Spill timing window              : {beam.spill_timing_window_s * 1e9:.1f} ns")
    print("Timing note                      : metadata only; not folded into flux shape")
    print(f"Protons/s                        : {beam.protons_per_second:.6e}")
    print(f"Protons/pulse                    : {beam.protons_per_pulse:.6e}")
    print(f"Protons/bunch                    : {beam.protons_per_bunch:.6e}")
    print(f"Yield per proton per flavor      : {beam.neutrino_yield_per_proton_per_flavor:.3f}")
    print(f"Yield fractional uncertainty     : {beam.yield_fractional_uncertainty:.3%}")
    print(f"Neutrinos/s per flavor           : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Neutrinos/pulse per flavor       : {beam.neutrinos_per_pulse_per_flavor:.6e}")
    print(f"Neutrinos/year per flavor        : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"Prompt nu_mu line energy         : {E_NU_MU_PROMPT_MEV:.6f} MeV")
    print(f"Delayed endpoint                 : {E_NU_MAX_MEV:.6f} MeV")
    print(f"Chosen baseline                  : {distance_m:.1f} m")
    print(f"Prompt nu_mu line flux           : {prompt_numu_line_flux(distance_m, beam=beam):.6e} /cm^2/s")
    print(f"Prompt nu_mu line fluence / POT  : {prompt_numu_line_fluence_per_pot(distance_m, beam=beam):.6e} /cm^2/POT")

    E = np.linspace(0.0, E_NU_MAX_MEV, 500)
    E_edges = np.linspace(0.0, E_NU_MAX_MEV, 265)

    flux_point = total_differential_flux(E, distance_m, beam=beam)
    fluence_point = total_differential_fluence_per_pot(E, distance_m, beam=beam)
    flux_binned = binned_total_flux(E_edges, distance_m, beam=beam)
    fluence_binned = binned_total_fluence_per_pot(E_edges, distance_m, beam=beam)

    print(f"Peak nue average differential flux      : {flux_point['phi_nue'].max():.6e} /cm^2/s/MeV")
    print(f"Peak numubar average differential flux  : {flux_point['phi_numubar'].max():.6e} /cm^2/s/MeV")
    print(f"Peak nue fluence per POT                : {fluence_point['Phi_nue'].max():.6e} /cm^2/POT/MeV")
    print(f"Peak numubar fluence per POT            : {fluence_point['Phi_numubar'].max():.6e} /cm^2/POT/MeV")

    save_point_flux_csv(
        f"{OUTPUT_DIR}/jparc_delayed_flux_point_grid.csv",
        flux_point["E_MeV"],
        flux_point["phi_nue"],
        flux_point["phi_numubar"],
        flux_point["phi_delayed_sum"],
    )
    save_binned_flux_csv(f"{OUTPUT_DIR}/jparc_total_flux_binned.csv", flux_binned)
    save_point_fluence_per_pot_csv(
        f"{OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.csv",
        fluence_point["E_MeV"],
        fluence_point["Phi_nue"],
        fluence_point["Phi_numubar"],
        fluence_point["Phi_delayed_sum"],
    )
    save_binned_fluence_per_pot_csv(
        f"{OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.csv",
        fluence_binned,
    )

    plot_point_fluxes(
        f"{OUTPUT_DIR}/jparc_delayed_flux_point_grid.png",
        flux_point["E_MeV"],
        flux_point["phi_nue"],
        flux_point["phi_numubar"],
        flux_point["phi_delayed_sum"],
    )
    plot_binned_total_flux(f"{OUTPUT_DIR}/jparc_total_flux_binned.png", flux_binned)
    plot_point_fluence_per_pot(
        f"{OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.png",
        fluence_point["E_MeV"],
        fluence_point["Phi_nue"],
        fluence_point["Phi_numubar"],
        fluence_point["Phi_delayed_sum"],
    )
    plot_binned_total_fluence_per_pot(
        f"{OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.png",
        fluence_binned,
    )

    print("\nSaved files:")
    print(f"  {OUTPUT_DIR}/jparc_delayed_flux_point_grid.csv")
    print(f"  {OUTPUT_DIR}/jparc_total_flux_binned.csv")
    print(f"  {OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.csv")
    print(f"  {OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.csv")
    print(f"  {OUTPUT_DIR}/jparc_delayed_flux_point_grid.png")
    print(f"  {OUTPUT_DIR}/jparc_total_flux_binned.png")
    print(f"  {OUTPUT_DIR}/jparc_delayed_fluence_per_pot_point_grid.png")
    print(f"  {OUTPUT_DIR}/jparc_total_fluence_per_pot_binned.png")
