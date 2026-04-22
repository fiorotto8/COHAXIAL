from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from cevens import CEvNSCalculator, carbon12_target, fluorine19_target
from ESS_flux import (
    ESSBeamConfig,
    E_NU_MU_PROMPT_MEV,
    E_NU_MAX_MEV,
    differential_flux_delayed,
    prompt_numu_line_flux,
)

# ============================================================
# CF4 CEvNS differential-rate scan at an ESS-like pion-DAR source
#
# What this script computes
# -------------------------
# For a CF4 molecule, it computes:
#
#   dR/dEr  [s^-1 keV^-1 molecule^-1]
#
# split into:
#   - prompt nu_mu contribution
#   - delayed nu_e contribution
#   - delayed anti-nu_mu contribution
#   - total delayed contribution
#   - total contribution
#
# It also stores separate C and F pieces, and vector / axial split
# for fluorine, so you can inspect how much of the molecule-level
# rate is driven by 19F and how large the axial fraction is.
#
# Normalization
# -------------
# By default everything is normalized to:
#
#   one CF4 molecule
#
# so the rates are "interaction rate per molecule".
#
# To obtain rates for a detector, multiply later by the number of
# molecules in the fiducial target and by the live time.
#
# Units
# -----
#   Er          : keV
#   E_nu        : MeV
#   flux        : cm^-2 s^-1 MeV^-1   (delayed)
#   prompt flux : cm^-2 s^-1          (monochromatic line)
#   dσ/dEr      : cm^2 keV^-1
#   dR/dEr      : s^-1 keV^-1 molecule^-1
# ============================================================

OUTPUT_DIR = "cevens_rate_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class RateConfig:
    distance_m: float = 20.0

    # recoil-energy grid
    er_min_kev: float = 0.0
    er_max_kev: float = 120.0
    n_er: int = 1201

    # neutrino-energy grid for delayed components
    enu_min_mev: float = 1E-6
    enu_max_mev: float = E_NU_MAX_MEV
    n_enu: int = 3000

    # optional threshold for integrated-above-threshold summaries
    threshold_kev: float = 0.0

    # built-in 19F axial approx settings
    fluorine_axial_model: str = "approx"
    fluorine_sp: float = 0.475
    fluorine_sn: float = -0.009
    fluorine_lambda_a_gev: float = 0.35


def integrate_over_enu(
    enu_grid_mev: np.ndarray,
    flux_per_cm2_s_mev: np.ndarray,
    dsig_dEr_cm2_per_kev: np.ndarray,
) -> float:
    """
    Compute:
        integral dE_nu phi(E_nu) * dσ/dEr(E_nu, Er)

    Returns:
        dR/dEr in s^-1 keV^-1
    """
    return float(np.trapz(flux_per_cm2_s_mev * dsig_dEr_cm2_per_kev, enu_grid_mev))


def build_dsigma_vs_enu(
    calc: CEvNSCalculator,
    target,
    enu_grid_mev: np.ndarray,
    er_kev: float,
    mode: str = "total",
) -> np.ndarray:
    """
    mode = "total", "vector", "axial"
    """
    out = np.zeros_like(enu_grid_mev, dtype=float)

    for i, enu in enumerate(enu_grid_mev):
        if mode == "total":
            out[i] = calc.differential_cross_section_cm2_per_kev(target, enu, er_kev)
        elif mode == "vector":
            out[i] = calc.differential_vector_cross_section_cm2_per_kev(target, enu, er_kev)
        elif mode == "axial":
            out[i] = calc.differential_axial_cross_section_cm2_per_kev(target, enu, er_kev)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return out


def compute_component_rates_per_target(
    calc: CEvNSCalculator,
    target,
    er_grid_kev: np.ndarray,
    enu_grid_mev: np.ndarray,
    beam: ESSBeamConfig,
    distance_m: float,
) -> Dict[str, np.ndarray]:
    """
    Per-target rates:
        dR/dEr [s^-1 keV^-1 target^-1]
    with prompt and delayed components split.
    """
    phi_prompt_line = prompt_numu_line_flux(distance_m, beam=beam)  # cm^-2 s^-1
    phi_nue = differential_flux_delayed(enu_grid_mev, distance_m, "nue", beam=beam)
    phi_numubar = differential_flux_delayed(enu_grid_mev, distance_m, "numubar", beam=beam)

    n_er = len(er_grid_kev)

    rate_prompt_total = np.zeros(n_er)
    rate_prompt_vector = np.zeros(n_er)
    rate_prompt_axial = np.zeros(n_er)

    rate_nue_total = np.zeros(n_er)
    rate_nue_vector = np.zeros(n_er)
    rate_nue_axial = np.zeros(n_er)

    rate_numubar_total = np.zeros(n_er)
    rate_numubar_vector = np.zeros(n_er)
    rate_numubar_axial = np.zeros(n_er)

    for i, er in enumerate(er_grid_kev):
        # Prompt monochromatic nu_mu line
        ds_prompt_total = calc.differential_cross_section_cm2_per_kev(
            target, E_NU_MU_PROMPT_MEV, er
        )
        ds_prompt_vector = calc.differential_vector_cross_section_cm2_per_kev(
            target, E_NU_MU_PROMPT_MEV, er
        )
        ds_prompt_axial = calc.differential_axial_cross_section_cm2_per_kev(
            target, E_NU_MU_PROMPT_MEV, er
        )

        rate_prompt_total[i] = phi_prompt_line * ds_prompt_total
        rate_prompt_vector[i] = phi_prompt_line * ds_prompt_vector
        rate_prompt_axial[i] = phi_prompt_line * ds_prompt_axial

        # Delayed nue
        ds_nue_total = build_dsigma_vs_enu(calc, target, enu_grid_mev, er, mode="total")
        ds_nue_vector = build_dsigma_vs_enu(calc, target, enu_grid_mev, er, mode="vector")
        ds_nue_axial = build_dsigma_vs_enu(calc, target, enu_grid_mev, er, mode="axial")

        rate_nue_total[i] = integrate_over_enu(enu_grid_mev, phi_nue, ds_nue_total)
        rate_nue_vector[i] = integrate_over_enu(enu_grid_mev, phi_nue, ds_nue_vector)
        rate_nue_axial[i] = integrate_over_enu(enu_grid_mev, phi_nue, ds_nue_axial)

        # Delayed anti-nu_mu
        ds_numubar_total = ds_nue_total
        ds_numubar_vector = ds_nue_vector
        ds_numubar_axial = ds_nue_axial
        # same CEvNS cross section in this SM implementation; only flux differs

        rate_numubar_total[i] = integrate_over_enu(enu_grid_mev, phi_numubar, ds_numubar_total)
        rate_numubar_vector[i] = integrate_over_enu(enu_grid_mev, phi_numubar, ds_numubar_vector)
        rate_numubar_axial[i] = integrate_over_enu(enu_grid_mev, phi_numubar, ds_numubar_axial)

    out = {
        "prompt_total": rate_prompt_total,
        "prompt_vector": rate_prompt_vector,
        "prompt_axial": rate_prompt_axial,
        "nue_total": rate_nue_total,
        "nue_vector": rate_nue_vector,
        "nue_axial": rate_nue_axial,
        "numubar_total": rate_numubar_total,
        "numubar_vector": rate_numubar_vector,
        "numubar_axial": rate_numubar_axial,
    }

    out["delayed_total"] = out["nue_total"] + out["numubar_total"]
    out["delayed_vector"] = out["nue_vector"] + out["numubar_vector"]
    out["delayed_axial"] = out["nue_axial"] + out["numubar_axial"]

    out["total"] = out["prompt_total"] + out["delayed_total"]
    out["total_vector"] = out["prompt_vector"] + out["delayed_vector"]
    out["total_axial"] = out["prompt_axial"] + out["delayed_axial"]

    return out


def integrate_rate_over_recoil(er_grid_kev: np.ndarray, rate_per_s_per_kev: np.ndarray) -> float:
    """
    Integral over recoil energy:
        R = ∫ dEr (dR/dEr)
    Returns s^-1
    """
    return float(np.trapz(rate_per_s_per_kev, er_grid_kev))


def integrate_rate_above_threshold(
    er_grid_kev: np.ndarray,
    rate_per_s_per_kev: np.ndarray,
    threshold_kev: float,
) -> float:
    mask = er_grid_kev >= threshold_kev
    if not np.any(mask):
        return 0.0
    return float(np.trapz(rate_per_s_per_kev[mask], er_grid_kev[mask]))


def write_csv(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
) -> None:
    """
    Output molecule-level and per-target rates.
    CF4 = 1*C + 4*F
    """
    c_mult = 1.0
    f_mult = 4.0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Er_keV",

            # carbon contributions per C nucleus
            "dR_dEr_C_prompt_total_per_s_per_keV_per_C",
            "dR_dEr_C_nue_total_per_s_per_keV_per_C",
            "dR_dEr_C_numubar_total_per_s_per_keV_per_C",
            "dR_dEr_C_delayed_total_per_s_per_keV_per_C",
            "dR_dEr_C_total_per_s_per_keV_per_C",

            # fluorine contributions per F nucleus
            "dR_dEr_F_prompt_total_per_s_per_keV_per_F",
            "dR_dEr_F_nue_total_per_s_per_keV_per_F",
            "dR_dEr_F_numubar_total_per_s_per_keV_per_F",
            "dR_dEr_F_delayed_total_per_s_per_keV_per_F",
            "dR_dEr_F_total_per_s_per_keV_per_F",

            # fluorine vector / axial
            "dR_dEr_F_total_vector_per_s_per_keV_per_F",
            "dR_dEr_F_total_axial_per_s_per_keV_per_F",
            "F_axial_fraction",

            # CF4 molecule totals
            "dR_dEr_CF4_prompt_total_per_s_per_keV_per_molecule",
            "dR_dEr_CF4_nue_total_per_s_per_keV_per_molecule",
            "dR_dEr_CF4_numubar_total_per_s_per_keV_per_molecule",
            "dR_dEr_CF4_delayed_total_per_s_per_keV_per_molecule",
            "dR_dEr_CF4_total_per_s_per_keV_per_molecule",

            # CF4 molecule decomposition
            "dR_dEr_CF4_C_piece_per_s_per_keV_per_molecule",
            "dR_dEr_CF4_4F_piece_per_s_per_keV_per_molecule",
            "CF4_F_fraction",
        ])

        for i, er in enumerate(er_grid_kev):
            cf4_prompt = c_mult * rates_c["prompt_total"][i] + f_mult * rates_f["prompt_total"][i]
            cf4_nue = c_mult * rates_c["nue_total"][i] + f_mult * rates_f["nue_total"][i]
            cf4_numubar = c_mult * rates_c["numubar_total"][i] + f_mult * rates_f["numubar_total"][i]
            cf4_delayed = c_mult * rates_c["delayed_total"][i] + f_mult * rates_f["delayed_total"][i]
            cf4_total = c_mult * rates_c["total"][i] + f_mult * rates_f["total"][i]

            cf4_c_piece = c_mult * rates_c["total"][i]
            cf4_f_piece = f_mult * rates_f["total"][i]

            f_ax_frac = (
                rates_f["total_axial"][i] / rates_f["total"][i]
                if rates_f["total"][i] > 0.0 else 0.0
            )
            cf4_f_frac = cf4_f_piece / cf4_total if cf4_total > 0.0 else 0.0

            writer.writerow([
                er,

                rates_c["prompt_total"][i],
                rates_c["nue_total"][i],
                rates_c["numubar_total"][i],
                rates_c["delayed_total"][i],
                rates_c["total"][i],

                rates_f["prompt_total"][i],
                rates_f["nue_total"][i],
                rates_f["numubar_total"][i],
                rates_f["delayed_total"][i],
                rates_f["total"][i],

                rates_f["total_vector"][i],
                rates_f["total_axial"][i],
                f_ax_frac,

                cf4_prompt,
                cf4_nue,
                cf4_numubar,
                cf4_delayed,
                cf4_total,

                cf4_c_piece,
                cf4_f_piece,
                cf4_f_frac,
            ])


def plot_cf4_rates(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
) -> None:
    c_mult = 1.0
    f_mult = 4.0

    cf4_prompt = c_mult * rates_c["prompt_total"] + f_mult * rates_f["prompt_total"]
    cf4_nue = c_mult * rates_c["nue_total"] + f_mult * rates_f["nue_total"]
    cf4_numubar = c_mult * rates_c["numubar_total"] + f_mult * rates_f["numubar_total"]
    cf4_delayed = c_mult * rates_c["delayed_total"] + f_mult * rates_f["delayed_total"]
    cf4_total = c_mult * rates_c["total"] + f_mult * rates_f["total"]

    plt.figure(figsize=(8, 5))
    plt.plot(er_grid_kev, cf4_prompt, label=r"prompt $\nu_\mu$")
    plt.plot(er_grid_kev, cf4_nue, label=r"delayed $\nu_e$")
    plt.plot(er_grid_kev, cf4_numubar, label=r"delayed $\bar{\nu}_\mu$")
    plt.plot(er_grid_kev, cf4_delayed, "--", label="delayed total")
    plt.plot(er_grid_kev, cf4_total, linewidth=2, label="total")
    plt.yscale("log")
    plt.xlabel("Recoil energy [keV]")
    plt.ylabel(r"$dR/dE_r$ [s$^{-1}$ keV$^{-1}$ molecule$^{-1}$]")
    plt.title("CF4 differential CEvNS rate at ESS-like pion-DAR source")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_cf4_composition(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
) -> None:
    cf4_c_piece = rates_c["total"]
    cf4_4f_piece = 4.0 * rates_f["total"]
    cf4_total = cf4_c_piece + cf4_4f_piece

    plt.figure(figsize=(8, 5))
    plt.plot(er_grid_kev, cf4_c_piece, label="1 x 12C contribution")
    plt.plot(er_grid_kev, cf4_4f_piece, label="4 x 19F contribution")
    plt.plot(er_grid_kev, cf4_total, linewidth=2, label="CF4 total")
    plt.yscale("log")
    plt.xlabel("Recoil energy [keV]")
    plt.ylabel(r"$dR/dE_r$ [s$^{-1}$ keV$^{-1}$ molecule$^{-1}$]")
    plt.title("CF4 composition of the CEvNS recoil spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_fluorine_axial_fraction(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_f: Dict[str, np.ndarray],
) -> None:
    frac = np.zeros_like(er_grid_kev)
    mask = rates_f["total"] > 0.0
    frac[mask] = rates_f["total_axial"][mask] / rates_f["total"][mask]

    plt.figure(figsize=(8, 5))
    plt.plot(er_grid_kev, frac)
    plt.xlabel("Recoil energy [keV]")
    plt.ylabel("Axial fraction in 19F total rate")
    plt.title("19F axial fraction in the flux-folded CEvNS rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def print_summary(
    cfg: RateConfig,
    beam: ESSBeamConfig,
    carbon,
    fluorine,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
    er_grid_kev: np.ndarray,
) -> None:
    cf4_prompt = rates_c["prompt_total"] + 4.0 * rates_f["prompt_total"]
    cf4_nue = rates_c["nue_total"] + 4.0 * rates_f["nue_total"]
    cf4_numubar = rates_c["numubar_total"] + 4.0 * rates_f["numubar_total"]
    cf4_delayed = rates_c["delayed_total"] + 4.0 * rates_f["delayed_total"]
    cf4_total = rates_c["total"] + 4.0 * rates_f["total"]

    cf4_total_integrated = integrate_rate_over_recoil(er_grid_kev, cf4_total)
    cf4_prompt_integrated = integrate_rate_over_recoil(er_grid_kev, cf4_prompt)
    cf4_delayed_integrated = integrate_rate_over_recoil(er_grid_kev, cf4_delayed)

    cf4_above_thr = integrate_rate_above_threshold(er_grid_kev, cf4_total, cfg.threshold_kev)

    f_total = integrate_rate_over_recoil(er_grid_kev, rates_f["total"])
    f_axial = integrate_rate_over_recoil(er_grid_kev, rates_f["total_axial"])
    f_axial_fraction = f_axial / f_total if f_total > 0.0 else 0.0

    c_piece = integrate_rate_over_recoil(er_grid_kev, rates_c["total"])
    f_piece = integrate_rate_over_recoil(er_grid_kev, 4.0 * rates_f["total"])
    f_molecule_fraction = f_piece / (c_piece + f_piece) if (c_piece + f_piece) > 0.0 else 0.0

    print("=== Configuration ===")
    print(f"Distance to source              : {cfg.distance_m:.3f} m")
    print(f"Prompt neutrino energy          : {E_NU_MU_PROMPT_MEV:.6f} MeV")
    print(f"Delayed endpoint                : {E_NU_MAX_MEV:.6f} MeV")
    print(f"Recoil grid                     : {cfg.er_min_kev:.3f} -> {cfg.er_max_kev:.3f} keV ({cfg.n_er} bins)")
    print(f"Delayed E_nu grid               : {cfg.enu_min_mev:.3f} -> {cfg.enu_max_mev:.3f} MeV ({cfg.n_enu} bins)")
    print(f"Threshold for summary integral  : {cfg.threshold_kev:.3f} keV")
    print()

    print("=== Target kinematic endpoints ===")
    print(f"12C: Er_max(prompt)             : {carbon.max_recoil_kev(E_NU_MU_PROMPT_MEV):.6f} keV")
    print(f"12C: Er_max(delayed endpoint)   : {carbon.max_recoil_kev(E_NU_MAX_MEV):.6f} keV")
    print(f"19F: Er_max(prompt)             : {fluorine.max_recoil_kev(E_NU_MU_PROMPT_MEV):.6f} keV")
    print(f"19F: Er_max(delayed endpoint)   : {fluorine.max_recoil_kev(E_NU_MAX_MEV):.6f} keV")
    print()

    print("=== CF4 molecule integrated rates ===")
    print(f"Prompt rate                     : {cf4_prompt_integrated:.6e} s^-1 molecule^-1")
    print(f"Delayed rate                    : {cf4_delayed_integrated:.6e} s^-1 molecule^-1")
    print(f"  - nue                         : {integrate_rate_over_recoil(er_grid_kev, cf4_nue):.6e} s^-1 molecule^-1")
    print(f"  - numubar                     : {integrate_rate_over_recoil(er_grid_kev, cf4_numubar):.6e} s^-1 molecule^-1")
    print(f"Total rate                      : {cf4_total_integrated:.6e} s^-1 molecule^-1")
    print(f"Rate above {cfg.threshold_kev:.3f} keV         : {cf4_above_thr:.6e} s^-1 molecule^-1")
    print()

    print("=== Useful derived fractions ===")
    print(f"Prompt / total                  : {cf4_prompt_integrated / cf4_total_integrated:.6f}" if cf4_total_integrated > 0 else "Prompt / total                  : 0")
    print(f"Delayed / total                 : {cf4_delayed_integrated / cf4_total_integrated:.6f}" if cf4_total_integrated > 0 else "Delayed / total                 : 0")
    print(f"4F contribution / CF4 total     : {f_molecule_fraction:.6f}")
    print(f"19F axial / 19F total           : {f_axial_fraction:.6f}")
    print()

    print("=== Beam numbers ===")
    print(f"Neutrinos/s/flavor              : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Neutrinos/year/flavor           : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"Duty factor                     : {beam.duty_factor:.6e}")
    print(f"Protons/pulse                   : {beam.protons_per_pulse:.6e}")


def main() -> None:
    cfg = RateConfig()
    beam = ESSBeamConfig()
    calc = CEvNSCalculator()

    carbon = carbon12_target()
    fluorine = fluorine19_target(
        axial_model=cfg.fluorine_axial_model,
        Sp=cfg.fluorine_sp,
        Sn=cfg.fluorine_sn,
        lambda_a_gev=cfg.fluorine_lambda_a_gev,
    )

    er_grid_kev = np.linspace(cfg.er_min_kev, cfg.er_max_kev, cfg.n_er)
    enu_grid_mev = np.linspace(cfg.enu_min_mev, cfg.enu_max_mev, cfg.n_enu)

    rates_c = compute_component_rates_per_target(
        calc=calc,
        target=carbon,
        er_grid_kev=er_grid_kev,
        enu_grid_mev=enu_grid_mev,
        beam=beam,
        distance_m=cfg.distance_m,
    )

    rates_f = compute_component_rates_per_target(
        calc=calc,
        target=fluorine,
        er_grid_kev=er_grid_kev,
        enu_grid_mev=enu_grid_mev,
        beam=beam,
        distance_m=cfg.distance_m,
    )

    csv_file = os.path.join(OUTPUT_DIR, "cf4_differential_rate_per_molecule.csv")
    fig_rate = os.path.join(OUTPUT_DIR, "cf4_differential_rate_components.png")
    fig_comp = os.path.join(OUTPUT_DIR, "cf4_composition_c_vs_4f.png")
    fig_ax = os.path.join(OUTPUT_DIR, "fluorine_axial_fraction_flux_folded.png")

    write_csv(csv_file, er_grid_kev, rates_c, rates_f)
    plot_cf4_rates(fig_rate, er_grid_kev, rates_c, rates_f)
    plot_cf4_composition(fig_comp, er_grid_kev, rates_c, rates_f)
    plot_fluorine_axial_fraction(fig_ax, er_grid_kev, rates_f)

    print_summary(cfg, beam, carbon, fluorine, rates_c, rates_f, er_grid_kev)

    print()
    print("Saved files:")
    print(f"  {csv_file}")
    print(f"  {fig_rate}")
    print(f"  {fig_comp}")
    print(f"  {fig_ax}")


if __name__ == "__main__":
    main()