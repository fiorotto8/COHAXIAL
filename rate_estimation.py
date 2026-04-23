from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import ESS_flux
import JPARK_flux
import SNS_flux
from cevens import (
    CEvNSCalculator,
    NeutrinoElectronCalculator,
    cf4_electron_target,
    carbon12_target,
    fluorine19_target,
)

# ============================================================
# CF4 differential-rate scan at a benchmark pion-DAR source
#
# What this script computes
# -------------------------
# For a CF4 molecule, it computes:
#
#   dR/dEr  [s^-1 keV^-1 molecule^-1]   for CEvNS nuclear recoils
#   dR/dTe  [s^-1 keV^-1 molecule^-1]   for neutrino-electron recoils
#
# split into source components:
#   - prompt nu_mu contribution
#   - delayed nu_e contribution
#   - delayed anti-nu_mu contribution
#   - total delayed contribution
#   - total contribution
#
# For CEvNS it also stores separate C and F pieces, plus the vector / axial
# split for fluorine, so you can inspect how much of the molecule-level rate
# is driven by 19F and how large the axial fraction is.
#
# For neutrino-electron scattering it stores both per-electron and per-CF4
# molecule recoil spectra, where the molecule-level rate includes all
# 42 target electrons in CF4.
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


@dataclass(frozen=True)
class DARSourceModel:
    key: str
    label: str
    beam_factory: object
    prompt_energy_mev: float
    delayed_endpoint_mev: float
    differential_flux_delayed: object
    prompt_numu_line_flux: object
    default_output_dir: str


SOURCE_MODELS: Dict[str, DARSourceModel] = {
    "ess": DARSourceModel(
        key="ess",
        label="ESS",
        beam_factory=ESS_flux.ESSBeamConfig,
        prompt_energy_mev=ESS_flux.E_NU_MU_PROMPT_MEV,
        delayed_endpoint_mev=ESS_flux.E_NU_MAX_MEV,
        differential_flux_delayed=ESS_flux.differential_flux_delayed,
        prompt_numu_line_flux=ESS_flux.prompt_numu_line_flux,
        default_output_dir="cevens_rate_output",
    ),
    "sns": DARSourceModel(
        key="sns",
        label="SNS FTS",
        beam_factory=SNS_flux.SNSBeamConfig,
        prompt_energy_mev=SNS_flux.E_NU_MU_PROMPT_MEV,
        delayed_endpoint_mev=SNS_flux.E_NU_MAX_MEV,
        differential_flux_delayed=SNS_flux.differential_flux_delayed,
        prompt_numu_line_flux=SNS_flux.prompt_numu_line_flux,
        default_output_dir="cevens_rate_output_sns",
    ),
    "jparc": DARSourceModel(
        key="jparc",
        label="J-PARC MLF",
        beam_factory=JPARK_flux.JPARCMLFBeamConfig,
        prompt_energy_mev=JPARK_flux.E_NU_MU_PROMPT_MEV,
        delayed_endpoint_mev=JPARK_flux.E_NU_MAX_MEV,
        differential_flux_delayed=JPARK_flux.differential_flux_delayed,
        prompt_numu_line_flux=JPARK_flux.prompt_numu_line_flux,
        default_output_dir="cevens_rate_output_jparc",
    ),
}


def get_source_model(source_model: str) -> DARSourceModel:
    key = source_model.strip().lower().replace("-", "_")
    aliases = {
        "ess": "ess",
        "sns": "sns",
        "jparc": "jparc",
        "j_parc": "jparc",
        "jparc_mlf": "jparc",
        "mlf": "jparc",
    }
    normalized = aliases.get(key)
    if normalized is None:
        raise ValueError("source_model must be one of: 'ess', 'sns', 'jparc'")
    return SOURCE_MODELS[normalized]


class ProgressReporter:
    """Small stderr progress reporter for long scans; numerical logic is unchanged."""

    def __init__(
        self,
        label: Optional[str],
        total: int,
        *,
        min_interval_s: float = 2.0,
        n_updates: int = 50,
    ) -> None:
        self.label = label or ""
        self.total = max(0, int(total))
        self.enabled = bool(self.label) and self.total > 0
        self.min_interval_s = float(min_interval_s)
        self.update_step = max(1, self.total // max(1, int(n_updates)))
        self.start_time = time.monotonic()
        self.last_emit_time = self.start_time
        self.last_emit_completed = 0
        self.finished = False
        self.use_carriage_return = sys.stderr.isatty()

        if self.enabled:
            self._emit(0, final=False)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            return f"{seconds:.0f}s"
        minutes, sec = divmod(int(round(seconds)), 60)
        if minutes < 60:
            return f"{minutes:d}m{sec:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"

    def _emit(self, completed: int, *, final: bool) -> None:
        now = time.monotonic()
        elapsed = now - self.start_time
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        eta_text = (
            self._format_duration((self.total - completed) / rate)
            if rate > 0.0
            else "--"
        )
        pct = 100.0 * completed / self.total if self.total else 100.0

        line = (
            f"[{self.label}] {pct:6.2f}% "
            f"({completed}/{self.total}) "
            f"elapsed {self._format_duration(elapsed)} "
            f"ETA {eta_text} "
            f"rate {rate:.2f} grid-points/s"
        )
        if final:
            line += " done"

        if self.use_carriage_return:
            print("\r" + line, end="\n" if final else "", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

        self.last_emit_time = now
        self.last_emit_completed = completed

    def update(self, completed: int) -> None:
        if not self.enabled or self.finished:
            return

        completed = min(max(0, int(completed)), self.total)
        now = time.monotonic()
        final = completed >= self.total
        should_emit = (
            final
            or completed == 1
            or completed - self.last_emit_completed >= self.update_step
            or now - self.last_emit_time >= self.min_interval_s
        )
        if not should_emit:
            return

        self._emit(completed, final=final)
        self.finished = final

    def done(self) -> None:
        self.update(self.total)


@dataclass
class RateConfig:
    # DAR source model used in the flux folding.
    # Options: "ess", "sns", "jparc".
    source_model: str = "ess"
    distance_m: float = 20.0
    output_dir: Optional[str] = None

    # nuclear recoil-energy grid for CEvNS
    er_min_kev: float = 0.0
    er_max_kev: float = 120.0
    n_er: int = 1201

    # electron recoil-energy grid for neutrino-electron scattering
    te_min_kev: float = 0.0
    te_max_kev: float = 60000.0
    n_te: int = 3001

    # neutrino-energy grid for delayed components
    enu_min_mev: float = 1E-6
    enu_max_mev: Optional[float] = None
    n_enu: int = 3000

    # optional threshold for integrated-above-threshold summaries
    threshold_kev: float = 0.0

    # 19F axial model used by the CF4 rate scan.
    # Options: "hoferichter_19f_fast", "hoferichter_19f_central", "none", "toy".
    fluorine_axial_model: str = "hoferichter_19f_central"
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


def build_dsigma_electron_vs_enu(
    calc: NeutrinoElectronCalculator,
    flavor: str,
    enu_grid_mev: np.ndarray,
    te_kev: float,
) -> np.ndarray:
    out = np.zeros_like(enu_grid_mev, dtype=float)

    for i, enu in enumerate(enu_grid_mev):
        out[i] = calc.differential_cross_section_cm2_per_kev(flavor, enu, te_kev)

    return out


def compute_component_rates_per_target(
    calc: CEvNSCalculator,
    target,
    er_grid_kev: np.ndarray,
    enu_grid_mev: np.ndarray,
    source: DARSourceModel,
    beam,
    distance_m: float,
    progress_label: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-target rates:
        dR/dEr [s^-1 keV^-1 target^-1]
    with prompt and delayed components split.
    """
    phi_prompt_line = source.prompt_numu_line_flux(distance_m, beam=beam)  # cm^-2 s^-1
    phi_nue = source.differential_flux_delayed(enu_grid_mev, distance_m, "nue", beam=beam)
    phi_numubar = source.differential_flux_delayed(enu_grid_mev, distance_m, "numubar", beam=beam)

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

    progress = ProgressReporter(progress_label, n_er)

    for i, er in enumerate(er_grid_kev):
        # Prompt monochromatic nu_mu line
        ds_prompt_total = calc.differential_cross_section_cm2_per_kev(
            target, source.prompt_energy_mev, er
        )
        ds_prompt_vector = calc.differential_vector_cross_section_cm2_per_kev(
            target, source.prompt_energy_mev, er
        )
        ds_prompt_axial = calc.differential_axial_cross_section_cm2_per_kev(
            target, source.prompt_energy_mev, er
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

        progress.update(i + 1)

    progress.done()

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


def compute_electron_scattering_rates(
    calc: NeutrinoElectronCalculator,
    te_grid_kev: np.ndarray,
    enu_grid_mev: np.ndarray,
    source: DARSourceModel,
    beam,
    distance_m: float,
    progress_label: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Electron-scattering rates:
        dR/dTe [s^-1 keV^-1]

    Returns both per-electron and per-molecule rates. The molecule-level
    rates include all target electrons in the configured electron target.
    """
    phi_prompt_line = source.prompt_numu_line_flux(distance_m, beam=beam)  # cm^-2 s^-1
    phi_nue = source.differential_flux_delayed(enu_grid_mev, distance_m, "nue", beam=beam)
    phi_numubar = source.differential_flux_delayed(enu_grid_mev, distance_m, "numubar", beam=beam)

    n_te = len(te_grid_kev)
    prompt_per_electron = np.zeros(n_te)
    nue_per_electron = np.zeros(n_te)
    numubar_per_electron = np.zeros(n_te)

    progress = ProgressReporter(progress_label, n_te)

    for i, te in enumerate(te_grid_kev):
        ds_prompt = calc.differential_cross_section_cm2_per_kev("numu", source.prompt_energy_mev, te)
        prompt_per_electron[i] = phi_prompt_line * ds_prompt

        ds_nue = build_dsigma_electron_vs_enu(calc, "nue", enu_grid_mev, te)
        ds_numubar = build_dsigma_electron_vs_enu(calc, "numubar", enu_grid_mev, te)

        nue_per_electron[i] = integrate_over_enu(enu_grid_mev, phi_nue, ds_nue)
        numubar_per_electron[i] = integrate_over_enu(enu_grid_mev, phi_numubar, ds_numubar)

        progress.update(i + 1)

    progress.done()

    delayed_per_electron = nue_per_electron + numubar_per_electron
    total_per_electron = prompt_per_electron + delayed_per_electron

    electron_multiplier = float(calc.electron_target.electrons_per_molecule)

    return {
        "prompt_per_electron": prompt_per_electron,
        "nue_per_electron": nue_per_electron,
        "numubar_per_electron": numubar_per_electron,
        "delayed_per_electron": delayed_per_electron,
        "total_per_electron": total_per_electron,
        "prompt_per_molecule": electron_multiplier * prompt_per_electron,
        "nue_per_molecule": electron_multiplier * nue_per_electron,
        "numubar_per_molecule": electron_multiplier * numubar_per_electron,
        "delayed_per_molecule": electron_multiplier * delayed_per_electron,
        "total_per_molecule": electron_multiplier * total_per_electron,
        "electrons_per_molecule": np.full(n_te, electron_multiplier),
    }


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


def write_electron_csv(
    filename: str,
    te_grid_kev: np.ndarray,
    electron_rates: Dict[str, np.ndarray],
) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Te_keV",
            "dR_dTe_numu_prompt_per_s_per_keV_per_electron",
            "dR_dTe_nue_delayed_per_s_per_keV_per_electron",
            "dR_dTe_numubar_delayed_per_s_per_keV_per_electron",
            "dR_dTe_delayed_total_per_s_per_keV_per_electron",
            "dR_dTe_total_per_s_per_keV_per_electron",
            "dR_dTe_CF4_numu_prompt_per_s_per_keV_per_molecule",
            "dR_dTe_CF4_nue_delayed_per_s_per_keV_per_molecule",
            "dR_dTe_CF4_numubar_delayed_per_s_per_keV_per_molecule",
            "dR_dTe_CF4_delayed_total_per_s_per_keV_per_molecule",
            "dR_dTe_CF4_total_per_s_per_keV_per_molecule",
            "CF4_electrons_per_molecule",
        ])

        for i, te in enumerate(te_grid_kev):
            writer.writerow([
                te,
                electron_rates["prompt_per_electron"][i],
                electron_rates["nue_per_electron"][i],
                electron_rates["numubar_per_electron"][i],
                electron_rates["delayed_per_electron"][i],
                electron_rates["total_per_electron"][i],
                electron_rates["prompt_per_molecule"][i],
                electron_rates["nue_per_molecule"][i],
                electron_rates["numubar_per_molecule"][i],
                electron_rates["delayed_per_molecule"][i],
                electron_rates["total_per_molecule"][i],
                electron_rates["electrons_per_molecule"][i],
            ])


def plot_cf4_rates(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
    source_label: str,
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
    plt.title(f"CF4 differential CEvNS rate at {source_label} pion-DAR source")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_cf4_electron_rates(
    filename: str,
    te_grid_kev: np.ndarray,
    electron_rates: Dict[str, np.ndarray],
    source_label: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(te_grid_kev, electron_rates["prompt_per_molecule"], label=r"prompt $\nu_\mu e$")
    plt.plot(te_grid_kev, electron_rates["nue_per_molecule"], label=r"delayed $\nu_e e$")
    plt.plot(te_grid_kev, electron_rates["numubar_per_molecule"], label=r"delayed $\bar{\nu}_\mu e$")
    plt.plot(te_grid_kev, electron_rates["delayed_per_molecule"], "--", label="delayed total")
    plt.plot(te_grid_kev, electron_rates["total_per_molecule"], linewidth=2, label="total")
    plt.yscale("log")
    plt.xlabel("Electron recoil energy [keV]")
    plt.ylabel(r"$dR/dT_e$ [s$^{-1}$ keV$^{-1}$ molecule$^{-1}$]")
    plt.title(rf"CF4 differential $\nu$-e rate at {source_label} pion-DAR source")
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
    source_label: str,
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
    plt.title(f"CF4 composition of the CEvNS recoil spectrum at {source_label}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_fluorine_axial_fraction(
    filename: str,
    er_grid_kev: np.ndarray,
    rates_f: Dict[str, np.ndarray],
    source_label: str,
) -> None:
    frac = np.zeros_like(er_grid_kev)
    mask = rates_f["total"] > 0.0
    frac[mask] = rates_f["total_axial"][mask] / rates_f["total"][mask]

    plt.figure(figsize=(8, 5))
    plt.plot(er_grid_kev, frac)
    plt.xlabel("Recoil energy [keV]")
    plt.ylabel("Axial fraction in 19F total rate")
    plt.title(f"19F axial fraction in the flux-folded CEvNS rate at {source_label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def print_summary(
    cfg: RateConfig,
    source: DARSourceModel,
    beam,
    carbon,
    fluorine,
    electron_target,
    rates_c: Dict[str, np.ndarray],
    rates_f: Dict[str, np.ndarray],
    electron_rates: Dict[str, np.ndarray],
    er_grid_kev: np.ndarray,
    te_grid_kev: np.ndarray,
) -> None:
    effective_enu_max_mev = cfg.enu_max_mev if cfg.enu_max_mev is not None else source.delayed_endpoint_mev

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

    electron_prompt_integrated = integrate_rate_over_recoil(te_grid_kev, electron_rates["prompt_per_molecule"])
    electron_nue_integrated = integrate_rate_over_recoil(te_grid_kev, electron_rates["nue_per_molecule"])
    electron_numubar_integrated = integrate_rate_over_recoil(te_grid_kev, electron_rates["numubar_per_molecule"])
    electron_delayed_integrated = integrate_rate_over_recoil(te_grid_kev, electron_rates["delayed_per_molecule"])
    electron_total_integrated = integrate_rate_over_recoil(te_grid_kev, electron_rates["total_per_molecule"])
    electron_above_thr = integrate_rate_above_threshold(
        te_grid_kev,
        electron_rates["total_per_molecule"],
        cfg.threshold_kev,
    )

    print("=== Configuration ===")
    print(f"Source model                    : {source.label} ({source.key})")
    print(f"Distance to source              : {cfg.distance_m:.3f} m")
    print(f"Prompt neutrino energy          : {source.prompt_energy_mev:.6f} MeV")
    print(f"Delayed endpoint                : {source.delayed_endpoint_mev:.6f} MeV")
    print(f"Nuclear recoil grid             : {cfg.er_min_kev:.3f} -> {cfg.er_max_kev:.3f} keV ({cfg.n_er} bins)")
    print(f"Electron recoil grid            : {cfg.te_min_kev:.3f} -> {cfg.te_max_kev:.3f} keV ({cfg.n_te} bins)")
    print(f"Delayed E_nu grid               : {cfg.enu_min_mev:.3f} -> {effective_enu_max_mev:.3f} MeV ({cfg.n_enu} bins)")
    print(f"Threshold for summary integral  : {cfg.threshold_kev:.3f} keV")
    print()

    print("=== Target kinematic endpoints ===")
    print(f"12C: Er_max(prompt)             : {carbon.max_recoil_kev(source.prompt_energy_mev):.6f} keV")
    print(f"12C: Er_max(delayed endpoint)   : {carbon.max_recoil_kev(source.delayed_endpoint_mev):.6f} keV")
    print(f"19F: Er_max(prompt)             : {fluorine.max_recoil_kev(source.prompt_energy_mev):.6f} keV")
    print(f"19F: Er_max(delayed endpoint)   : {fluorine.max_recoil_kev(source.delayed_endpoint_mev):.6f} keV")
    print(f"e- : Te_max(prompt)             : {electron_target.max_recoil_kev(source.prompt_energy_mev):.6f} keV")
    print(f"e- : Te_max(delayed endpoint)   : {electron_target.max_recoil_kev(source.delayed_endpoint_mev):.6f} keV")
    print()

    print("=== CF4 molecule integrated CEvNS rates ===")
    print(f"Prompt rate                     : {cf4_prompt_integrated:.6e} s^-1 molecule^-1")
    print(f"Delayed rate                    : {cf4_delayed_integrated:.6e} s^-1 molecule^-1")
    print(f"  - nue                         : {integrate_rate_over_recoil(er_grid_kev, cf4_nue):.6e} s^-1 molecule^-1")
    print(f"  - numubar                     : {integrate_rate_over_recoil(er_grid_kev, cf4_numubar):.6e} s^-1 molecule^-1")
    print(f"Total rate                      : {cf4_total_integrated:.6e} s^-1 molecule^-1")
    print(f"Rate above {cfg.threshold_kev:.3f} keV         : {cf4_above_thr:.6e} s^-1 molecule^-1")
    print()

    print("=== CF4 molecule integrated nu-e rates ===")
    print(f"Target electrons / molecule     : {electron_target.electrons_per_molecule}")
    print(f"Prompt rate                     : {electron_prompt_integrated:.6e} s^-1 molecule^-1")
    print(f"Delayed rate                    : {electron_delayed_integrated:.6e} s^-1 molecule^-1")
    print(f"  - nue                         : {electron_nue_integrated:.6e} s^-1 molecule^-1")
    print(f"  - numubar                     : {electron_numubar_integrated:.6e} s^-1 molecule^-1")
    print(f"Total rate                      : {electron_total_integrated:.6e} s^-1 molecule^-1")
    print(f"Rate above {cfg.threshold_kev:.3f} keV         : {electron_above_thr:.6e} s^-1 molecule^-1")
    print()

    print("=== Useful derived fractions ===")
    print(f"Prompt / total                  : {cf4_prompt_integrated / cf4_total_integrated:.6f}" if cf4_total_integrated > 0 else "Prompt / total                  : 0")
    print(f"Delayed / total                 : {cf4_delayed_integrated / cf4_total_integrated:.6f}" if cf4_total_integrated > 0 else "Delayed / total                 : 0")
    print(f"4F contribution / CF4 total     : {f_molecule_fraction:.6f}")
    print(f"19F axial / 19F total           : {f_axial_fraction:.6f}")
    print(
        f"nu-e / CEvNS above threshold    : {electron_above_thr / cf4_above_thr:.6f}"
        if cf4_above_thr > 0.0
        else "nu-e / CEvNS above threshold    : 0"
    )
    print()

    print("=== Beam numbers ===")
    print(f"Neutrinos/s/flavor              : {beam.neutrinos_per_second_per_flavor:.6e}")
    print(f"Neutrinos/year/flavor           : {beam.neutrinos_per_year_per_flavor:.6e}")
    print(f"Duty factor                     : {beam.duty_factor:.6e}")
    print(f"Protons/pulse                   : {beam.protons_per_pulse:.6e}")
    if hasattr(beam, "yield_fractional_uncertainty"):
        print(f"Yield fractional uncertainty    : {beam.yield_fractional_uncertainty:.3%}")


def main() -> None:
    cfg = RateConfig()
    source = get_source_model(cfg.source_model)
    beam = source.beam_factory()
    output_dir = cfg.output_dir or source.default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    calc = CEvNSCalculator()
    electron_target = cf4_electron_target()
    electron_calc = NeutrinoElectronCalculator(electron_target=electron_target)

    carbon = carbon12_target()
    fluorine = fluorine19_target(
        axial_model=cfg.fluorine_axial_model,
        Sp=cfg.fluorine_sp,
        Sn=cfg.fluorine_sn,
        lambda_a_gev=cfg.fluorine_lambda_a_gev,
    )

    er_grid_kev = np.linspace(cfg.er_min_kev, cfg.er_max_kev, cfg.n_er)
    te_grid_kev = np.linspace(cfg.te_min_kev, cfg.te_max_kev, cfg.n_te)
    enu_max_mev = cfg.enu_max_mev if cfg.enu_max_mev is not None else source.delayed_endpoint_mev
    enu_grid_mev = np.linspace(cfg.enu_min_mev, enu_max_mev, cfg.n_enu)

    rates_c = compute_component_rates_per_target(
        calc=calc,
        target=carbon,
        er_grid_kev=er_grid_kev,
        enu_grid_mev=enu_grid_mev,
        source=source,
        beam=beam,
        distance_m=cfg.distance_m,
        progress_label=f"{source.label} CEvNS 12C recoil grid",
    )

    rates_f = compute_component_rates_per_target(
        calc=calc,
        target=fluorine,
        er_grid_kev=er_grid_kev,
        enu_grid_mev=enu_grid_mev,
        source=source,
        beam=beam,
        distance_m=cfg.distance_m,
        progress_label=f"{source.label} CEvNS 19F recoil grid",
    )

    electron_rates = compute_electron_scattering_rates(
        calc=electron_calc,
        te_grid_kev=te_grid_kev,
        enu_grid_mev=enu_grid_mev,
        source=source,
        beam=beam,
        distance_m=cfg.distance_m,
        progress_label=f"{source.label} nu-e CF4 electron grid",
    )

    csv_file = os.path.join(output_dir, "cf4_differential_rate_per_molecule.csv")
    electron_csv_file = os.path.join(output_dir, "cf4_electron_differential_rate_per_molecule.csv")
    fig_rate = os.path.join(output_dir, "cf4_differential_rate_components.png")
    fig_electron_rate = os.path.join(output_dir, "cf4_electron_differential_rate_components.png")
    fig_comp = os.path.join(output_dir, "cf4_composition_c_vs_4f.png")
    fig_ax = os.path.join(output_dir, "fluorine_axial_fraction_flux_folded.png")

    write_csv(csv_file, er_grid_kev, rates_c, rates_f)
    write_electron_csv(electron_csv_file, te_grid_kev, electron_rates)
    plot_cf4_rates(fig_rate, er_grid_kev, rates_c, rates_f, source.label)
    plot_cf4_electron_rates(fig_electron_rate, te_grid_kev, electron_rates, source.label)
    plot_cf4_composition(fig_comp, er_grid_kev, rates_c, rates_f, source.label)
    plot_fluorine_axial_fraction(fig_ax, er_grid_kev, rates_f, source.label)

    print_summary(
        cfg,
        source,
        beam,
        carbon,
        fluorine,
        electron_target,
        rates_c,
        rates_f,
        electron_rates,
        er_grid_kev,
        te_grid_kev,
    )

    print()
    print("Saved files:")
    print(f"  {csv_file}")
    print(f"  {electron_csv_file}")
    print(f"  {fig_rate}")
    print(f"  {fig_electron_rate}")
    print(f"  {fig_comp}")
    print(f"  {fig_ax}")


if __name__ == "__main__":
    main()
