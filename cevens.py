from __future__ import annotations

"""
Minimal, extensible CEvNS module with explicit vector and axial pieces.

Design goals
------------
- Clean separation between nuclear target data, vector form factor, axial form factor,
  and the cross-section calculator.
- Safe defaults for spin-zero nuclei (e.g. 12C => axial = 0).
- A generic exact interface for the axial structure functions S_00, S_01, S_11.
- A simple approximate axial model for 19F, useful for fast studies until a full
  shell-model table is plugged in.
- Units chosen for convenience in phenomenology:
    * neutrino energy: MeV
    * recoil energy: keV
    * masses: GeV internally
    * differential cross section: cm^2 / keV

Physics notes
-------------
The implemented differential cross section follows the standard CEvNS split into
vector and pure-axial pieces,

    dσ/dEr = G_F^2 m_N / (4π) [
        (1 - m_N Er / (2 Eν^2) - Er/Eν) Q_W^2 |F_W(q^2)|^2
      + (1 + m_N Er / (2 Eν^2) - Er/Eν) F_A(q^2)
    ]

with q^2 = 2 m_N Er.

For the axial term, a generic implementation of

    F_A = 8π/(2J+1) [ (gAs)^2 S00 - gA gAs S01 + gA^2 S11 ]

is provided.

The exact shell-model / ab-initio S_ij(q^2) tables are NOT hard-coded here.
You can inject them later as callables or interpolation tables.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Protocol
import argparse
import json
import math


# =========================
# Constants and conversions
# =========================

GF_GEV = 1.1663787e-5  # Fermi constant [GeV^-2]
SIN2_THETA_W = 0.23857
G_V_PROTON = 0.5 - 2.0 * SIN2_THETA_W
G_V_NEUTRON = -0.5
G_A_NUCLEON = 1.27
G_A_STRANGE_DEFAULT = 0.0
HBARC_GEV_CM = 1.973269804e-14  # [GeV cm]
GEV2_TO_CM2 = HBARC_GEV_CM ** 2
AMU_TO_GEV = 0.93149410242
FM_TO_GEV_INV = 5.067730716156395


def mev_to_gev(x_mev: float) -> float:
    return x_mev * 1.0e-3


def kev_to_gev(x_kev: float) -> float:
    return x_kev * 1.0e-6


def gev_to_kev(x_gev: float) -> float:
    return x_gev * 1.0e6


# =========================
# Protocols / interfaces
# =========================

class ScalarQ2Function(Protocol):
    def __call__(self, q2_gev2: float) -> float:
        ...


class VectorFormFactor(Protocol):
    def __call__(self, q2_gev2: float) -> float:
        ...


class AxialStructureFunctions(Protocol):
    def s00(self, q2_gev2: float) -> float:
        ...

    def s01(self, q2_gev2: float) -> float:
        ...

    def s11(self, q2_gev2: float) -> float:
        ...


# =========================
# Vector form factors
# =========================

@dataclass(frozen=True)
class HelmFormFactor:
    """Helm form factor.

    Parameters are in fm. If not provided, standard phenomenological defaults are used:
      c = 1.23 A^(1/3) - 0.60 fm
      a = 0.52 fm
      s = 0.90 fm
      R_n^2 = c^2 + (7/3) π^2 a^2 - 5 s^2

    and
      F(q) = 3 j1(q R_n)/(q R_n) * exp[-(q s)^2 / 2].

    This is a practical phenomenology default, not an ab-initio weak-charge form factor.
    """

    A: int
    s_fm: float = 0.9
    a_fm: float = 0.52
    c_fm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError("A must be positive.")

    def __call__(self, q2_gev2: float) -> float:
        if q2_gev2 <= 0.0:
            return 1.0

        q_gev = math.sqrt(q2_gev2)
        q_fm_inv = q_gev * FM_TO_GEV_INV

        if self.c_fm is None:
            c_fm = 1.23 * (self.A ** (1.0 / 3.0)) - 0.60
        else:
            c_fm = self.c_fm

        rn2 = c_fm ** 2 + (7.0 / 3.0) * (math.pi * self.a_fm) ** 2 - 5.0 * self.s_fm ** 2
        rn_fm = math.sqrt(max(rn2, 1e-12))
        x = q_fm_inv * rn_fm

        if abs(x) < 1e-8:
            j1_over_x = 1.0 / 3.0
        else:
            j1 = math.sin(x) / (x * x) - math.cos(x) / x
            j1_over_x = j1 / x

        return 3.0 * j1_over_x * math.exp(-0.5 * (q_fm_inv * self.s_fm) ** 2)


@dataclass(frozen=True)
class UnityFormFactor:
    def __call__(self, q2_gev2: float) -> float:
        return 1.0


# =========================
# Axial models
# =========================

@dataclass(frozen=True)
class GenericAxialFormFactor:
    """Exact structural decomposition if S_ij(q^2) are known.

    Implements
        F_A = 8π/(2J+1) [ (gAs)^2 S00 - gA gAs S01 + gA^2 S11 ]
    """

    J: float
    structures: AxialStructureFunctions
    gA: float = G_A_NUCLEON
    gAs: float = G_A_STRANGE_DEFAULT

    def __call__(self, q2_gev2: float) -> float:
        if self.J <= 0.0:
            return 0.0
        s00 = self.structures.s00(q2_gev2)
        s01 = self.structures.s01(q2_gev2)
        s11 = self.structures.s11(q2_gev2)
        return (8.0 * math.pi / (2.0 * self.J + 1.0)) * (
            (self.gAs ** 2) * s00 - self.gA * self.gAs * s01 + (self.gA ** 2) * s11
        )


@dataclass(frozen=True)
class SpinExpectationAxialApprox:
    """Fast approximate axial model.

    This approximation keeps the dominant isovector zero-momentum normalization and multiplies it by
    an ad-hoc dipole-like falloff. It is useful for feasibility studies and software development, but
    it should be replaced by shell-model / tabulated S_ij(q^2) for proposal-grade results.

    FA(0) ≈ gA^2 * (32π/3) * (J+1)/(J(2J+1)) * (Sp - Sn)^2
    FA(q^2) = FA(0) * f(q^2)

    with f(q^2) = (1 + q^2 / Lambda_A^2)^(-2 * power)
    """

    J: float
    Sp: float
    Sn: float
    gA: float = G_A_NUCLEON
    lambda_a_gev: float = 0.35
    power: float = 2.0

    def __call__(self, q2_gev2: float) -> float:
        if self.J <= 0.0:
            return 0.0
        delta_s = self.Sp - self.Sn
        fa0 = (self.gA ** 2) * (32.0 * math.pi / 3.0) * ((self.J + 1.0) / (self.J * (2.0 * self.J + 1.0))) * (delta_s ** 2)
        shape = (1.0 + q2_gev2 / (self.lambda_a_gev ** 2)) ** (-self.power)
        return fa0 * shape


@dataclass(frozen=True)
class ZeroAxialFormFactor:
    def __call__(self, q2_gev2: float) -> float:
        return 0.0


# =========================
# Targets and compounds
# =========================

@dataclass(frozen=True)
class NuclearTarget:
    name: str
    Z: int
    N: int
    mass_gev: float
    J: float
    vector_form_factor: VectorFormFactor
    axial_form_factor: Callable[[float], float] = field(default_factory=ZeroAxialFormFactor)
    metadata: Dict[str, float | str] = field(default_factory=dict)

    @property
    def A(self) -> int:
        return self.Z + self.N

    @property
    def weak_charge(self) -> float:
        return self.Z * (1.0 - 4.0 * SIN2_THETA_W) - self.N

    def q2_from_recoil_kev(self, recoil_kev: float) -> float:
        er_gev = kev_to_gev(recoil_kev)
        return 2.0 * self.mass_gev * er_gev

    def max_recoil_kev(self, neutrino_energy_mev: float) -> float:
        enu = mev_to_gev(neutrino_energy_mev)
        er_max = 2.0 * enu * enu / (self.mass_gev + 2.0 * enu)
        return gev_to_kev(er_max)

    def min_neutrino_energy_mev(self, recoil_kev: float) -> float:
        er = kev_to_gev(recoil_kev)
        enu_min = math.sqrt(self.mass_gev * er / 2.0)
        return enu_min * 1.0e3


@dataclass(frozen=True)
class StoichiometricMixture:
    name: str
    components: Mapping[NuclearTarget, int]

    def differential_xs_cm2_per_kev_per_molecule(
        self,
        neutrino_energy_mev: float,
        recoil_kev: float,
        model: "CEvNSCalculator",
    ) -> float:
        total = 0.0
        for target, multiplicity in self.components.items():
            total += multiplicity * model.differential_cross_section_cm2_per_kev(
                target=target,
                neutrino_energy_mev=neutrino_energy_mev,
                recoil_kev=recoil_kev,
            )
        return total


# =========================
# Main calculator
# =========================

@dataclass(frozen=True)
class CEvNSCalculator:
    """Standard-Model CEvNS calculator with explicit vector and axial terms."""

    def kinematic_factor_vector(self, target: NuclearTarget, neutrino_energy_mev: float, recoil_kev: float) -> float:
        enu = mev_to_gev(neutrino_energy_mev)
        er = kev_to_gev(recoil_kev)
        return 1.0 - target.mass_gev * er / (2.0 * enu * enu) - er / enu

    def kinematic_factor_axial(self, target: NuclearTarget, neutrino_energy_mev: float, recoil_kev: float) -> float:
        enu = mev_to_gev(neutrino_energy_mev)
        er = kev_to_gev(recoil_kev)
        return 1.0 + target.mass_gev * er / (2.0 * enu * enu) - er / enu

    def is_kinematically_allowed(self, target: NuclearTarget, neutrino_energy_mev: float, recoil_kev: float) -> bool:
        return 0.0 <= recoil_kev <= target.max_recoil_kev(neutrino_energy_mev)

    def vector_term(self, target: NuclearTarget, recoil_kev: float) -> float:
        q2 = target.q2_from_recoil_kev(recoil_kev)
        fw = target.vector_form_factor(q2)
        return (target.weak_charge ** 2) * (fw ** 2)

    def axial_term(self, target: NuclearTarget, recoil_kev: float) -> float:
        q2 = target.q2_from_recoil_kev(recoil_kev)
        return target.axial_form_factor(q2)

    def differential_cross_section_cm2_per_kev(
        self,
        target: NuclearTarget,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> float:
        if neutrino_energy_mev <= 0.0:
            return 0.0
        if recoil_kev < 0.0:
            return 0.0
        if not self.is_kinematically_allowed(target, neutrino_energy_mev, recoil_kev):
            return 0.0

        enu = mev_to_gev(neutrino_energy_mev)
        prefactor = (GF_GEV ** 2) * target.mass_gev / (4.0 * math.pi)

        vec = self.kinematic_factor_vector(target, neutrino_energy_mev, recoil_kev) * self.vector_term(target, recoil_kev)
        axi = self.kinematic_factor_axial(target, neutrino_energy_mev, recoil_kev) * self.axial_term(target, recoil_kev)

        dsig_dEr_gev = prefactor * (vec + axi)  # [GeV^-3] in natural units
        dsig_dEr_cm2_per_gev = dsig_dEr_gev * GEV2_TO_CM2
        dsig_dEr_cm2_per_kev = dsig_dEr_cm2_per_gev * 1.0e-6
        return max(dsig_dEr_cm2_per_kev, 0.0)

    def differential_vector_cross_section_cm2_per_kev(
        self,
        target: NuclearTarget,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> float:
        if not self.is_kinematically_allowed(target, neutrino_energy_mev, recoil_kev):
            return 0.0
        prefactor = (GF_GEV ** 2) * target.mass_gev / (4.0 * math.pi)
        value = prefactor * self.kinematic_factor_vector(target, neutrino_energy_mev, recoil_kev) * self.vector_term(target, recoil_kev)
        return max(value * GEV2_TO_CM2 * 1.0e-6, 0.0)

    def differential_axial_cross_section_cm2_per_kev(
        self,
        target: NuclearTarget,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> float:
        if not self.is_kinematically_allowed(target, neutrino_energy_mev, recoil_kev):
            return 0.0
        prefactor = (GF_GEV ** 2) * target.mass_gev / (4.0 * math.pi)
        value = prefactor * self.kinematic_factor_axial(target, neutrino_energy_mev, recoil_kev) * self.axial_term(target, recoil_kev)
        return max(value * GEV2_TO_CM2 * 1.0e-6, 0.0)


# =========================
# Built-in target factories
# =========================


def carbon12_target(*, use_helm: bool = True) -> NuclearTarget:
    ff: VectorFormFactor = HelmFormFactor(A=12) if use_helm else UnityFormFactor()
    return NuclearTarget(
        name="12C",
        Z=6,
        N=6,
        mass_gev=12.0 * AMU_TO_GEV,
        J=0.0,
        vector_form_factor=ff,
        axial_form_factor=ZeroAxialFormFactor(),
        metadata={"note": "Spin-zero built-in target; axial term set to zero."},
    )



def fluorine19_target(
    *,
    use_helm: bool = True,
    axial_model: str = "approx",
    Sp: float = 0.475,
    Sn: float = -0.009,
    lambda_a_gev: float = 0.35,
) -> NuclearTarget:
    ff: VectorFormFactor = HelmFormFactor(A=19) if use_helm else UnityFormFactor()

    if axial_model == "none":
        axial = ZeroAxialFormFactor()
        note = "Axial term disabled by request."
    elif axial_model == "approx":
        axial = SpinExpectationAxialApprox(
            J=0.5,
            Sp=Sp,
            Sn=Sn,
            lambda_a_gev=lambda_a_gev,
        )
        note = (
            "Approximate axial model from zero-momentum spin expectation values; "
            "replace with shell-model S_ij(q^2) for final studies."
        )
    else:
        raise ValueError(f"Unknown axial_model={axial_model!r}. Supported: 'approx', 'none'.")

    return NuclearTarget(
        name="19F",
        Z=9,
        N=10,
        mass_gev=19.0 * AMU_TO_GEV,
        J=0.5,
        vector_form_factor=ff,
        axial_form_factor=axial,
        metadata={
            "Sp": Sp,
            "Sn": Sn,
            "axial_model": axial_model,
            "note": note,
        },
    )



def cf4_molecule(
    *,
    carbon: Optional[NuclearTarget] = None,
    fluorine: Optional[NuclearTarget] = None,
) -> StoichiometricMixture:
    carbon = carbon or carbon12_target()
    fluorine = fluorine or fluorine19_target()
    return StoichiometricMixture(name="CF4", components={carbon: 1, fluorine: 4})


# =========================
# Helper for user-defined S_ij
# =========================

@dataclass(frozen=True)
class TabulatedAxialStructureFunctions:
    """Lightweight interpolation wrapper for user-supplied S_ij(q^2) samples.

    The input q2 values must be sorted and expressed in GeV^2.
    Linear interpolation is used between samples; constant extrapolation is used outside range.
    """

    q2_grid_gev2: tuple[float, ...]
    s00_grid: tuple[float, ...]
    s01_grid: tuple[float, ...]
    s11_grid: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.q2_grid_gev2)
        if n < 2:
            raise ValueError("Need at least two q^2 grid points.")
        if not (len(self.s00_grid) == len(self.s01_grid) == len(self.s11_grid) == n):
            raise ValueError("All grids must have the same length.")
        if any(self.q2_grid_gev2[i + 1] <= self.q2_grid_gev2[i] for i in range(n - 1)):
            raise ValueError("q^2 grid must be strictly increasing.")

    def _interp(self, x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    def s00(self, q2_gev2: float) -> float:
        return self._interp(q2_gev2, self.q2_grid_gev2, self.s00_grid)

    def s01(self, q2_gev2: float) -> float:
        return self._interp(q2_gev2, self.q2_grid_gev2, self.s01_grid)

    def s11(self, q2_gev2: float) -> float:
        return self._interp(q2_gev2, self.q2_grid_gev2, self.s11_grid)


# =========================
# CLI / demo
# =========================


def build_target_from_args(args: argparse.Namespace) -> NuclearTarget:
    if args.target.lower() in {"c", "12c", "carbon", "carbon12"}:
        return carbon12_target(use_helm=not args.pointlike)
    if args.target.lower() in {"f", "19f", "fluorine", "fluorine19"}:
        return fluorine19_target(
            use_helm=not args.pointlike,
            axial_model=args.axial_model,
            Sp=args.sp,
            Sn=args.sn,
            lambda_a_gev=args.lambda_a_gev,
        )
    raise ValueError(f"Unknown target: {args.target}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CEvNS differential cross sections with explicit vector and axial terms.")
    parser.add_argument("--target", default="19F", help="Target name: 12C or 19F")
    parser.add_argument("--enu-mev", type=float, required=True, help="Neutrino energy in MeV")
    parser.add_argument("--er-kev", type=float, required=True, help="Nuclear recoil energy in keV")
    parser.add_argument("--pointlike", action="store_true", help="Use F(q^2)=1 instead of Helm form factor")
    parser.add_argument("--axial-model", default="approx", choices=["approx", "none"], help="Axial model for 19F")
    parser.add_argument("--sp", type=float, default=0.475, help="<S_p> for the approximate 19F axial model")
    parser.add_argument("--sn", type=float, default=-0.009, help="<S_n> for the approximate 19F axial model")
    parser.add_argument("--lambda-a-gev", type=float, default=0.35, help="Axial dipole scale in GeV for approximate 19F axial model")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    calc = CEvNSCalculator()
    target = build_target_from_args(args)

    total = calc.differential_cross_section_cm2_per_kev(target, args.enu_mev, args.er_kev)
    vector = calc.differential_vector_cross_section_cm2_per_kev(target, args.enu_mev, args.er_kev)
    axial = calc.differential_axial_cross_section_cm2_per_kev(target, args.enu_mev, args.er_kev)
    q2 = target.q2_from_recoil_kev(args.er_kev)
    result = {
        "target": target.name,
        "enu_mev": args.enu_mev,
        "er_kev": args.er_kev,
        "q2_gev2": q2,
        "weak_charge": target.weak_charge,
        "vector_ff": target.vector_form_factor(q2),
        "axial_ff": target.axial_form_factor(q2),
        "dsig_dEr_total_cm2_per_kev": total,
        "dsig_dEr_vector_cm2_per_kev": vector,
        "dsig_dEr_axial_cm2_per_kev": axial,
        "metadata": target.metadata,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Target: {result['target']}")
    print(f"E_nu   : {result['enu_mev']:.6g} MeV")
    print(f"E_r    : {result['er_kev']:.6g} keV")
    print(f"q^2    : {result['q2_gev2']:.6e} GeV^2")
    print(f"Q_W    : {result['weak_charge']:.6f}")
    print(f"F_W    : {result['vector_ff']:.6e}")
    print(f"F_A    : {result['axial_ff']:.6e}")
    print(f"dσ/dEr total  = {result['dsig_dEr_total_cm2_per_kev']:.6e} cm^2/keV")
    print(f"dσ/dEr vector = {result['dsig_dEr_vector_cm2_per_kev']:.6e} cm^2/keV")
    print(f"dσ/dEr axial  = {result['dsig_dEr_axial_cm2_per_kev']:.6e} cm^2/keV")


if __name__ == "__main__":
    main()
