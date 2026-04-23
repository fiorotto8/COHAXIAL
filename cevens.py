from __future__ import annotations

"""
Minimal, extensible scattering module for CEvNS and neutrino-electron elastic scattering.

Design goals
------------
- Clean separation between nuclear target data, vector form factor, axial form factor,
  and the cross-section calculator.
- Safe defaults for spin-zero nuclei (e.g. 12C => axial = 0).
- A generic exact interface for the axial structure functions S_00, S_01, S_11.
- Two literature-backed Hoferichter/Menendez/Schwenk 19F axial levels:
  a fast polynomial-fit transverse response and a central response including
  delta0/delta00 corrections.
- A deliberately simple toy axial model for debugging only.
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

For 19F, the built-in Hoferichter models follow arXiv:2007.08529 / PRD 102
(2020): Eq. (66) for the vector+axial CEvNS split, Eq. (67) for F_A,
Eqs. (80)-(85) for S00/S01/S11, Eq. (86) for delta0/delta00, Table I for
gA/gAs/rA, Table V for two-body-current central inputs, Table VIII for the
19F polynomial response coefficients and b, and Table IV only as a q2 -> 0
spin-expectation cross-check.

The module also includes Standard-Model neutrino-electron elastic scattering,
which is useful for gas targets such as CF4 where electron recoils can provide
an additional signal channel.
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
HOFERICHTER_G_A = 1.27641
HOFERICHTER_G_A_STRANGE_FAST = 0.0
HOFERICHTER_G_A_STRANGE_CENTRAL = -0.085
HOFERICHTER_AXIAL_RADIUS_SQ_FM2 = 0.46
HOFERICHTER_RHO_CENTRAL_FM3 = 0.10
HOFERICHTER_C1_GEV_INV = -1.20
HOFERICHTER_C3_GEV_INV = -4.45
HOFERICHTER_C4_GEV_INV = 2.96
HOFERICHTER_C6_GEV_INV = 5.01
HOFERICHTER_CD_CENTRAL = 0.5 * (-6.08 + 0.30)
HOFERICHTER_LAMBDA_CHI_GEV = 0.700
F_PI_GEV = 0.09228
M_PI_GEV = 0.13957039
M_NUCLEON_GEV = 0.938918754
G_PI_NN = math.sqrt(4.0 * math.pi * 13.7)
HBARC_GEV_CM = 1.973269804e-14  # [GeV cm]
GEV2_TO_CM2 = HBARC_GEV_CM ** 2
AMU_TO_GEV = 0.93149410242
FM_TO_GEV_INV = 5.067730716156395
M_ELECTRON_GEV = 0.510998950e-3


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
class AxialResponsePolynomial:
    """Polynomial shell-model response fit using the Appendix E convention."""

    coefficients: tuple[float, ...]

    def __call__(self, u: float) -> float:
        polynomial = sum(coef * (u ** i) for i, coef in enumerate(self.coefficients))
        return math.exp(-0.5 * u) * polynomial


class HoferichterDeltaCorrections(Protocol):
    def delta0(self, q2_gev2: float) -> float:
        ...

    def delta00(self, q2_gev2: float) -> float:
        ...


@dataclass(frozen=True)
class HoferichterCentralDeltaCorrections:
    """Central delta0/delta00 prescription for the detailed 19F model.

    Implements Hoferichter, Menendez, Schwenk Eq. (86), with the two-body
    current pieces delta a(q^2) and delta a^P(q^2) from Eqs. (75)-(76).
    Central inputs are from Table I (gA, g_A^{s,N}, axial radius,
    F_pi, g_piNN) and Table V (rho, c1, c3, c4, c6, cD). The contact
    cD is fixed to the midpoint of the Table V range for this single
    central-value option; uncertainty scans are left to a future layer.
    """

    rho_fm3: float = HOFERICHTER_RHO_CENTRAL_FM3
    c1_gev_inv: float = HOFERICHTER_C1_GEV_INV
    c3_gev_inv: float = HOFERICHTER_C3_GEV_INV
    c4_gev_inv: float = HOFERICHTER_C4_GEV_INV
    c6_gev_inv: float = HOFERICHTER_C6_GEV_INV
    cD: float = HOFERICHTER_CD_CENTRAL
    gA: float = HOFERICHTER_G_A
    axial_radius_sq_fm2: float = HOFERICHTER_AXIAL_RADIUS_SQ_FM2
    f_pi_gev: float = F_PI_GEV
    m_pi_gev: float = M_PI_GEV
    m_nucleon_gev: float = M_NUCLEON_GEV
    g_pi_nn: float = G_PI_NN
    lambda_chi_gev: float = HOFERICHTER_LAMBDA_CHI_GEV

    @property
    def rho_gev3(self) -> float:
        return self.rho_fm3 / (FM_TO_GEV_INV ** 3)

    @property
    def kf_gev(self) -> float:
        return ((3.0 * math.pi * math.pi * self.rho_gev3) / 2.0) ** (1.0 / 3.0)

    @property
    def axial_radius_sq_gev_inv2(self) -> float:
        return self.axial_radius_sq_fm2 * (FM_TO_GEV_INV ** 2)

    def _arccot(self, x: float) -> float:
        return math.atan2(1.0, x)

    def _log_ratio(self, q_gev: float) -> float:
        kf = self.kf_gev
        mpi = self.m_pi_gev
        numerator = mpi * mpi + (kf - 0.5 * q_gev) ** 2
        denominator = mpi * mpi + (kf + 0.5 * q_gev) ** 2
        return math.log(numerator / denominator)

    def _fermi_gas_integrals(self, q_gev: float) -> tuple[float, float, float, float]:
        """Klos/Menendez/Gazit/Schwenk exchange integrals used in Eqs. (75)-(76).

        The analytic expressions are quoted in Appendix B of Klos et al.,
        Phys. Rev. D 88, 083516 (2013), which is the formalism referenced by
        Hoferichter et al. around Eqs. (75)-(76). All momenta are in GeV, so
        the integrals are dimensionless.
        """

        kf = self.kf_gev
        mpi = self.m_pi_gev
        if abs(q_gev) < 1e-4:
            common = 1.0 - 3.0 * (mpi ** 2) / (kf ** 2) + 3.0 * (mpi ** 3) / (kf ** 3) * math.atan(kf / mpi)
            return common, common, 0.0, 0.0

        q = q_gev
        q2 = q * q
        k2 = kf * kf
        m2 = mpi * mpi
        acot = self._arccot((m2 + 0.25 * q2 - k2) / (2.0 * mpi * kf))
        log_ratio = self._log_ratio(q)

        i_sigma1 = (
            8.0 * kf * q * (48.0 * (k2 + m2) ** 2 + 32.0 * (k2 - 3.0 * m2) * q2 - 3.0 * q2 * q2)
            + 768.0 * (mpi ** 3) * (q ** 3) * acot
            + 3.0
            * (16.0 * (k2 + m2) ** 2 - 8.0 * (k2 - 5.0 * m2) * q2 + q2 * q2)
            * (4.0 * (k2 + m2) - q2)
            * log_ratio
        ) / (512.0 * (kf ** 3) * (q ** 3))

        i_sigma2 = (
            8.0 * kf * (2.0 * k2 - 3.0 * m2) * q
            + 24.0 * (mpi ** 3) * q * acot
            + 3.0 * m2 * (4.0 * k2 - q2 + 4.0 * m2) * log_ratio
        ) / (16.0 * (kf ** 3) * q)

        i_p = -(
            8.0 * kf * q * (48.0 * (k2 + m2) ** 2 - 32.0 * k2 * q2 - 3.0 * q2 * q2)
            + 3.0
            * (4.0 * (k2 + m2) - q2)
            * (4.0 * m2 + (2.0 * kf - q) ** 2)
            * (4.0 * m2 + (2.0 * kf + q) ** 2)
            * log_ratio
        ) * 3.0 / (512.0 * (kf ** 3) * (q ** 3))

        i_c6 = -(
            32.0 * (kf ** 3) * q
            + 32.0 * kf * m2 * q
            + 8.0 * kf * (q ** 3)
            + (16.0 * (k2 + m2) ** 2 + 8.0 * (m2 - k2) * q2 + q2 * q2)
            * math.log((4.0 * m2 + (2.0 * kf - q) ** 2) / (4.0 * m2 + (2.0 * kf + q) ** 2))
        ) * 9.0 / (128.0 * (kf ** 3) * q)

        return i_sigma1, i_sigma2, i_p, i_c6

    def two_body_delta_a(self, q2_gev2: float) -> float:
        q2 = max(q2_gev2, 0.0)
        q = math.sqrt(q2)
        i1, i2, _ip, ic6 = self._fermi_gas_integrals(q)
        bracket = (
            (self.c4_gev_inv / 3.0) * (3.0 * i2 - i1)
            - ((self.c3_gev_inv - 1.0 / (4.0 * self.m_nucleon_gev)) / 3.0) * i1
            - (self.c6_gev_inv / 12.0) * ic6
            - self.cD / (4.0 * self.gA * self.lambda_chi_gev)
        )
        return -(self.rho_gev3 / (self.f_pi_gev ** 2)) * bracket

    def two_body_delta_a_p(self, q2_gev2: float) -> float:
        q2 = max(q2_gev2, 0.0)
        q = math.sqrt(q2)
        i1, i2, ip, ic6 = self._fermi_gas_integrals(q)
        pion_den = self.m_pi_gev ** 2 + q2
        bracket = (
            -2.0 * (self.c3_gev_inv - 2.0 * self.c1_gev_inv) * (self.m_pi_gev ** 2) * q2 / (pion_den ** 2)
            + ((self.c3_gev_inv + self.c4_gev_inv - 1.0 / (4.0 * self.m_nucleon_gev)) / 3.0) * ip
            - (self.c6_gev_inv / 12.0 - (2.0 / 3.0) * self.c1_gev_inv * (self.m_pi_gev ** 2) / pion_den) * ic6
            - (q2 / pion_den)
            * (
                (self.c3_gev_inv / 3.0) * (i1 + ip)
                + (self.c4_gev_inv / 3.0) * (i1 + ip - 3.0 * i2)
            )
            - self.cD / (4.0 * self.gA * self.lambda_chi_gev) * q2 / pion_den
        )
        return (self.rho_gev3 / (self.f_pi_gev ** 2)) * bracket

    def delta0(self, q2_gev2: float) -> float:
        q2 = max(q2_gev2, 0.0)
        return -q2 * self.axial_radius_sq_gev_inv2 / 6.0 + self.two_body_delta_a(q2)

    def delta00(self, q2_gev2: float) -> float:
        q2 = max(q2_gev2, 0.0)
        pion_pole = -(
            self.g_pi_nn * self.f_pi_gev / (self.gA * self.m_nucleon_gev)
        ) * q2 / (q2 + self.m_pi_gev ** 2)
        return pion_pole + self.two_body_delta_a(q2) + self.two_body_delta_a_p(q2)


@dataclass(frozen=True)
class Hoferichter19FTransverseStructureFunctions:
    """19F transverse axial structure functions from Hoferichter et al.

    Uses Appendix E / Table VIII coefficients from Hoferichter, Menendez,
    Schwenk, "Coherent elastic neutrino-nucleus scattering: EFT analysis and
    nuclear responses", arXiv:2007.08529 / PRD 102 (2020).

    Eqs. (80)-(85) form S00/S01/S11 from proton/neutron transverse responses.
    The fast model sets delta0=delta00=0; the central model passes explicit
    Eq. (86) corrections through the corrections object.
    With gAs=0 and no delta corrections, the polynomial-fit coefficients give
    FA(0) around 2.25 for the Eq. (67) normalization. This is the same
    q2 -> 0 logic as the Table IV spin-expectation cross-check, but the
    polynomial response is the model input used here.
    """

    b_fm: float = 1.7623
    corrections: Optional[HoferichterDeltaCorrections] = None
    sigma_prime_p: AxialResponsePolynomial = field(
        default_factory=lambda: AxialResponsePolynomial((0.269513, -0.18098, 0.0296873))
    )
    sigma_prime_n: AxialResponsePolynomial = field(
        default_factory=lambda: AxialResponsePolynomial((-0.00113172, 0.00038188, 0.000744991))
    )
    sigma_double_prime_p: AxialResponsePolynomial = field(
        default_factory=lambda: AxialResponsePolynomial((0.190574, -0.125204, 0.0206132))
    )
    sigma_double_prime_n: AxialResponsePolynomial = field(
        default_factory=lambda: AxialResponsePolynomial((-0.000800244, 0.00106046, -0.000167277))
    )

    def u(self, q2_gev2: float) -> float:
        if q2_gev2 <= 0.0:
            return 0.0
        q_gev = math.sqrt(q2_gev2)
        q_fm_inv = q_gev * FM_TO_GEV_INV
        return 0.5 * (q_fm_inv ** 2) * (self.b_fm ** 2)

    def _structure_factors(self, q2_gev2: float) -> tuple[float, float, float]:
        u = self.u(q2_gev2)

        sigma_prime_p = self.sigma_prime_p(u)
        sigma_prime_n = self.sigma_prime_n(u)
        sigma_double_prime_p = self.sigma_double_prime_p(u)
        sigma_double_prime_n = self.sigma_double_prime_n(u)

        # Eq. (85): isoscalar/isovector combinations F^+ = F^p + F^n,
        # F^- = F^p - F^n. For 19F, the Table VIII sum over L has only L=1.
        sigma_prime_plus = sigma_prime_p + sigma_prime_n
        sigma_prime_minus = sigma_prime_p - sigma_prime_n
        sigma_double_prime_plus = sigma_double_prime_p + sigma_double_prime_n
        sigma_double_prime_minus = sigma_double_prime_p - sigma_double_prime_n

        if self.corrections is None:
            delta0 = 0.0
            delta00 = 0.0
        else:
            delta0 = self.corrections.delta0(q2_gev2)
            delta00 = self.corrections.delta00(q2_gev2)

        s00 = sigma_prime_plus ** 2 + sigma_double_prime_plus ** 2
        s11 = ((1.0 + delta0) * sigma_prime_minus) ** 2 + ((1.0 + delta00) * sigma_double_prime_minus) ** 2
        s01 = (
            2.0 * (1.0 + delta0) * sigma_prime_plus * sigma_prime_minus
            + 2.0 * (1.0 + delta00) * sigma_double_prime_plus * sigma_double_prime_minus
        )
        return s00, s01, s11

    def s00(self, q2_gev2: float) -> float:
        return self._structure_factors(q2_gev2)[0]

    def s01(self, q2_gev2: float) -> float:
        return self._structure_factors(q2_gev2)[1]

    def s11(self, q2_gev2: float) -> float:
        return self._structure_factors(q2_gev2)[2]


@dataclass(frozen=True)
class Hoferichter19FFastAxial:
    """Fast default 19F axial form factor from Hoferichter et al.

    Level 1 model: Table VIII polynomial-fit transverse responses, gAs=0, and
    delta0=delta00=0. This is the default because it is fast and uses the
    literature-backed shell-model response without the extra correction inputs.
    """

    structures: Hoferichter19FTransverseStructureFunctions = field(
        default_factory=Hoferichter19FTransverseStructureFunctions
    )
    gA: float = HOFERICHTER_G_A
    gAs: float = HOFERICHTER_G_A_STRANGE_FAST

    def __post_init__(self) -> None:
        self._check_positive_normalization("fast")

    def __call__(self, q2_gev2: float) -> float:
        value = GenericAxialFormFactor(J=0.5, structures=self.structures, gA=self.gA, gAs=self.gAs)(q2_gev2)
        # Physical q2 values should give a finite positive default 19F axial response.
        if not math.isfinite(value):
            raise ValueError("Hoferichter fast 19F axial form factor is not finite.")
        return value

    def _check_positive_normalization(self, label: str) -> None:
        for q2_gev2 in (0.0, 0.01):
            value = GenericAxialFormFactor(J=0.5, structures=self.structures, gA=self.gA, gAs=self.gAs)(q2_gev2)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Hoferichter {label} 19F axial form factor must be finite and positive.")


@dataclass(frozen=True)
class Hoferichter19FCentralAxial(Hoferichter19FFastAxial):
    """Detailed central 19F axial form factor from Hoferichter et al.

    Level 2 model: the same Table VIII 19F polynomial-fit transverse responses,
    Table I gA/gAs/axial-radius central inputs, Table V rho/c_i/cD central
    inputs, and Eq. (86) delta0(q^2), delta00(q^2) corrections.
    """

    structures: Hoferichter19FTransverseStructureFunctions = field(
        default_factory=lambda: Hoferichter19FTransverseStructureFunctions(
            corrections=HoferichterCentralDeltaCorrections()
        )
    )
    gA: float = HOFERICHTER_G_A
    gAs: float = HOFERICHTER_G_A_STRANGE_CENTRAL

    def __post_init__(self) -> None:
        self._check_positive_normalization("central")

    def __call__(self, q2_gev2: float) -> float:
        value = GenericAxialFormFactor(J=0.5, structures=self.structures, gA=self.gA, gAs=self.gAs)(q2_gev2)
        if not math.isfinite(value):
            raise ValueError("Hoferichter central 19F axial form factor is not finite.")
        return value


Hoferichter19FTabulatedAxial = Hoferichter19FFastAxial


@dataclass(frozen=True)
class SpinExpectationAxialToyModel:
    """Toy/debug axial model, not a default physics model.

    This deliberately simple model keeps a zero-momentum spin-expectation
    normalization and multiplies it by a phenomenological dipole-like falloff.
    It is retained only for debugging/benchmarking and is not a literature-grade
    19F CEvNS nuclear-response model.

    FA(0) ≈ gA^2 * (32π/3) * (J+1)/(J(2J+1)) * (Sp - Sn)^2
    FA(q^2) = FA(0) * (1 + q^2 / Lambda_A^2)^(-exponent)
    """

    J: float
    Sp: float
    Sn: float
    gA: float = G_A_NUCLEON
    lambda_a_gev: float = 0.35
    exponent: float = 2.0

    def __call__(self, q2_gev2: float) -> float:
        if self.J <= 0.0:
            return 0.0
        delta_s = self.Sp - self.Sn
        fa0 = (self.gA ** 2) * (32.0 * math.pi / 3.0) * ((self.J + 1.0) / (self.J * (2.0 * self.J + 1.0))) * (delta_s ** 2)
        shape = (1.0 + q2_gev2 / (self.lambda_a_gev ** 2)) ** (-self.exponent)
        return fa0 * shape


SpinExpectationAxialApprox = SpinExpectationAxialToyModel


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
    metadata: Dict[str, float | str] = field(default_factory=dict, compare=False, hash=False)

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

    @property
    def electrons_per_molecule(self) -> int:
        return sum(multiplicity * target.Z for target, multiplicity in self.components.items())

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


@dataclass(frozen=True)
class ElectronTarget:
    name: str = "electron"
    electrons_per_molecule: int = 1
    metadata: Dict[str, float | str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.electrons_per_molecule <= 0:
            raise ValueError("electrons_per_molecule must be positive.")

    @property
    def mass_gev(self) -> float:
        return M_ELECTRON_GEV

    def max_recoil_kev(self, neutrino_energy_mev: float) -> float:
        enu = mev_to_gev(neutrino_energy_mev)
        recoil_max_gev = 2.0 * enu * enu / (self.mass_gev + 2.0 * enu)
        return gev_to_kev(recoil_max_gev)

    def min_neutrino_energy_mev(self, recoil_kev: float) -> float:
        recoil_gev = kev_to_gev(recoil_kev)
        enu_min_gev = 0.5 * (
            recoil_gev + math.sqrt(recoil_gev * recoil_gev + 2.0 * self.mass_gev * recoil_gev)
        )
        return 1.0e3 * enu_min_gev


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


def canonical_neutrino_flavor(flavor: str) -> tuple[str, bool]:
    key = flavor.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "nue": ("nue", False),
        "electronneutrino": ("nue", False),
        "electronnu": ("nue", False),
        "numu": ("numu", False),
        "muonneutrino": ("numu", False),
        "muonnu": ("numu", False),
        "nutau": ("nutau", False),
        "tauneutrino": ("nutau", False),
        "taunu": ("nutau", False),
        "nuebar": ("nue", True),
        "antinue": ("nue", True),
        "electronantineutrino": ("nue", True),
        "antielectronneutrino": ("nue", True),
        "numubar": ("numu", True),
        "antinumu": ("numu", True),
        "muonantineutrino": ("numu", True),
        "antimuonneutrino": ("numu", True),
        "nutaubar": ("nutau", True),
        "antinutau": ("nutau", True),
        "tauantineutrino": ("nutau", True),
        "antitauneutrino": ("nutau", True),
    }

    if key not in aliases:
        raise ValueError(
            f"Unknown neutrino flavor {flavor!r}. Supported examples: "
            "'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'."
        )
    return aliases[key]


@dataclass(frozen=True)
class NeutrinoElectronCalculator:
    """Standard-Model neutrino-electron elastic scattering calculator."""

    electron_target: ElectronTarget = field(default_factory=ElectronTarget)

    def chiral_couplings(self, flavor: str) -> tuple[float, float, bool]:
        base_flavor, is_antineutrino = canonical_neutrino_flavor(flavor)

        g_left = -0.5 + SIN2_THETA_W
        g_right = SIN2_THETA_W

        if base_flavor == "nue":
            # Add the charged-current contribution for electron flavor.
            g_left += 1.0

        return g_left, g_right, is_antineutrino

    def is_kinematically_allowed(
        self,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> bool:
        return 0.0 <= recoil_kev <= self.electron_target.max_recoil_kev(neutrino_energy_mev)

    def differential_cross_section_cm2_per_kev(
        self,
        flavor: str,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> float:
        if neutrino_energy_mev <= 0.0:
            return 0.0
        if recoil_kev < 0.0:
            return 0.0
        if not self.is_kinematically_allowed(neutrino_energy_mev, recoil_kev):
            return 0.0

        enu = mev_to_gev(neutrino_energy_mev)
        recoil = kev_to_gev(recoil_kev)
        y = recoil / enu

        g_left, g_right, is_antineutrino = self.chiral_couplings(flavor)
        leading = g_right if is_antineutrino else g_left
        subleading = g_left if is_antineutrino else g_right

        prefactor = 2.0 * (GF_GEV ** 2) * self.electron_target.mass_gev / math.pi
        kinematic = (
            leading * leading
            + subleading * subleading * (1.0 - y) ** 2
            - g_left * g_right * self.electron_target.mass_gev * recoil / (enu * enu)
        )

        dsig_dT_gev = prefactor * kinematic
        dsig_dT_cm2_per_gev = dsig_dT_gev * GEV2_TO_CM2
        dsig_dT_cm2_per_kev = dsig_dT_cm2_per_gev * 1.0e-6
        return max(dsig_dT_cm2_per_kev, 0.0)

    def differential_cross_section_cm2_per_kev_per_molecule(
        self,
        flavor: str,
        neutrino_energy_mev: float,
        recoil_kev: float,
    ) -> float:
        return (
            self.electron_target.electrons_per_molecule
            * self.differential_cross_section_cm2_per_kev(flavor, neutrino_energy_mev, recoil_kev)
        )


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
    axial_model: str = "hoferichter_19f_fast",
    Sp: float = 0.475,
    Sn: float = -0.009,
    lambda_a_gev: float = 0.35,
    axial_form_factor: Optional[Callable[[float], float]] = None,
) -> NuclearTarget:
    ff: VectorFormFactor = HelmFormFactor(A=19) if use_helm else UnityFormFactor()
    requested_axial_model = axial_model
    axial_model_key = axial_model.strip().lower().replace("-", "_")

    if axial_model_key == "none":
        axial = ZeroAxialFormFactor()
        axial_model = "none"
        note = "Axial term disabled by request."
    elif axial_model_key in {"hoferichter_19f_fast", "hoferichter_19f", "hoferichter19f", "hms_19f"}:
        axial = Hoferichter19FFastAxial()
        axial_model = "hoferichter_19f_fast"
        note = (
            "Fast literature-backed 19F axial model using Hoferichter, Menendez, "
            "Schwenk Appendix E Table VIII transverse shell-model polynomial fits; "
            "gAs=0 and delta0/delta00 are set to zero."
        )
    elif axial_model_key in {"hoferichter_19f_central", "central", "hms_19f_central"}:
        axial = Hoferichter19FCentralAxial()
        axial_model = "hoferichter_19f_central"
        note = (
            "Central Hoferichter 19F axial model using Table VIII polynomial fits, "
            "Table I gA/gAs/axial-radius inputs, Table V rho/c_i/cD inputs, and "
            "Eq. (86) delta0/delta00 corrections."
        )
    elif axial_model_key == "approx":
        axial = Hoferichter19FFastAxial()
        axial_model = "hoferichter_19f_fast"
        note = (
            "Legacy axial_model='approx' request mapped to hoferichter_19f_fast. "
            "Use axial_model='toy' only for the old simplified debug/testing model."
        )
    elif axial_model_key in {"toy", "toy_constant_fa0"}:
        axial = SpinExpectationAxialToyModel(
            J=0.5,
            Sp=Sp,
            Sn=Sn,
            lambda_a_gev=lambda_a_gev,
        )
        axial_model = "toy"
        note = (
            "Toy/debug axial model based on zero-momentum spin expectation values "
            "with dipole-like suppression; not a literature-grade 19F CEvNS "
            "nuclear-response model."
        )
    elif axial_model_key == "tabulated":
        if axial_form_factor is None:
            raise ValueError("axial_model='tabulated' requires axial_form_factor.")
        axial = axial_form_factor
        axial_model = "tabulated"
        note = (
            "User-supplied axial form factor; use GenericAxialFormFactor with "
            "tabulated S_ij(q^2) for exact 19F structure inputs."
        )
    else:
        raise ValueError(
            f"Unknown axial_model={requested_axial_model!r}. Supported: "
            "'hoferichter_19f_fast', 'hoferichter_19f_central', 'none', 'toy', "
            "'tabulated'. Legacy aliases: 'hoferichter_19f' and 'approx' map to "
            "'hoferichter_19f_fast'."
        )

    metadata: Dict[str, float | str] = {
        "axial_model": axial_model,
        "requested_axial_model": requested_axial_model,
        "note": note,
    }
    if axial_model in {"hoferichter_19f_fast", "hoferichter_19f_central"}:
        metadata.update({
            "reference": "Hoferichter, Menendez, Schwenk, PRD 102 (2020), arXiv:2007.08529",
            "coefficient_table": "Appendix E, Table VIII",
            "response_type": "transverse shell-model polynomial fits F^{Sigma_0} / F^{Sigma_{00}}, L=1",
            "b_fm": axial.structures.b_fm,
            "J": 0.5,
            "gA": axial.gA,
            "gAs": axial.gAs,
        })
    if axial_model == "hoferichter_19f_fast":
        metadata["delta_corrections"] = "delta0=0, delta00=0"
        if axial_model_key == "approx":
            metadata["warning"] = "Legacy 'approx' now maps to 'hoferichter_19f_fast'; use 'toy' for the old debug model."
    if axial_model == "hoferichter_19f_central":
        corrections = axial.structures.corrections
        if isinstance(corrections, HoferichterCentralDeltaCorrections):
            metadata.update({
                "delta_corrections": "delta0(q^2), delta00(q^2) included from Eq. (86)",
                "rho_fm3": corrections.rho_fm3,
                "c1_gev_inv": corrections.c1_gev_inv,
                "c3_gev_inv": corrections.c3_gev_inv,
                "c4_gev_inv": corrections.c4_gev_inv,
                "c6_gev_inv": corrections.c6_gev_inv,
                "cD": corrections.cD,
                "axial_radius_sq_fm2": corrections.axial_radius_sq_fm2,
                "f_pi_gev": corrections.f_pi_gev,
                "m_pi_gev": corrections.m_pi_gev,
                "g_pi_nn": corrections.g_pi_nn,
            })
    if axial_model == "toy":
        metadata.update({
            "Sp": Sp,
            "Sn": Sn,
            "lambda_a_gev": lambda_a_gev,
            "axial_exponent": axial.exponent,
            "warning": "Toy/debug only; not proposal-grade 19F axial physics.",
        })

    return NuclearTarget(
        name="19F",
        Z=9,
        N=10,
        mass_gev=19.0 * AMU_TO_GEV,
        J=0.5,
        vector_form_factor=ff,
        axial_form_factor=axial,
        metadata=metadata,
    )



def cf4_molecule(
    *,
    carbon: Optional[NuclearTarget] = None,
    fluorine: Optional[NuclearTarget] = None,
) -> StoichiometricMixture:
    carbon = carbon or carbon12_target()
    fluorine = fluorine or fluorine19_target()
    return StoichiometricMixture(name="CF4", components={carbon: 1, fluorine: 4})


def electron_target_for_mixture(mixture: StoichiometricMixture) -> ElectronTarget:
    return ElectronTarget(
        name=f"{mixture.name} electrons",
        electrons_per_molecule=mixture.electrons_per_molecule,
        metadata={
            "parent_mixture": mixture.name,
            "electrons_per_molecule": mixture.electrons_per_molecule,
        },
    )


def cf4_electron_target(
    *,
    carbon: Optional[NuclearTarget] = None,
    fluorine: Optional[NuclearTarget] = None,
) -> ElectronTarget:
    return electron_target_for_mixture(cf4_molecule(carbon=carbon, fluorine=fluorine))


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
    parser.add_argument(
        "--axial-model",
        default="hoferichter_19f_fast",
        choices=["hoferichter_19f_fast", "hoferichter_19f_central", "none", "toy"],
        help="Axial model for 19F: fast Hoferichter polynomial fit, central Hoferichter corrections, none, or toy/debug",
    )
    parser.add_argument("--sp", type=float, default=0.475, help="<S_p> for the toy/debug 19F axial model")
    parser.add_argument("--sn", type=float, default=-0.009, help="<S_n> for the toy/debug 19F axial model")
    parser.add_argument("--lambda-a-gev", type=float, default=0.35, help="Axial dipole scale in GeV for toy/debug 19F axial model")
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
