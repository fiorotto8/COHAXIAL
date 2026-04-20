from cevens import CEvNSCalculator, carbon12_target, fluorine19_target

calc = CEvNSCalculator()
carbon = carbon12_target()
fluorine = fluorine19_target()

enu_mev = 30.0
recoils_kev = [1, 2, 5, 10, 20]

print(f"E_nu = {enu_mev} MeV")
print("Er_keV   dσ/dEr(12C) [cm^2/keV]   dσ/dEr(19F) [cm^2/keV]   axial_fraction_F")
for er in recoils_kev:
    ds_c = calc.differential_cross_section_cm2_per_kev(carbon, enu_mev, er)
    ds_f = calc.differential_cross_section_cm2_per_kev(fluorine, enu_mev, er)
    ds_f_ax = calc.differential_axial_cross_section_cm2_per_kev(fluorine, enu_mev, er)
    frac = ds_f_ax / ds_f if ds_f > 0 else 0.0
    print(f"{er:6.1f}   {ds_c: .6e}   {ds_f: .6e}   {frac: .4f}")
