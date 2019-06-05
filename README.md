#pycmech

A super simple implementation of molecular dynamics of gases on python.
The Verlet algorithm is used for propagation of coordinates.
Internally the units used are:

Length: angstrom
Time: femtosecond
Mass: grams per mol
Charge: proton charge
Temperature: kelvin

Derived units are calculated from these units, for instance energy is actually
in units of (g/mol)x(angstrom/femtosecond)

