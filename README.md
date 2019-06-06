
[![CodeFactor](https://www.codefactor.io/repository/github/ignaciojpickering/cm_simulator/badge)](https://www.codefactor.io/repository/github/ignaciojpickering/cm_simulator)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/IgnacioJPickering/cm_simulator.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IgnacioJPickering/cm_simulator/alerts/)

# pycmech

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

