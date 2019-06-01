import sys
import numpy as np
#from ..pycmech import pycmech
input_path = './test_data/some_atoms.xyz'
chem_list = ['X','H','He','Li','Be','B','C','N','O','F','Ne','Na',
        'Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V',
        'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
        'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag',
        'Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
        'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po',
        'At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']
chem_dict = {key:value for value,key in enumerate(chem_list)}

def parse_elinput(elinput,chem_dict):
    znumbers = []
    for el in elinput:
        try:
            znum = int(el)
        except ValueError:
            znumbers.append(chem_dict[el.lower().capitalize()])
        else:
            znumbers.append(znum)
    return np.array(znumbers)

def read_coordinates_from_xyz(input_path,chem_dict):
    coordinates = []
    elinput = []
    with open(input_path) as myfile:
        for num, line in enumerate(myfile):
            if num == 0:
                num_atoms = line
            if num == 1: 
                file_comment = line
            if num > 1:
                values = line.split(' ')
                elinput.append(values[0])
                coordinates.append([float(x) for x in values[1:]])
        coordinates = np.array(coordinates)
        znumbers = parse_elinput(elinput,chem_dict)
        return coordinates, znumbers
def znumbers_to_symbols(znumbers,chem_dict):
    symbols = []
    for num in znumbers:
        symbols.append(
                list(chem_dict.keys())[list(chem_dict.values()).index(num)])
    return np.array(symbols)

coordinates, znumbers = read_coordinates_from_xyz(input_path,chem_dict)
symbols = znumbers_to_symbols(znumbers,chem_dict)
print(coordinates)
print(znumbers)
print(symbols)
