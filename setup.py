from setuptools import setup, find_packages
import sys
#print(find_packages())
setup_attrs = {
        'name': 'pychmech',
        'version':'0.1.0',
        'description':'Simple molecular dynamics simulators for gases',
        'url':'https://github.com/IgnacioJPickering/cm_simulator/pycmech',
        'author':'Ignacio Pickering',
        'author_email':'ign.pickering@gmail.com',
        'license':'GNU',
        'packages':find_packages(where='.'),
        'include_package_data':True,
        'install_requires':['numpy','matplotlib']
        }
