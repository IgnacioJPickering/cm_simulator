from setuptools import setup, find_packages
import sys
#print(find_packages())
with open('README.md','r') as descfile:
    long_description = descfile.read()
setup_attrs = {
        'name': 'pychmech',
        'version':'0.1.0',
        'description':'Simple molecular dynamics simulators for gases',
        'long_description':long_description,
        'long_desctiption_content_type':'text/markdown',
        'url':'https://github.com/IgnacioJPickering/cm_simulator/pycmech',
        'author':'Ignacio Pickering',
        'author_email':'ign.pickering@gmail.com',
        'license':'GNU',
        'packages':find_packages(where='.'),
        'include_package_data':True,
        'install_requires':['numpy','matplotlib']
        }
setup(**setup_attrs)
