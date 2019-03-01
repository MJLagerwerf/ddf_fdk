#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

with open(os.path.join('ddf_fdk','VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
	'astra-toolbox',
	'odl',
	'Cython',
	'pyfftw',
	'scikit-image',
	'tabulate',
	'matplotlib',
	'porespy'
    # Add your project's requirements here, e.g.,
    # 'tables==3.4.4',

]

setup_requirements = [ ]

test_requirements = [ ]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    
    ]

ext_modules=[
    Extension("ddf_fdk.phantom_objects",
              sources=["ddf_fdk/phantom_objects.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native",
                                    "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )]


setup(
    author="Rien Lagerwerf",
    author_email='m.j.lagerwerf@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Supporting code for the ['Improving FDK reconstruction by Data-Dependent Filtering'] paper",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ddf_fdk',
    name='ddf_fdk',
    packages=find_packages(include=['ddf_fdk']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements },
    url='https://github.com/mjlagerwerf/ddf_fdk',
    version=version,
    zip_safe=False,
    cmdclass = {"build_ext": build_ext},
    ext_modules = cythonize(ext_modules, force=True)
