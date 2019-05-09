# -*- coding: utf-8 -*-

"""Top-level package for Data Dependent Filters for FDK."""

__author__ = """Rien Lagerwerf"""
__email__ = 'm.j.lagerwerf@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from .CCB_CT_class import CCB_CT
from .phantom_class import phantom
from .real_data_class import real_data

# Import the support functions, maybe take a subset
from .image_measures import *
from .support_functions import *
from .full_data_prepare import *

