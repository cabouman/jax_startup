from setuptools import setup, find_packages, Extension
import numpy as np
import os

NAME = "jax_startup"
VERSION = "0.1"
DESCR = "helps get started with JAX"
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'jax_startup development team'
EMAIL = "buzzard@purdue.edu"
PACKAGE_DIR = "jax_startup"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['jax_startup']),
      )

