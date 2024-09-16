#!/bin/bash
# This script purges the docs and environment

cd ..
/bin/rm -r docs/build
/bin/rm -r dist
/bin/rm -r jax_startup.egg-info
/bin/rm -r build

pip uninstall jax_startup

cd dev_scripts
