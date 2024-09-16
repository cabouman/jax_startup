#!/bin/bash
# This script purges the docs and rebuilds them

cd ../docs
/bin/rm -r build

make clean html

echo ""
echo "*** The html documentation is at jax_startup/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts
