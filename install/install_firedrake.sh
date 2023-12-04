#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Mmg and ParMmg.                                             #
# ====================================================================== #

# *** YOU MAY WISH TO EDIT THESE ENVIRONMENT VARIABLES: ***

FIREDRAKE_ENV=firedrake-sep23
PETSC_BRANCH=jwallwork23/parmmg-rebased

# *** YOU SHOULD NOT NEED TO EDIT ANYTHING BELOW. ***

# Environment variables for Firedrake installation
CWD=$(pwd)
unset PYTHONPATH
unset PETSC_DIR
unset PETSC_ARCH
export PETSC_CONFIGURE_OPTIONS="$(cat petsc_options.txt | tr '\n' ' ')"
FIREDRAKE_DIR=${SOFTWARE}/${FIREDRAKE_ENV}

# Check environment variables
echo "FIREDRAKE_ENV=${FIREDRAKE_ENV}"
echo "FIREDRAKE_DIR=${FIREDRAKE_DIR}"
echo "PETSC_BRANCH=${PETSC_BRANCH}"
echo "PETSC_CONFIGURE_OPTIONS=${PETSC_CONFIGURE_OPTIONS}"
echo "python3=$(which python3)"
echo "Are these settings okay? Press any key to continue or Ctrl+C to exit."
read chk

# Install Firedrake using the above configuration
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --venv-name ${FIREDRAKE_ENV} --package-branch petsc ${PETSC_BRANCH}
source ${FIREDRAKE_DIR}/bin/activate
unset PETSC_CONFIGURE_OPTIONS

cd ${CWD}
