#!/bin/bash

# ====================================================================== #
# Bash script for installing Firedrake based on a PETSc installation     #
# which uses Mmg and ParMmg.                                             #
#                                                                        #
# Note that we use a custom PETSc branch.                                #
# ====================================================================== #

# Unset PYTHONPATH and PETSc env variables
PYTHONPATH_TMP=${PYTHONPATH}
unset PYTHONPATH
unset PETSC_DIR
unset PETSC_ARCH

# Environment variables for MPI
MPICC=/usr/bin/mpicc.mpich
MPICXX=/usr/bin/mpicxx.mpich
MPIEXEC=/usr/bin/mpiexec.mpich
MPIF90=/usr/bin/mpif90.mpich
for mpi in ${MPICC} ${MPICXX} ${MPIEXEC} ${MPIF90}
do
	if [ ! -f ${mpi} ]
	then
		echo "Cannot find ${mpi} in /usr/bin."
		exit 1
	fi
done

# Environment variables for Firedrake installation
FIREDRAKE_ENV=firedrake-adapt
FIREDRAKE_DIR=${SOFTWARE}/${FIREDRAKE_ENV}
PETSC_BRANCH=jwallwork23/parmmg-rebased 
export PETSC_CONFIGURE_OPTIONS="$(cat petsc_options.txt | tr '\n' ' ')"
export PETSC_CONFIGURE_OPTIONS="${PETSC_CONFIGURE_OPTIONS} --with-mpiexec=${MPIEXEC} --CC=${MPICC} --CXX=${MPICXX} --FC=${MPIF90}"

# Check environment variables
echo "MPICC=${MPICC}"
echo "MPICXX=${MPICXX}"
echo "MPIF90=${MPIF90}"
echo "MPIEXEC=${MPIEXEC}"
echo "FIREDRAKE_ENV=${FIREDRAKE_ENV}"
echo "FIREDRAKE_DIR=${FIREDRAKE_DIR}"
echo "PETSC_BRANCH=${PETSC_BRANCH}"
echo "PETSC_CONFIGURE_OPTIONS=${PETSC_CONFIGURE_OPTIONS}"
echo "python3=$(which python3)"
echo "Are these settings okay? Press enter to continue."
read chk

# Install Firedrake
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --venv-name ${FIREDRAKE_ENV} --package-branch petsc ${PETSC_BRANCH} \
    --mpicc ${MPICC} --mpicxx ${MPICXX} --mpif90 ${MPIF90} --mpiexec ${MPIEXEC}
source ${FIREDRAKE_DIR}/bin/activate

# Reset environment
export PYTHONPATH=${PYTHONPATH_TMP}
unset PYTHONPATH_TMP
unset PETSC_CONFIGURE_OPTIONS

# Install Animate and test that the adaptation functionality is working
cd ${FIREDRAKE_DIR}/src
git clone git@github.com:pyroteus/animate
cd animate
make install
make test
