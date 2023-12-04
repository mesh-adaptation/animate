# Animate
## Anisotropic mesh adaptation toolkit for Firedrake
![GitHub top language](https://img.shields.io/github/languages/top/pyroteus/animate)
![GitHub repo size](https://img.shields.io/github/repo-size/pyroteus/animate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Animate is a mesh adaptation toolkit enabling users to anisotropically adapt a mesh based on a Riemannian metric framework with control over the shape, orientation and size of the resulting mesh elements. Animate is built for use with Firedrake and  strongly leverages PETSc DMPlex functionality. The implementation of metric-based mesh adaptation used in PETSc assumes that the metric is piece-wise linear and continuous, with its degrees of freedom at the mesh vertices.

For more information on Firedrake, please see: [Firedrake Website](https://www.firedrakeproject.org/).
For more information on PETSc, please see: [PETSc metric based mesh adaptation](https://petsc.org/release/docs/manual/dmplex/#metric-based-mesh-adaptation)

## Linux Installation

The following installation instructions assume a Linux or WSL operating system. The options below include installation from a custom shell script which also installs the custom setup for Firedrake and PETSc. If Firedrake is already installed, please see instructions to install Animate via `git clone`.

### To install Animate with all in one shell script

The 'all-in-one' shell script will install Animate and all its key dependencies, including Firedrake and PETSc.

Instructions:
- Download installation files either:
	-  manually from: or via `curl -O <filename>`
		- install/install_firedrake_custom_mpi.sh
		- /install/petsc_options.txt
	- via curl:
		- `curl -O https://raw.githubusercontent.com/pyroteus/animate/main/install/install_firedrake_custom_mpi.sh`
		- `curl -O https://raw.githubusercontent.com/pyroteus/animate/main/install/petsc_options.txt`
-  Install firedrake and associated dependencies to a local environment via `source install_firedrake_custom_mpi.sh`
- Install animate via `make animate`
- Execute the test suite to confirm installation was successful via `make test`

### To install Animate via Docker image

The instructions for installing the Animate docker image can be found in the `Docker` folder, including a makefile for installation and execution.

Alternatively, manually install the Animate docker image:
- Pull the docker image: `docker pull jwallwork/firedrake-parmmg`
- Run the docker image: `docker run --rm -it -v ${HOME}:${HOME} jwallwork/firedrake-parmmg`

Please note, by installing via a Docker image with `${HOME}` you are giving Docker access to your home space.


### To install Animate via `git clone`
Installing Animate via cloning the GitHub repository assumes prior installation of Firedrake and its dependencies. For separate instructions for installing Firedrake please See: [Firedrake download instructions](https://www.firedrakeproject.org/download.html).

To install Animate via `git clone`:
- from the main Animate GitHub, `git clone` the repository using HTTPS or SSH.
- navigate to the local cloned repository, activate your virtual environment and install via: `pip install .`.