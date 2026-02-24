"""Setup script for building Cython extensions for the `animate` package.

The script performs PETSc detection and populates Extension objects based on the local
PETSc installation (via petsctools/petsc4py).
"""

import os
from dataclasses import dataclass, field

import numpy as np
import petsc4py
import petsctools
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


@dataclass
class ExternalDependency:
    """Courtesy of Firedrake. See
    https://github.com/firedrakeproject/firedrake/blob/main/setup.py
    """

    include_dirs: list[str] = field(default_factory=list, init=True)
    extra_compile_args: list[str] = field(default_factory=list, init=True)
    libraries: list[str] = field(default_factory=list, init=True)
    library_dirs: list[str] = field(default_factory=list, init=True)
    extra_link_args: list[str] = field(default_factory=list, init=True)
    runtime_library_dirs: list[str] = field(default_factory=list, init=True)

    def __add__(self, other):
        combined = {}
        for f in self.__dataclass_fields__.keys():
            combined[f] = getattr(self, f) + getattr(other, f)
        return self.__class__(**combined)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as attr_err:
            raise KeyError(f"Key {key} not present") from attr_err


def extensions():
    """Returns a list of Cython extensions to be compiled."""

    # Define external dependencies
    mpi_ = ExternalDependency(
        extra_compile_args=petsctools.get_petscvariables()["MPICC_SHOW"].split()[1:],
    )
    numpy_ = ExternalDependency(include_dirs=[np.get_include()])
    petsc_dir = petsctools.get_petsc_dir()
    petsc_dirs = [petsc_dir, os.path.join(petsc_dir, petsctools.get_petsc_arch())]
    petsc_includes = [petsc4py.get_include()] + [
        os.path.join(d, "include") for d in petsc_dirs
    ]
    petsc_ = ExternalDependency(
        libraries=["petsc"],
        include_dirs=petsc_includes,
        library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
        runtime_library_dirs=[os.path.join(petsc_dirs[-1], "lib")],
    )

    # Define Cython extensions
    cython_list = [
        Extension(
            name="animate.cython.numbering",
            language="c",
            sources=[os.path.join("animate", "cython", "numbering.pyx")],
            **(mpi_ + petsc_ + numpy_),
        )
    ]
    return cythonize(cython_list)


if __name__ == "__main__":
    # Run the setup function to build the Cython extensions
    setup(
        packages=find_packages(),
        ext_modules=extensions(),
    )
