"""
Functions for computing mesh quality measures.
"""
import os
import firedrake
from firedrake import Function, FunctionSpace, interpolate
from firedrake.mesh import MeshGeometry
from firedrake.petsc import PETSc
from pyop2 import op2
from pyop2.utils import get_petsc_dir
import ufl


PETSC_DIR, PETSC_ARCH = get_petsc_dir()
include_dir = ["%s/include/eigen3" % PETSC_ARCH]


class QualityMeasure:
    """
    Class for computing quality measures associated with a given mesh.

    Choices of quality measure:
      * ``min_angle``: the minimum angle of each cell
      * ``area``: the area of each cell in a 2D triangular mesh
      * ``volume``: the volume of each cell in a 3D tetrahedral mesh
      * ``facet_area``: the area of each *facet*.
      * ``aspect_ratio``: the aspect ratio of each cell
      * ``eskew``: the equiangle skew of each cell
      * ``skewness``: the skewness of each cell in a 2D triangular mesh
      * ``scaled_jacobian``: the scaled Jacobian of each cell
      * ``metric``:  given a Riemannian metric, this function outputs the
        value of the quality measure :eq:`Q_M` based on the transformation
        encoded by the metric.
    """

    _measures = (
        "min_angle",
        "area",
        "volume",
        "facet_area",
        "aspect_ratio",
        "eskew",
        "skewness",
        "scaled_jacobian",
        "metric",
    )

    @PETSc.Log.EventDecorator()
    def __init__(
        self, mesh: MeshGeometry, metric: Function = None, python: bool = False
    ):
        """
        :arg mesh: the input mesh to do computations on
        :arg metric: the tensor field representing the metric space transformation
        :kwarg python: compute the measure using Python?
        """
        self.mesh = mesh
        self.metric = metric
        self.python = python
        self.dim = mesh.topological_dimension()
        self.coords = mesh.coordinates
        self.P0 = FunctionSpace(mesh, "DG", 0)
        src_dir = os.path.join(os.path.dirname(__file__), "cxx")
        self.fname = os.path.join(src_dir, f"quality{self.dim}d.cxx")

    def _get_dats(self, func):
        dats = (
            func.dat(op2.WRITE, func.cell_node_map()),
            self.coords.dat(op2.READ, self.coords.cell_node_map()),
        )
        if self.metric is not None:
            dats += (self.metric.dat(op2.READ, self.metric.cell_node_map()),)
        return dats

    @PETSc.Log.EventDecorator()
    def __call__(self, name: str) -> Function:
        if name not in QualityMeasure._measures:
            raise ValueError(f"Quality measure '{name}' not recognised.")
        msg = (
            f"Quality measure '{name}' not implemented in the {self.dim}D case in C++."
        )
        if self.python:
            return self._call_python(name)
        elif name == "facet_area":
            raise NotImplementedError(msg)
        elif name == "skewness" and self.dim == 3:
            raise NotImplementedError(msg)
        with open(self.fname, "r") as f:
            code = f.read()
        func = Function(self.P0, name=name)
        kwargs = dict(cpp=True, include_dirs=include_dir)
        kernel = op2.Kernel(code, f"get_{name}", **kwargs)
        op2.par_loop(kernel, self.mesh.cell_set, *self._get_dats(func))
        return func

    @PETSc.Log.EventDecorator()
    def _call_python(self, name: str) -> Function:
        if name in ("area", "volume"):
            return interpolate(ufl.CellVolume(self.mesh), self.P0)
        elif name == "facet_area":
            HDivTrace = FunctionSpace(self.mesh, "HDiv Trace", 0)
            v = firedrake.TestFunction(HDivTrace)
            u = firedrake.TrialFunction(HDivTrace)
            facet_area = Function(HDivTrace, name="Facet areas")
            mass_term = v("+") * u("+") * ufl.dS + v * u * ufl.ds
            rhs = (
                v("+") * ufl.FacetArea(self.mesh) * ufl.dS
                + v * ufl.FacetArea(self.mesh) * ufl.ds
            )
            sp = {
                "snes_type": "ksponly",
                "ksp_type": "preonly",
                "pc_type": "jacobi",
            }
            firedrake.solve(mass_term == rhs, facet_area, solver_parameters=sp)
            return facet_area
        elif name == "aspect_ratio" and self.dim == 2:
            P0_ten = firedrake.TensorFunctionSpace(self.mesh, "DG", 0)
            J = interpolate(ufl.Jacobian(self.mesh), P0_ten)
            edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
            edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
            edge3 = edge1 - edge2
            a = ufl.sqrt(ufl.dot(edge1, edge1))
            b = ufl.sqrt(ufl.dot(edge2, edge2))
            c = ufl.sqrt(ufl.dot(edge3, edge3))
            ar = Function(self.P0)
            ar.interpolate(a * b * c / ((a + b - c) * (b + c - a) * (c + a - b)))
            return ar
        elif name == "scaled_jacobian" and self.dim == 2:
            P0_ten = firedrake.TensorFunctionSpace(self.mesh, "DG", 0)
            J = interpolate(ufl.Jacobian(self.mesh), P0_ten)
            edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
            edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
            edge3 = edge1 - edge2
            a = ufl.sqrt(ufl.dot(edge1, edge1))
            b = ufl.sqrt(ufl.dot(edge2, edge2))
            c = ufl.sqrt(ufl.dot(edge3, edge3))
            detJ = ufl.JacobianDeterminant(self.mesh)
            jacobian_sign = ufl.sign(detJ)
            max_product = ufl.max_value(
                ufl.max_value(ufl.max_value(a * b, a * c), ufl.max_value(b * c, b * a)),
                ufl.max_value(c * a, c * b),
            )
            return interpolate(detJ / max_product * jacobian_sign, self.P0)
        else:
            raise NotImplementedError(
                f"Quality measure '{name}' not implemented in the {self.dim}D case in Python."
            )
