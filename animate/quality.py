"""
Functions for computing mesh quality measures.
"""

import firedrake
import ufl
from firedrake.petsc import PETSc

__all__ = ["QualityMeasure"]


class QualityMeasure:
    """
    Class for computing quality measures associated with a given mesh.

    Choices of quality measure:
      * ``area``: the area of each cell in a 2D triangular mesh
      * ``volume``: the volume of each cell in a 3D tetrahedral mesh
      * ``facet_area``: the area of each *facet*.
      * ``aspect_ratio``: the aspect ratio of each cell
      * ``scaled_jacobian``: the scaled Jacobian of each cell
    """

    _measures = (
        "area",
        "volume",
        "facet_area",
        "aspect_ratio",
        "scaled_jacobian",
    )

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh):
        """
        :arg mesh: the input mesh to do computations on
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        """
        self.mesh = mesh
        self.dim = mesh.topological_dimension
        self.coords = mesh.coordinates
        self.P0 = firedrake.FunctionSpace(mesh, "DG", 0)

    @PETSc.Log.EventDecorator()
    def __call__(self, name):
        """
        :arg name: the quality measure name
        :type name: :class:`str`
        :returns: the quality measure applied to the provided mesh
        :rtype: :class:`firedrake.function.Function`
        """
        if name not in QualityMeasure._measures:
            raise ValueError(f"Quality measure '{name}' not recognised.")
        if name in ("area", "volume"):
            return firedrake.Function(self.P0).interpolate(ufl.CellVolume(self.mesh))
        elif name == "facet_area":
            HDivTrace = firedrake.FunctionSpace(self.mesh, "HDiv Trace", 0)
            v = firedrake.TestFunction(HDivTrace)
            u = firedrake.TrialFunction(HDivTrace)
            facet_area = firedrake.Function(HDivTrace, name="Facet areas")
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
            J = firedrake.Function(P0_ten).interpolate(ufl.Jacobian(self.mesh))
            edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
            edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
            edge3 = edge1 - edge2
            a = ufl.sqrt(ufl.dot(edge1, edge1))
            b = ufl.sqrt(ufl.dot(edge2, edge2))
            c = ufl.sqrt(ufl.dot(edge3, edge3))
            ar = firedrake.Function(self.P0)
            ar.interpolate(a * b * c / ((a + b - c) * (b + c - a) * (c + a - b)))
            return ar
        elif name == "scaled_jacobian" and self.dim == 2:
            P0_ten = firedrake.TensorFunctionSpace(self.mesh, "DG", 0)
            J = firedrake.Function(P0_ten).interpolate(ufl.Jacobian(self.mesh))
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
            return firedrake.Function(self.P0).interpolate(
                detJ / max_product * jacobian_sign
            )
        else:
            raise NotImplementedError(
                f"Quality measure '{name}' not implemented in the {self.dim}D case."
            )
