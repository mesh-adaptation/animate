---
title: 'Animate: an anisotropic mesh adaptation toolkit for Firedrake'
tags:
  - Python
  - mesh adaptation
  - finite element
  - Firedrake
authors:
  - name: Joseph G. Wallwork
    affiliation: "1"
    orcid: 0000-0002-3646-091X
    corresponding: true
  - name: Stephan C. Kramer
    affiliation: "2"
    orcid: 0000-0002-9193-5092
  - name: Davor Dundovic
    affiliation: "3"
    orcid: 0009-0000-6085-4794
  - name: Eleda Johnson
    affiliation: "2"
    orcid: 0009-0008-9741-7836
  - name: Matthew D. Piggott
    affiliation: "2"
    orcid: 0000-0002-7526-6853
affiliations:
 - name: Institute of Computing for Climate Science, University of Cambridge, UK
   index: 1
 - name: Department of Earth Science and Engineering, Imperial College London, UK
   index: 2
 - name: Department of Geosciences, University of Oslo, Norway
   index: 3
date: 20 April 2025
bibliography: paper.bib

---

# Summary

*Animate* is a new Python package providing metric-based mesh adaptation
functionality for the *Firedrake* finite element package [@Firedrake].
Standard mesh adaptation approaches only control an element's size, whereas
metric-based adaptation has the key advantage that it can also be used to
control element orientation and shape.
As such, it can be used to introduce anisotropic mesh elements that conform to
features in the domain geometry and/or solution fields, allowing for better
representations thereof.

*Riemannian metric fields* are at the core of the metric-based approach.
Animate implements these in the *UFL* [@UFL] domain-specific language used by
Firedrake, with interfaces to the underlying metric implementation in *PETSc*
[@PETSc;@WK22].
In doing so, it exposes the *Mmg* [@Mmg] and *ParMmg* [@ParMmg] metric-based
mesh adaptation libraries via their coupling with PETSc.

# Statement of need

Mesh adaptation is an advanced discretisation technique for solving partial
differential equations (PDEs), which allows for variation of mesh resolution in
time and/or space.
By varying the degrees of freedom (DoFs) in this way, it is possible to
construct approximate PDE solutions with higher accuracy than a fixed-mesh
reference solution with an equal number of DoFs.
Viewed another way, mesh adaptation can be used to achieve the same solution
accuracy using fewer overall DoFs.
When implemented in an efficient way, this can also imply lower overall
time-to-solution and memory requirements.

Several research publications have investigated metric-based mesh adaptation in
Firedrake [@Pragmatic;@WB20;@DW25]
In previous years, this was achieved using the *Pragmatic* metric-based mesh
adaptation library via a branch coupling it into PETSc [@Pragmatic].
Unfortunately, Pragmatic is no longer maintained, so there was appetite to
couple to a different, more actively developed metric-based mesh adaptation
package.
This was achieved by @WK22, in which the Mmg (serial) and ParMmg (MPI parallel)
packages were coupled into PETSc.
It was initially proposed to expose this metric-based mesh adaptation
functionality in Firedrake itself via bindings between C and Python.
We later decided to create a separate package to handle metric-based mesh
adaptation, so that we could support advanced Riemannian metric constructions
(e.g., related to goal-oriented mesh adaptation) without bloating Firedrake with
functionality not integral to the finite element method.

<!-- TODO: Mention SOFIA mesh -->

## State of the field

Given that Animate provides mesh adaptation functionality for Firedrake
specifically, we restrict attention to other such works in the following review.
There are two main approaches to mesh adaptation: topology-preserving methods
($r$-adaptation) that only move the position of vertices and topology-deforming
methods ($h$-adaptation) that change the connectivity of the mesh.

### $r$-adaptation

Several $r$-adaptation approaches have been implemented in Firedrake, including
standard PDE-based approaches [@McRae18;@Fireshape], as well as data-driven ones
[@M2N;@UM2N;@RM24].
While such approaches can be very powerful, they lack the ability to
vary degrees of freedom in time -- the same number of DoFs must be used at all
timesteps -- and the initial mesh topology puts severe restrictions on how DoFs
may be redistributed spatially.

### $h$-adaptation

In terms of topology-deforming approaches, the previous coupling with Pragmatic
was a notable contribution, which provided the same adaptation operations as
Animate, albeit with a different driver.
More recently, hierarchical, *adaptive mesh refinement (AMR)* approaches have
been made available thanks to *ngsPETSc* [@ngsPETSc], which makes *NETGEN*
available via PETSc and exposes this in Firedrake.
AMR is a powerful approach, but a fundamentally different one from the
metric-based adaptation provided by Animate.

Recently, $h$-adaptive methods in Firedrake have been applied to free-boundary
problems posed as variational inequalities by @VIAMR.
They implemented algorithms that, in an *a posteriori* manner, locate the free
boundary and refine the mesh in its vicinity.
This may be driven by either using ngsPETSc's AMR or Animate's metric-based
approach.

For completeness, another $h$-adaptive approach can be found in the latest
release of *Icepack* [@Icepack], which uses *Triangle* and *MeshPy* to remesh
based on differences between approximate solutions with different degrees.

# Software description

Like Firedrake, Animate is a Python-based package written in the UFL [@UFL]
domain-specific language for finite element computations.
It makes use of object-oriented Python, with the `RiemannianMetric` class both
being the foundational object and providing the interfaces to PETSc's mesh
adaptation functionality via the *petsc4py* bindings [@petsc4py].
`RiemannianMetric` is a subclass of Firedrake's `Function` class, which
restricts attention to symmetric-positive-definite matrix-valued fields.
It implements standard metric construction methods, such as taking the Hessian
of an approximate PDE solution, as well as more advanced methods, such as those
based on goal-oriented error indicators.
It also provides utilities to normalise the metric and interface with PETSc to
set Mmg parameters and combine with one or more other metrics.

Animate has an automated test suite of unit and integration tests with excellent
coverage, as well as an automated linting workflow.
API documentation, examples, and long-form documentation may be found in the
appropriate subsection of the
[https://mesh-adaptation.github.io](https://mesh-adaptation.github.io) website
that Animate shares with other mesh adaptation tools developed and maintained by
the authors.

# Examples of use

Animate includes several examples as training materials and showcases of its
functionality.
[TODO: Dundo, please could you add the nice bubble shear image you generated
using Animate as a showcase?]

An earlier implementation (called *Pyroteus*) included both Animate and the
goal-oriented error estimation functionality now provided by *Goalie*.
These have since been separated out to give a clearer separation of concerns.
Animate (and its earlier implementation in Pyroteus) has been applied to several
computational and geophysical fluid mechanics applications spanning tracer
transport, tidal energy, and glaciology in @WB22;@WA22;@DW25.
Animate has also been used to provide training data for a data-driven mesh
adaptation approach in @WL22 and in a hybrid approach as a counterpart to a
data-driven $r$-adaptation technique in @JW25.

[TODO: Mention range of scales, from nanometres [@JA26] to global scale]

# Future development

In addition to the core functionality described above, Animate implements
several other utilities for mesh adaptation, such as mesh quality measures,
derivative recovery, and the Clément interpolant.
Work is currently underway to upstream these to PETSc and Firedrake so that the
wider community may benefit from them.

While Mmg supports mesh adaptation for both triangular (2D) and tetrahedral
(3D) meshes, the MPI parallel ParMmg implementation only supports the latter.
Work is currently underway in the ParMmg team to implement MPI parallel
adaptation on triangular meshes, which we will then expose in PETSc and Animate.
We are also working on exposing Mmg's level-set functionality.

# Acknowledgments

This work received funding under the embedded CSE program of the ARCHER2 UK
National Supercomputing Service
[https://www.archer2.ac.uk](https://www.archer2.ac.uk).
Support is also acknowledged from EPSRC under grant EP/R029423/1.
Ongoing maintenance of Animate has been supported by a grant from Huawei
Corporation Ltd.

# References
