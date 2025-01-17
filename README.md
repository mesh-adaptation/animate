# Animate
## Anisotropic mesh adaptation toolkit for Firedrake
![GitHub top language](https://img.shields.io/github/languages/top/mesh-adaptation/animate)
![GitHub repo size](https://img.shields.io/github/repo-size/mesh-adaptation/animate)
[![Slack](https://img.shields.io/badge/Animate_Slack_Channel-4A154B?logo=slack&logoColor=fff)](https://firedrakeproject.slack.com/archives/C07KTDB3JNB)

Animate is a mesh adaptation toolkit enabling users to anisotropically adapt a mesh based on a Riemannian metric framework with control over the shape, orientation and size of the resulting mesh elements. Animate is built for use with Firedrake and  strongly leverages PETSc DMPlex functionality. The implementation of metric-based mesh adaptation used in PETSc assumes that the metric is piece-wise linear and continuous, with its degrees of freedom at the mesh vertices.

For more information on Firedrake, please see: [Firedrake Website](https://www.firedrakeproject.org/).
For more information on PETSc, please see: [PETSc metric-based mesh adaptation](https://petsc.org/release/docs/manual/dmplex/#metric-based-mesh-adaptation)

## Installation

For installation instructions, we refer to the [Wiki page](https://github.com/mesh-adaptation/mesh-adaptation-docs/wiki/Installation-Instructions).
