"""
Helpers for building :class:`pyop3.Function` kernels from the plain-C source
files under ``animate/c``.

The C files use a simple section convention:

* ``/* PREAMBLE-BEGIN */`` ... ``/* PREAMBLE-END */`` marks code that should
  go into loopy *preambles* (helpers shared by several kernels, includes).
  It is emitted once per kernel.
* Each remaining top-level ``void name(args...) { body }`` is a kernel.
"""

from __future__ import annotations

import os
import re

import pyop3 as op3
from firedrake.slate.slac.compiler import BLASLAPACK_INCLUDE, BLASLAPACK_LIB

_PREAMBLE_BEGIN = "/* PREAMBLE-BEGIN */"
_PREAMBLE_END = "/* PREAMBLE-END */"


def _c_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "c"))


def _split_source(source):
    """Split a C source file into (preamble, kernel-file-text).

    :arg source: contents of a C source file
    :type source: str
    :return: preamble and kernel file text
    :rtype: tuple[str, str]
    """
    begin = source.index(_PREAMBLE_BEGIN) + len(_PREAMBLE_BEGIN)
    end = source.index(_PREAMBLE_END)
    preamble = source[begin:end]
    rest = source[: source.index(_PREAMBLE_BEGIN)] + source[end + len(_PREAMBLE_END) :]
    return preamble, rest


def _extract_function_body(source, name):
    """Extract the body of ``void name(...) { ... }`` from ``source``.

    Uses simple brace counting; the source files are hand-written and free
    of nested braces inside strings or comments.

    :arg source: contents of a C source file
    :type source: str
    :arg name: name of the kernel to call
    :type name: str
    :return: the kernel body
    :rtype: str
    """
    # Anchor on "void name(" at start-of-line (public kernels are top-level).
    pattern = re.compile(rf"^void\s+{re.escape(name)}\s*\(", re.MULTILINE)
    m = pattern.search(source)
    if m is None:
        raise ValueError(f"Function '{name}' not found in kernel source.")
    # Find the opening brace of the body.
    open_idx = source.index("{", m.end())
    depth = 0
    for i in range(open_idx, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_idx + 1 : i]
    raise ValueError(f"Unbalanced braces in kernel '{name}'.")


def get_metric_kernel(func, dim, args):
    """
    Build a :class:`pyop3.Function` for one of the metric kernels in
    ``animate/c/metric{dim}d.c``.

    :arg func: kernel name (must match a ``void func(...) { ... }`` definition
        in the C file).
    :type func: str
    :arg dim: spatial dimension (2 or 3).
    :type dim: int
    :arg args: list of ``(arg_name, dtype, intent)`` triples describing the
        loopy kernel arguments. Dtypes are usually ``op3.ScalarType``;
        intents are ``op3.READ`` / ``op3.WRITE`` / ``op3.RW``.
    :type args: list[tuple[str, object, object]]
    :return: the compiled kernel, ready for ``op3.loop``.
    :rtype: op3.Function
    """
    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D metric kernels are supported, not {dim}D.")
    path = os.path.join(_c_dir(), f"metric{dim}d.c")
    with open(path) as fh:
        source = fh.read()
    preamble, kernels = _split_source(source)
    body = _extract_function_body(kernels, func)
    return op3.Function.from_c_string(
        func,
        body,
        args,
        preambles=[
            ("10_blaslapack", "#include <petscblaslapack.h>"),
            ("20_metric", preamble),
        ],
        include_dirs=BLASLAPACK_INCLUDE,
        libs=BLASLAPACK_LIB,
    )
