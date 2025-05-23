import numpy as np
import ufl
import ufl.core.expr

__all__ = ["gram_schmidt", "construct_basis"]


def gram_schmidt(*vectors, normalise=False):
    """
    Given some vectors, construct an orthogonal basis
    using Gram-Schmidt orthogonalisation.

    :arg vectors: the vectors to orthogonalise
    :type vectors: :class:`numpy.ndarray` or :class:`ufl.core.expr.Expr`
    :kwarg normalise: do we want an orthonormal basis?
    :type normalise: :class:`bool`
    """
    if isinstance(vectors[0], np.ndarray):
        expected = np.ndarray
        dot = np.dot
        sqrt = np.sqrt
    elif isinstance(vectors[0], ufl.core.expr.Expr):
        expected = ufl.core.expr.Expr
        dot = ufl.dot
        sqrt = ufl.sqrt
    else:
        raise TypeError(
            f"Cannot apply Gram-Schmidt to vectors of type '{type(vectors[0])}'."
        )

    # Check that vector types match
    for vi in vectors[1:]:
        if not isinstance(vi, expected):
            raise TypeError(
                f"Inconsistent vector types: '{expected}' vs. '{type(vi)}'."
            )

    def proj(x, y):
        return dot(x, y) / dot(x, x) * x

    # Apply Gram-Schmidt algorithm
    u = []
    for i, vi in enumerate(vectors):
        if i > 0:
            vi -= sum([proj(uj, vi) for uj in u])
        u.append(vi / sqrt(dot(vi, vi)) if normalise else vi)

    # Ensure consistency of outputs
    if isinstance(vectors[0], np.ndarray):
        u = [np.array(ui) for ui in u]

    return u


def construct_basis(vector, normalise=True):
    """
    Construct a basis from a given vector.

    :arg vector: the starting vector
    :type vector: :class:`numpy.ndarray` or :class:`ufl.core.expr.Expr`
    :kwarg normalise: do we want an orthonormal basis?
    :type normalise: :class:`bool`
    """
    is_numpy = isinstance(vector, np.ndarray)
    if is_numpy:
        if len(vector.shape) > 1:
            raise ValueError(
                f"Expected a vector, got an array of shape {vector.shape}."
            )
        as_vector = np.array
        dim = vector.shape[0]
    else:
        if not isinstance(vector, ufl.core.expr.Expr):
            raise TypeError(f"Expected UFL Expr, not '{type(vector)}'.")
        as_vector = ufl.as_vector
        dim = ufl.domain.extract_unique_domain(vector).topological_dimension()

    if dim not in (2, 3):
        raise ValueError(f"Dimension {dim} not supported.")
    vectors = [vector]

    # Generate some arbitrary vectors and apply Gram-Schmidt
    if dim == 2:
        vectors.append(as_vector((-vector[1], vector[0])))
    else:
        vectors.append(as_vector((vector[1], vector[2], vector[0])))
        vectors.append(as_vector((vector[2], vector[0], vector[1])))
        # TODO: Account for the case where all three components match (#129)
    return gram_schmidt(*vectors, normalise=normalise)
