"""Éléments de référence P2 / Q2 et quadratures."""
from math import factorial

import numpy as np
import pytest

from reference import (Q2_A, Q2_B, QUAD, TRI, p2_gradients, p2_values, q2_gradients, q2_values,
                       quad_quadrature, triangle_quadrature)

P2_NODES = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], dtype=float)
Q2_NODES = np.column_stack((Q2_A, Q2_B)).astype(float)
ELEMENTS = [(p2_values, p2_gradients, P2_NODES), (q2_values, q2_gradients, Q2_NODES)]


@pytest.mark.parametrize("values, _, nodes", ELEMENTS)
def test_lagrange_nodal(values, _, nodes):
    """psi_i(x_j) = delta_ij."""
    np.testing.assert_allclose(values(nodes[:, 0], nodes[:, 1]), np.eye(len(nodes)), atol=1e-14)


@pytest.mark.parametrize("ref", [TRI, QUAD])
def test_partition_of_unity(ref):
    np.testing.assert_allclose(ref.psi.sum(axis=0), 1.0, atol=1e-14)
    np.testing.assert_allclose(ref.dpsi.sum(axis=1), 0.0, atol=1e-13)


@pytest.mark.parametrize("values, gradients, _", ELEMENTS)
def test_gradients_match_finite_differences(values, gradients, _):
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.05, 0.45, (2, 10))
    eps = 1e-6
    gx = (values(x + eps, y) - values(x - eps, y)) / (2 * eps)
    gy = (values(x, y + eps) - values(x, y - eps)) / (2 * eps)
    np.testing.assert_allclose(gradients(x, y), np.array([gx, gy]), atol=1e-8)


@pytest.mark.parametrize("a, b", [(a, b) for a in range(5) for b in range(5) if a + b <= 4])
def test_triangle_quadrature_degree_4(a, b):
    """int_T x^a y^b = a! b! / (a + b + 2)! sur le triangle de référence."""
    pts, w = triangle_quadrature()
    exact = factorial(a) * factorial(b) / factorial(a + b + 2)
    assert np.isclose(w @ (pts[:, 0] ** a * pts[:, 1] ** b), exact, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("a, b", [(a, b) for a in range(6) for b in range(6)])
def test_quad_quadrature_degree_5(a, b):
    """Gauss-Legendre 3 x 3 : exacte pour x^a y^b, a, b <= 5, sur [-1, 1]^2."""
    pts, w = quad_quadrature()
    moment = [2 / (n + 1) if n % 2 == 0 else 0.0 for n in (a, b)]
    assert np.isclose(w @ (pts[:, 0] ** a * pts[:, 1] ** b), moment[0] * moment[1], atol=1e-13)
