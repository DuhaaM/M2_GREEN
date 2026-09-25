"""Éléments de référence P2 / Q2 et formules de quadrature (sections 5.2 et 5.3 de main.tex).

Convention : les valeurs sont renvoyées sous la forme psi[k, q] et les gradients
sous la forme dpsi[a, k, q] (a = direction x^ ou y^, k = nœud local, q = point).
"""
import numpy as np


# ----------------------------------------------------------------------------
# Triangle P2 : sommets (0,0), (1,0), (0,1) puis milieux (1/2,0), (1/2,1/2), (0,1/2)
# ----------------------------------------------------------------------------
def p2_values(xh, yh):
    l1 = 1.0 - xh - yh
    return np.array([
        l1 * (2 * l1 - 1),
        xh * (2 * xh - 1),
        yh * (2 * yh - 1),
        4 * l1 * xh,
        4 * xh * yh,
        4 * l1 * yh,
    ])


def p2_gradients(xh, yh):
    l1 = 1.0 - xh - yh
    z = np.zeros_like(xh)
    gx = [-(4 * l1 - 1), 4 * xh - 1, z, 4 * (1 - 2 * xh - yh), 4 * yh, -4 * yh]
    gy = [-(4 * l1 - 1), z, 4 * yh - 1, -4 * xh, 4 * xh, 4 * (1 - xh - 2 * yh)]
    return np.array([gx, gy])


# ----------------------------------------------------------------------------
# Quadrilatère Q2 sur [-1,1]^2 : sommets, milieux (bas, droite, haut, gauche), centre
# ----------------------------------------------------------------------------
Q2_A = np.array([-1, 1, 1, -1, 0, 1, 0, -1, 0])
Q2_B = np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])


def _lagrange_1d(a, s):
    if a == -1:
        return 0.5 * s * (s - 1)
    if a == 0:
        return 1 - s**2
    return 0.5 * s * (s + 1)


def _dlagrange_1d(a, s):
    if a == -1:
        return s - 0.5
    if a == 0:
        return -2 * s
    return s + 0.5


def q2_values(xh, yh):
    return np.array([_lagrange_1d(a, xh) * _lagrange_1d(b, yh) for a, b in zip(Q2_A, Q2_B)])


def q2_gradients(xh, yh):
    gx = [_dlagrange_1d(a, xh) * _lagrange_1d(b, yh) for a, b in zip(Q2_A, Q2_B)]
    gy = [_lagrange_1d(a, xh) * _dlagrange_1d(b, yh) for a, b in zip(Q2_A, Q2_B)]
    return np.array([gx, gy])


# ----------------------------------------------------------------------------
# Quadratures
# ----------------------------------------------------------------------------
def triangle_quadrature():
    """Règle de Gauss à 6 points, exacte au degré 4 (somme des poids = 1/2)."""
    a, b = 0.445948490915965, 0.091576213509771
    wa, wb = 0.111690794839005, 0.054975871827661
    pts = np.array([[a, a], [1 - 2 * a, a], [a, 1 - 2 * a],
                    [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]])
    w = np.array([wa, wa, wa, wb, wb, wb])
    return pts, w


def quad_quadrature():
    """Gauss-Legendre 3 x 3 sur [-1,1]^2."""
    s = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
    w1 = np.array([5 / 9, 8 / 9, 5 / 9])
    X, Y = np.meshgrid(s, s, indexing="ij")
    W = np.outer(w1, w1)
    return np.column_stack((X.ravel(), Y.ravel())), W.ravel()


class ReferenceElement:
    """Valeurs et gradients des fonctions de base aux points de quadrature, calculés une fois."""

    def __init__(self, values, gradients, quadrature):
        pts, self.w = quadrature()
        self.psi = values(pts[:, 0], pts[:, 1])        # (nloc, nq)
        self.dpsi = gradients(pts[:, 0], pts[:, 1])    # (2, nloc, nq)
        self.nloc = self.psi.shape[0]
        self.nq = len(self.w)


TRI = ReferenceElement(p2_values, p2_gradients, triangle_quadrature)
QUAD = ReferenceElement(q2_values, q2_gradients, quad_quadrature)
