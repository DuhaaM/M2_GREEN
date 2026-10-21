"""Patch test et ordres de convergence (solution manufacturée de la section 6)."""
import numpy as np
import pytest

from assembly import assemble, total_errors
from cases import DISK
from config import LX, LY
from geometry import disk
from mesh import build_mesh
from solver import solve
from validation import validation_problem


@pytest.mark.parametrize("k", [0.0, 1.0])
def test_patch_quadratic(k):
    """u de degré 2 appartient à V_h sur les éléments droits : A U = F exactement sur les
    lignes des nœuds intérieurs dont le support ne contient aucun triangle courbe."""
    h = 0.5
    mesh = build_mesh(disk(DISK["center"], DISK["R"]), LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)

    def u(x, y):
        return 1 + 2 * x - y + 0.5 * x**2 + 0.3 * x * y - 0.7 * y**2

    A, F = assemble(mesh, lambda x, y: 0.4 + k * u(x, y), k)  # -Δu = -(1 - 1.4)
    U = u(*mesh.coords.T)

    P = mesh.coords
    vertices = P[mesh.tri[:, [0, 1, 2, 0]]]
    curved = (np.abs(P[mesh.tri[:, 3:]] - 0.5 * (vertices[:, :3] + vertices[:, 1:])) > 1e-12).any(axis=(1, 2))
    check = np.ones(mesh.n_nodes, dtype=bool)
    check[mesh.bnd] = False
    check[mesh.tri[curved].ravel()] = False
    assert check.sum() > 100

    residual = (A @ U - F)[check]
    assert np.abs(residual).max() < 1e-10 * np.abs(F).max()


def test_convergence_orders():
    """P2 / Q2 : ordre 3 en norme L2, ordre 2 en norme H1."""
    hs = [0.5, 0.25, 0.125]
    errs = np.array([total_errors(validation_problem(h)[2]) for h in hs])
    orders = np.log(errs[:-1] / errs[1:]) / np.log(2)
    assert (orders[:, 0] > 2.7).all(), orders
    assert (orders[:, 1] > 1.8).all(), orders


def test_negative_k_rejected():
    """k < 0 : A_II peut être indéfinie, la factorisation sans pivotage est refusée."""
    h = 0.5
    mesh = build_mesh(disk(DISK["center"], DISK["R"]), LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)
    with pytest.raises(ValueError, match="positif ou nul"):
        solve(mesh, lambda x, y: np.ones_like(x), k=-1.0)
