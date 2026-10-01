"""Conditions de Dirichlet et résolution (section 5.6 de main.tex)."""
import numpy as np
from scipy.sparse.linalg import spsolve

from assembly import assemble


def solve(mesh, f, g_beta=None, k=1.0):
    """Résout -Δu + k u = f : A_II U_I = F_I - A_IB G_B, puis U complété par U_B = G_B.

    G_B vaut 0 sur dOmega et g_beta(x, y) sur dbeta (0 si g_beta est None).
    """
    A, F = assemble(mesh, f, k)
    A = A.tocsr()
    U = np.zeros(mesh.n_nodes)
    if g_beta is not None:
        x, y = mesh.coords[mesh.bnd_beta].T
        U[mesh.bnd_beta] = g_beta(x, y)
    interior = np.ones(mesh.n_nodes, dtype=bool)
    interior[mesh.bnd] = False
    rhs = F[interior] - A[interior][:, ~interior] @ U[~interior]
    U[interior] = spsolve(A[interior][:, interior].tocsc(), rhs)
    return U
