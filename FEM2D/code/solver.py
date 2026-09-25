"""Conditions de Dirichlet et résolution (section 5.6 de main.tex)."""
import numpy as np
from scipy.sparse.linalg import spsolve

from assembly import assemble


def solve(mesh, f):
    """Résout A_II U_I = F_I et renvoie U complété par U_B = 0."""
    A, F = assemble(mesh, f)
    interior = np.ones(mesh.n_nodes, dtype=bool)
    interior[mesh.bnd] = False
    U = np.zeros(mesh.n_nodes)
    U[interior] = spsolve(A[interior][:, interior].tocsc(), F[interior])
    return U
