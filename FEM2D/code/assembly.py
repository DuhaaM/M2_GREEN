"""Matrices élémentaires, assemblage creux et calcul des erreurs (sections 5.4, 5.5 et 6.2).

Tous les éléments sont traités de façon isoparamétrique : F_e(x^) = sum_k x_k psi^_k(x^).
Pour un triangle droit ou un carré, cela redonne exactement la transformation affine.
"""
import numpy as np
from scipy.sparse import coo_matrix

from reference import QUAD, TRI


def element_blocks(mesh):
    """Paires (connectivité, élément de référence) non vides."""
    return [(conn, ref) for conn, ref in ((mesh.tri, TRI), (mesh.quad, QUAD)) if len(conn)]


def geometry_at(X, G):
    """X (ne, nloc, 2) : nœuds des éléments ; G (2, nloc) : gradients de référence en x^_q.

    Renvoie d_q = |det J_q| (ne,) et B_q = J_q^{-T} G_q (ne, 2, nloc).
    """
    J = np.einsum("ekd,ak->eda", X, G)  # J[e, d, a] = dx_d / dx^_a
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    if np.any(det <= 0):
        raise ValueError("Jacobien négatif ou nul : élément retourné ou dégénéré.")
    invJ = np.linalg.inv(J)
    B = np.einsum("ead,ak->edk", invJ, G)
    return det, B


def assemble(mesh, f):
    """Assemble A = K + M (creuse, CSR) et le second membre F."""
    N = mesh.n_nodes
    rows, cols, vals = [], [], []
    F = np.zeros(N)
    for conn, ref in element_blocks(mesh):
        X = mesh.coords[conn]
        ne, nloc = conn.shape
        Ae = np.zeros((ne, nloc, nloc))
        Fe = np.zeros((ne, nloc))
        for q in range(ref.nq):
            psi = ref.psi[:, q]
            det, B = geometry_at(X, ref.dpsi[:, :, q])
            xq = np.einsum("ekd,k->ed", X, psi)
            wd = ref.w[q] * det
            Ae += wd[:, None, None] * (np.einsum("edk,edl->ekl", B, B) + np.outer(psi, psi))
            Fe += (wd * f(xq[:, 0], xq[:, 1]))[:, None] * psi[None, :]
        rows.append(np.repeat(conn, nloc, axis=1).ravel())
        cols.append(np.tile(conn, (1, nloc)).ravel())
        vals.append(Ae.ravel())
        np.add.at(F, conn.ravel(), Fe.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsr()
    return A, F


def compute_errors(mesh, U, u_ex, grad_u_ex):
    """Erreurs ||u_ex - u_h||_L2 et ||u_ex - u_h||_H1 par quadrature."""
    err_l2, err_h1_semi = 0.0, 0.0
    for conn, ref in element_blocks(mesh):
        X = mesh.coords[conn]
        Ue = U[conn]
        for q in range(ref.nq):
            psi = ref.psi[:, q]
            det, B = geometry_at(X, ref.dpsi[:, :, q])
            xq = np.einsum("ekd,k->ed", X, psi)
            wd = ref.w[q] * det
            uh = Ue @ psi
            grad_uh = np.einsum("edk,ek->ed", B, Ue)
            gx, gy = grad_u_ex(xq[:, 0], xq[:, 1])
            err_l2 += np.sum(wd * (u_ex(xq[:, 0], xq[:, 1]) - uh) ** 2)
            err_h1_semi += np.sum(wd * ((gx - grad_uh[:, 0]) ** 2 + (gy - grad_uh[:, 1]) ** 2))
    return np.sqrt(err_l2), np.sqrt(err_l2 + err_h1_semi)
