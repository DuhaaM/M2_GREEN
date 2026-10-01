"""Matrices élémentaires, assemblage creux et calcul des erreurs, globales et par élément (sections 5.4, 5.5 et 6.2).

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


def assemble(mesh, f, k=1.0):
    """Assemble A = R + k M (creuse, CSR), pour -Δu + k u = f, et le second membre F."""
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
            Ae += wd[:, None, None] * (np.einsum("edk,edl->ekl", B, B) + k * np.outer(psi, psi))
            Fe += (wd * f(xq[:, 0], xq[:, 1]))[:, None] * psi[None, :]
        rows.append(np.repeat(conn, nloc, axis=1).ravel())
        cols.append(np.tile(conn, (1, nloc)).ravel())
        vals.append(Ae.ravel())
        np.add.at(F, conn.ravel(), Fe.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsr()
    return A, F


def element_errors(mesh, U, u_ex, grad_u_ex):
    """Erreurs locales par type d'élément ("tri", "quad").

    Renvoie {type: (|K|, ||u_ex - u_h||_L2(K)^2, |u_ex - u_h|_H1(K)^2, int_K (u_ex - u_h))},
    tableaux (ne,). La dernière quantité sert à l'écart-type de u_ex - u_h dans K.
    """
    out = {}
    for name, conn, ref in (("tri", mesh.tri, TRI), ("quad", mesh.quad, QUAD)):
        if not len(conn):
            continue
        X = mesh.coords[conn]
        Ue = U[conn]
        area, l2, h1_semi, mean = (np.zeros(len(conn)) for _ in range(4))
        for q in range(ref.nq):
            psi = ref.psi[:, q]
            det, B = geometry_at(X, ref.dpsi[:, :, q])
            xq = np.einsum("ekd,k->ed", X, psi)
            wd = ref.w[q] * det
            uh = Ue @ psi
            grad_uh = np.einsum("edk,ek->ed", B, Ue)
            gx, gy = grad_u_ex(xq[:, 0], xq[:, 1])
            area += wd
            e = u_ex(xq[:, 0], xq[:, 1]) - uh
            l2 += wd * e ** 2
            mean += wd * e
            h1_semi += wd * ((gx - grad_uh[:, 0]) ** 2 + (gy - grad_uh[:, 1]) ** 2)
        out[name] = (area, l2, h1_semi, mean)
    return out


def total_errors(errs):
    """Erreurs globales L2 et H1 à partir des erreurs locales de element_errors."""
    err_l2 = sum(l2.sum() for _, l2, _, _ in errs.values())
    err_h1_semi = sum(h1.sum() for _, _, h1, _ in errs.values())
    return np.sqrt(err_l2), np.sqrt(err_l2 + err_h1_semi)


def compute_errors(mesh, U, u_ex, grad_u_ex):
    """Erreurs ||u_ex - u_h||_L2 et ||u_ex - u_h||_H1 par quadrature."""
    return total_errors(element_errors(mesh, U, u_ex, grad_u_ex))
