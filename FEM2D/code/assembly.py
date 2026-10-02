"""Matrices élémentaires, assemblage creux et calcul des erreurs, globales et par élément (sections 5.4, 5.5 et 6.2).

Tous les éléments sont traités de façon isoparamétrique : F_e(x^) = sum_k x_k psi^_k(x^).
Pour un triangle droit ou un carré, cela redonne exactement la transformation affine.
"""
import numpy as np
from scipy.sparse import coo_matrix

from reference import QUAD, TRI

ELEMENT_NAME = {"tri": "triangle", "quad": "quadrilatère"}


def element_blocks(mesh):
    """Triplets (type, connectivité, élément de référence) non vides."""
    return [(name, conn, ref) for name, conn, ref in (("tri", mesh.tri, TRI), ("quad", mesh.quad, QUAD))
            if len(conn)]


def jacobian(X, G):
    """X (ne, nloc, 2) : nœuds des éléments ; G (2, nloc) : gradients de référence en x^_q.

    Renvoie J_q (ne, 2, 2), J[e, d, a] = dx_d / dx^_a, et det J_q (ne,).
    """
    J = np.einsum("ekd,ak->eda", X, G)
    return J, J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def check_jacobians(mesh):
    """Lève ValueError si det J <= 0 en un point de quadrature d'un élément (élément retourné).

    Arrive surtout pour un triangle dont l'arête courbe, posée sur dbeta, rejoint presque
    une autre partie de beta (deux composantes très proches, ou une partie étroite de beta).
    """
    bad = []
    for name, conn, ref in element_blocks(mesh):
        X = mesh.coords[conn]
        det = np.array([jacobian(X, ref.dpsi[:, :, q])[1] for q in range(ref.nq)])
        bad += [(name, e, X[e].mean(axis=0)) for e in np.nonzero((det <= 0).any(axis=0))[0]]
    if bad:
        where = ", ".join(f"{ELEMENT_NAME[name]} n° {e} vers ({c[0]:.4g}, {c[1]:.4g})"
                          for name, e, c in bad[:3])
        raise ValueError(f"Jacobien négatif ou nul pour {len(bad)} élément(s) : {where}"
                         f"{', ...' if len(bad) > 3 else ''}. Une arête courbe posée sur dbeta est trop "
                         "proche d'une autre partie de beta : réduire h_beta ou écarter les composantes.")


def quadrature_points(mesh, conn, ref):
    """Pour chaque point de quadrature x^_q : (psi_q (nloc,), x_q (ne, 2), w_q |det J_q| (ne,), B_q).

    B_q = J_q^{-T} G_q (ne, 2, nloc) : gradients physiques des fonctions de base en x_q.
    """
    X = mesh.coords[conn]
    for q in range(ref.nq):
        psi, G = ref.psi[:, q], ref.dpsi[:, :, q]
        J, det = jacobian(X, G)
        if np.any(det <= 0):
            check_jacobians(mesh)  # message détaillé
        B = np.einsum("ead,ak->edk", np.linalg.inv(J), G)
        yield psi, np.einsum("ekd,k->ed", X, psi), ref.w[q] * det, B


def assemble(mesh, f, k=1.0):
    """Assemble A = R + k M (creuse, CSR), pour -Δu + k u = f, et le second membre F."""
    N = mesh.n_nodes
    rows, cols, vals = [], [], []
    F = np.zeros(N)
    for _, conn, ref in element_blocks(mesh):
        ne, nloc = conn.shape
        Ae = np.zeros((ne, nloc, nloc))
        Fe = np.zeros((ne, nloc))
        for psi, xq, wd, B in quadrature_points(mesh, conn, ref):
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
    for name, conn, ref in element_blocks(mesh):
        Ue = U[conn]
        area, l2, h1_semi, mean = (np.zeros(len(conn)) for _ in range(4))
        for psi, xq, wd, B in quadrature_points(mesh, conn, ref):
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
