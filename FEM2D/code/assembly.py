"""Matrices élémentaires, assemblage creux et calcul des erreurs, globales et par élément (sections 5.4, 5.5 et 6.2).

Problème -div(a grad u) + k u = f, a > 0 et k >= 0 constants ou fonctions de (x, y) ; termes de
bord des conditions de Neumann et de Robin a du/dn + alpha u = g sur les arêtes P2 de bord.

Tous les éléments sont traités de façon isoparamétrique : F_e(x^) = sum_k x_k psi^_k(x^).
Pour un triangle droit ou un carré, cela redonne exactement la transformation affine.
"""
import numpy as np
from scipy.sparse import coo_matrix

from reference import QUAD, TRI

ELEMENT_NAME = {"tri": "triangle", "quad": "quadrilatère"}

# Arête P2 de bord, paramétrée par s dans [0, 1] : extrémités (s = 0, s = 1) puis milieu (s = 1/2),
# Gauss-Legendre à 3 points (exacte au degré 5)
_S = 0.5 + 0.5 * np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
EDGE_W = np.array([5, 8, 5]) / 18
EDGE_PSI = np.array([(1 - _S) * (1 - 2 * _S), _S * (2 * _S - 1), 4 * _S * (1 - _S)])  # (3, nq)
EDGE_DPSI = np.array([4 * _S - 3, 4 * _S - 1, 4 - 8 * _S])


def evaluate(c, x, y):
    """Coefficient ou donnée c (constante ou fonction (x, y)) aux points (x, y)."""
    return np.broadcast_to(np.asarray(c(x, y) if callable(c) else c, dtype=float), np.shape(x))


def element_blocks(mesh):
    """Triplets (type, connectivité, élément de référence) non vides."""
    return [(name, conn, ref) for name, conn, ref in (("tri", mesh.tri, TRI), ("quad", mesh.quad, QUAD))
            if len(conn)]


def jacobian(X, G):
    """X (ne, nloc, 2) : nœuds des éléments ; G (2, nloc) : gradients de référence en x^_q.

    Renvoie J_q (ne, 2, 2), J[e, d, a] = dx_d / dx^_a, et det J_q (ne,).
    """
    J = X.transpose(0, 2, 1) @ G.T
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
        B = np.linalg.inv(J).transpose(0, 2, 1) @ G
        yield psi, psi @ X, ref.w[q] * det, B


def assemble(mesh, f, k=1.0, a=1.0):
    """Assemble A (creuse, CSR), A_ij = int a grad phi_i . grad phi_j + k phi_i phi_j, pour
    -div(a grad u) + k u = f, et le second membre F. a, k, f : constantes ou fonctions (x, y)."""
    N = mesh.n_nodes
    rows, cols, vals = [], [], []
    F = np.zeros(N)
    for _, conn, ref in element_blocks(mesh):
        ne, nloc = conn.shape
        Ae = np.zeros((ne, nloc, nloc))
        Fe = np.zeros((ne, nloc))
        for psi, xq, wd, B in quadrature_points(mesh, conn, ref):
            aq, kq = evaluate(a, *xq.T), evaluate(k, *xq.T)
            if np.any(aq <= 0) or np.any(kq < 0):
                raise ValueError("Il faut a > 0 et k positif ou nul en tout point de D "
                                 f"(min a = {aq.min():.3g}, min k = {kq.min():.3g}).")
            # (B^T B)_kl = sum_d B_dk B_dl
            Ae += (wd * aq)[:, None, None] * (B.transpose(0, 2, 1) @ B) \
                + (wd * kq)[:, None, None] * np.outer(psi, psi)
            Fe += (wd * evaluate(f, *xq.T))[:, None] * psi[None, :]
        rows.append(np.repeat(conn, nloc, axis=1).ravel())
        cols.append(np.tile(conn, (1, nloc)).ravel())
        vals.append(Ae.ravel())
        np.add.at(F, conn.ravel(), Fe.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, N)).tocsr()
    return A, F


def edge_points(mesh, edges):
    """Pour chaque point de quadrature des arêtes edges (m, 3) : (psi_q (3,), x_q (m, 2), w_q |x'(s_q)|)."""
    X = mesh.coords[edges]
    for q in range(len(EDGE_W)):
        yield EDGE_PSI[:, q], EDGE_PSI[:, q] @ X, EDGE_W[q] * np.linalg.norm(EDGE_DPSI[:, q] @ X, axis=1)


def assemble_boundary(mesh, edges, alpha, g):
    """Termes de la condition a du/dn + alpha u = g (Neumann : alpha = 0) sur les arêtes edges :
    matrice creuse des int alpha phi_i phi_j et vecteur des int g phi_i."""
    N = mesh.n_nodes
    Ae = np.zeros((len(edges), 3, 3))
    Fe = np.zeros((len(edges), 3))
    for psi, xq, wl in edge_points(mesh, edges):
        al = evaluate(alpha, *xq.T)
        if np.any(al < 0):
            raise ValueError(f"Condition de Robin : alpha doit être positif ou nul (min {al.min():.3g}).")
        Ae += (wl * al)[:, None, None] * np.outer(psi, psi)
        Fe += (wl * evaluate(g, *xq.T))[:, None] * psi[None, :]
    A = coo_matrix((Ae.ravel(), (np.repeat(edges, 3, axis=1).ravel(), np.tile(edges, (1, 3)).ravel())),
                   shape=(N, N)).tocsr()
    F = np.zeros(N)
    np.add.at(F, edges.ravel(), Fe.ravel())
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
            grad_uh = (B @ Ue[:, :, None])[:, :, 0]
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
