"""Conditions aux limites et résolution (section 5.6 de main.tex).

Problème -div(a grad u) + k u = f dans D, avec sur chaque zone de bord (côté de Omega ou courbe
de dbeta, voir mesh.SIDES) l'une des conditions (n : normale sortante de D) :
    dirichlet  u = g
    neumann    a du/dn = g
    robin      a du/dn + alpha u = g   (alpha >= 0)
Par défaut : u = 0 sur dOmega et u = g_beta sur dbeta (0 si g_beta est None).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu

from assembly import assemble, assemble_boundary, evaluate
from mesh import SIDES

BC_KINDS = ("dirichlet", "neumann", "robin")


@dataclass(frozen=True)
class BC:
    kind: str = "dirichlet"
    g: object = 0.0      # constante ou fonction (x, y)
    alpha: object = 0.0  # coefficient de Robin (ignoré sinon)

    def __post_init__(self):
        if self.kind not in BC_KINDS:
            raise ValueError(f"condition inconnue : {self.kind} (choix : {', '.join(BC_KINDS)})")


def zone_matches(zone, name):
    """La zone donnée par l'utilisateur (omega, beta, un côté, betaN) désigne-t-elle la zone name ?"""
    return zone == name or (zone == "omega" and name in SIDES) \
        or (zone == "beta" and name.startswith("beta"))


def zone_bcs(mesh, bcs=None, g_beta=None):
    """Condition de chaque zone de mesh.zones : défauts, puis bcs, liste de (zone, BC) appliquée
    dans l'ordre (une condition donnée plus loin l'emporte)."""
    out = {name: BC(g=0.0 if name in SIDES or g_beta is None else g_beta) for name in mesh.zones}
    for zone, bc in bcs or []:
        names = [name for name in mesh.zones if zone_matches(zone, name)]
        if not names:
            raise ValueError(f"zone de bord inconnue : {zone} (zones de ce maillage : omega, beta, "
                             f"{', '.join(mesh.zones)})")
        out.update(dict.fromkeys(names, bc))
    return out


@dataclass
class System:
    A: object           # matrice complète (CSR), termes de Robin compris
    F: np.ndarray       # second membre complet, termes de Neumann et de Robin compris
    fixed: np.ndarray   # nœuds de Dirichlet (booléens)
    U0: np.ndarray      # valeurs imposées sur les nœuds de Dirichlet (0 ailleurs)
    bcs: dict           # condition de chaque zone (zone_bcs)
    a: object = 1.0     # coefficient de diffusion (flux des zones de Dirichlet, post.py)


def build_system(mesh, f, k=1.0, a=1.0, bcs=None, g_beta=None):
    """Assemble le système de -div(a grad u) + k u = f avec ses conditions aux limites."""
    if not callable(k) and k < 0:
        raise ValueError(f"k = {k:g} : k doit être positif ou nul (sinon A_II n'est plus définie "
                         "positive et la factorisation sans pivotage n'est plus sûre).")
    A, F = assemble(mesh, f, k, a)
    zbc = zone_bcs(mesh, bcs, g_beta)
    fixed, U0 = np.zeros(mesh.n_nodes, dtype=bool), np.zeros(mesh.n_nodes)
    for z, name in enumerate(mesh.zones):
        bc, edges = zbc[name], mesh.edges[mesh.edge_zone == z]
        if bc.kind == "dirichlet":
            nodes = np.unique(edges)
            U0[nodes] = evaluate(bc.g, *mesh.coords[nodes].T)
            fixed[nodes] = True
        else:
            Ab, Fb = assemble_boundary(mesh, edges, bc.alpha if bc.kind == "robin" else 0.0, bc.g)
            A, F = A + Ab, F + Fb
    if not fixed.any() and not callable(k) and k == 0 and all(bc.kind == "neumann" for bc in zbc.values()):
        raise ValueError("Neumann sur tout le bord avec k = 0 : u n'est définie qu'à une constante "
                         "près. Imposer u (dirichlet) ou une condition de Robin sur une zone, ou k > 0.")
    return System(A.tocsr(), F, fixed, U0, zbc, a)


def solve_system(system):
    """A_II U_I = F_I - A_IB G_B, puis U complété par U_B = G_B sur les nœuds de Dirichlet."""
    A, free = system.A, ~system.fixed
    U = system.U0.copy()
    rhs = system.F[free] - A[free][:, system.fixed] @ U[system.fixed]
    # A_II symétrique définie positive : renumérotation symétrique (minimum degree sur A + A^T)
    # et pivots pris sur la diagonale, environ 3 fois plus rapide que spsolve par défaut
    lu = splu(A[free][:, free].tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
              options=dict(SymmetricMode=True))
    U[free] = lu.solve(rhs)
    return U


def solve(mesh, f, g_beta=None, k=1.0, a=1.0, bcs=None):
    """Résout -div(a grad u) + k u = f ; sans bcs : u = 0 sur dOmega, u = g_beta sur dbeta."""
    return solve_system(build_system(mesh, f, k, a, bcs, g_beta))
