"""Validation (solution manufacturée, étude de convergence) et cas d'application.

Sections 6 et 7 de main.tex. Lancer :  python3 main.py
Les figures sont écrites dans le dossier resultats/ à côté de ce fichier.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from assembly import compute_errors
from geometry import Polygon, disk, domain_from_obstacle, from_step, peanut
from mesh import build_mesh
from solver import solve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats")

# Paramètres par défaut (section 7.1)
LX, LY = 7.0, 5.0
H, H_BETA, DELTA, M_BETA = 0.5, 0.15, 0.8, 1
DISK = dict(center=(3.5, 2.5), R=1.0)
PEANUT = dict(center=(3.1, 2.63), r0=0.8, eps=0.35)
# Obstacle en L (non étoilé, avec un coin rentrant), sommets dans l'ordre du contour
POLYGON = [(2.0, 1.5), (2.0, 3.5), (3.0, 3.5), (3.0, 2.5), (5.0, 2.5), (5.0, 1.5)]
# Obstacle importé d'un fichier CAO (généré par cao/make_obstacle.py)
STEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cao", "obstacle.step")
PUZZLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cao", "Puzzle.step")
# beta non connexe : deux composantes proches (généré par cao/make_obstacle.py)
DEUX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cao", "deux_obstacles.step")
# Distance minimale entre beta et le bord de Omega, construit autour du polygone (section 3.5)
D_MIN = M_BETA + 4*H


# ----------------------------------------------------------------------------
# Solution manufacturée (section 6.1)
# ----------------------------------------------------------------------------
def manufactured(cx, cy, R):
    """u_ex = S (phi + 1) : nulle sur dOmega, égale à S sur dbeta (g_beta = u_ex)."""
    a, b = np.pi / LX, np.pi / LY

    def S(x, y):
        return np.sin(a * x) * np.sin(b * y)

    def phi(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 - R**2

    def u_ex(x, y):
        return S(x, y) * (phi(x, y) + 1)

    def grad_u_ex(x, y):
        Sx = a * np.cos(a * x) * np.sin(b * y)
        Sy = b * np.sin(a * x) * np.cos(b * y)
        p1 = phi(x, y) + 1
        return p1 * Sx + S(x, y) * 2 * (x - cx), p1 * Sy + S(x, y) * 2 * (y - cy)

    def f(x, y):
        return ((1 + a**2 + b**2) * S(x, y) * (phi(x, y) + 1)
                - 4 * (a * (x - cx) * np.cos(a * x) * np.sin(b * y)
                       + b * (y - cy) * np.sin(a * x) * np.cos(b * y))
                - 4 * S(x, y))

    return u_ex, grad_u_ex, f


# ----------------------------------------------------------------------------
# Figures (section 7.3)
# ----------------------------------------------------------------------------
TRI_SUB = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
_Q_GRID = [[0, 4, 1], [7, 8, 5], [3, 6, 2]]  # nœuds Q2 rangés par lignes (y^ = -1, 0, 1)
QUAD_SUB = []
for _b in range(2):
    for _a in range(2):
        p, q = _Q_GRID[_b][_a], _Q_GRID[_b][_a + 1]
        r, s = _Q_GRID[_b + 1][_a + 1], _Q_GRID[_b + 1][_a]
        QUAD_SUB += [[p, q, r], [p, r, s]]


def plot_triangulation(mesh):
    """Sous-triangles P1 utilisant les nœuds milieux (4 par triangle P2, 8 par carré Q2)."""
    parts = [mesh.tri[:, s] for s in TRI_SUB] + [mesh.quad[:, s] for s in QUAD_SUB]
    return Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], np.vstack(parts))


def draw_beta(ax, mesh):
    for xy in mesh.beta_points:
        pts = np.vstack((xy, xy[:1]))
        ax.fill(pts[:, 0], pts[:, 1], color="0.8", zorder=3)
        ax.plot(pts[:, 0], pts[:, 1], "k", lw=1.2, zorder=4)


def plot_mesh(mesh, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    P = mesh.coords
    for v in mesh.quad[:, [0, 1, 2, 3, 0]]:
        ax.plot(P[v, 0], P[v, 1], color="0.55", lw=0.6)
    # Triangles tracés avec leurs milieux : les arêtes posées sur beta apparaissent courbes
    for v in mesh.tri[:, [0, 3, 1, 4, 2, 5, 0]]:
        ax.plot(P[v, 0], P[v, 1], color="tab:blue", lw=0.6)
    draw_beta(ax, mesh)
    ax.set_aspect("equal")
    ax.set_title(f"Maillage : {len(mesh.quad)} quadrilatères Q2, {len(mesh.tri)} triangles P2")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_solution(mesh, U, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.tripcolor(plot_triangulation(mesh), U, shading="gouraud", cmap="viridis")
    draw_beta(ax, mesh)
    fig.colorbar(im, ax=ax, shrink=0.8, label="$u_h$")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(hs, e_l2, e_h1, path):
    hs, e_l2, e_h1 = map(np.asarray, (hs, e_l2, e_h1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, e_l2, "o-", label="erreur $L^2$")
    ax.loglog(hs, e_h1, "s-", label="erreur $H^1$")
    ax.loglog(hs, e_l2[-1] * (hs / hs[-1]) ** 3, "k--", lw=0.8, label="pente 3")
    ax.loglog(hs, e_h1[-1] * (hs / hs[-1]) ** 2, "k:", lw=0.8, label="pente 2")
    ax.set_xlabel("$h$")
    ax.set_ylabel("erreur")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def validation():
    print("=== Validation : disque, solution manufacturée ===")
    obstacle = disk(DISK["center"], DISK["R"])
    u_ex, grad_u_ex, f = manufactured(*DISK["center"], DISK["R"])
    hs = [0.5, 0.25, 0.125, 0.0625]
    e_l2, e_h1 = [], []
    print(f"{'h':>8} {'nœuds':>8} {'erreur L2':>12} {'ordre':>6} {'erreur H1':>12} {'ordre':>6}")
    for k, h in enumerate(hs):
        mesh = build_mesh(obstacle, LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)
        U = solve(mesh, f, g_beta=u_ex)
        l2, h1 = compute_errors(mesh, U, u_ex, grad_u_ex)
        e_l2.append(l2)
        e_h1.append(h1)
        p_l2 = f"{np.log(e_l2[-2] / l2) / np.log(2):6.2f}" if k else f"{'-':>6}"
        p_h1 = f"{np.log(e_h1[-2] / h1) / np.log(2):6.2f}" if k else f"{'-':>6}"
        print(f"{h:8.4f} {mesh.n_nodes:8d} {l2:12.4e} {p_l2} {h1:12.4e} {p_h1}")
        if k == 0:
            plot_solution(mesh, U, os.path.join(OUT, "validation_solution.png"),
                          f"Validation : $u_h$ pour $h = {h}$")
    plot_convergence(hs, e_l2, e_h1, os.path.join(OUT, "convergence.png"))


def application(name, obstacle, suffix="", Lx=LX, Ly=LY, g_beta=None):
    print(f"\n=== Application : {name}, f = 1, Omega = [0, {Lx}] x [0, {Ly}] ===")
    mesh = build_mesh(obstacle, Lx, Ly, H, H_BETA, DELTA, M_BETA)
    print(f"{mesh.n_nodes} nœuds, {len(mesh.tri)} triangles, {len(mesh.quad)} quadrilatères, "
          f"{len(mesh.bnd)} nœuds de bord")
    U = solve(mesh, lambda x, y: np.ones_like(x), g_beta)
    print(f"min u_h = {U.min():.4e}, max u_h = {U.max():.4e}")
    plot_mesh(mesh, os.path.join(OUT, f"maillage{suffix}.png"))
    title = "Solution $u_h$ pour $f = 1$" + (r", $u = g_\beta$ sur $\partial\beta$" if g_beta else "")
    plot_solution(mesh, U, os.path.join(OUT, f"solution{suffix}.png"), title)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    validation()
    cacahuete, Lx, Ly, _ = domain_from_obstacle(peanut(PEANUT["center"], PEANUT["r0"], PEANUT["eps"]),
                                                H, D_MIN, M_BETA)
    application("cacahuète", cacahuete, Lx=Lx, Ly=Ly)
    application("cacahuète, u = g_beta = 1 + x/Lx sur dbeta", cacahuete, suffix="_dirichlet", Lx=Lx, Ly=Ly,
                g_beta=lambda x, y: 1 + x / Lx)
    polygon, Lx, Ly, _ = domain_from_obstacle(Polygon(POLYGON), H, D_MIN, M_BETA)
    application("polygone", polygon, suffix="_polygone", Lx=Lx, Ly=Ly,g_beta=lambda x,y : np.exp(0.1*(x-y)))
    step, Lx, Ly, _ = domain_from_obstacle(from_step(STEP), H, D_MIN, M_BETA)
    application("fichier STEP", step, suffix="_step", Lx=Lx, Ly=Ly)
    puzzle, Lx, Ly, _ = domain_from_obstacle(from_step(PUZZLE), H, D_MIN, M_BETA)
    application("pièce de puzzle (Puzzle.step)", puzzle, suffix="_puzzle", Lx=Lx, Ly=Ly)
    deux, Lx, Ly, _ = domain_from_obstacle(from_step(DEUX), H, D_MIN, M_BETA)
    application("beta non connexe (deux_obstacles.step)", deux, suffix="_non_connexe", Lx=Lx, Ly=Ly)
    print(f"\nFigures écrites dans {OUT}")
