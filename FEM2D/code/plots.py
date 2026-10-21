"""Figures : maillage, solution, convergence (section 7.3 de main.tex)."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.tri import Triangulation
import numpy as np

# Backends matplotlib sans fenêtre : les figures sont seulement écrites sur disque
NON_INTERACTIVE = {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}


def has_display():
    """Vrai si matplotlib peut ouvrir une fenêtre (backend interactif)."""
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE


def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")


TRI_SUB = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
_Q_GRID = [[0, 4, 1], [7, 8, 5], [3, 6, 2]]  # nœuds Q2 rangés par lignes (y^ = -1, 0, 1)
# Chaque sous-carré (i, j) du carré Q2, de sommets p q r s, coupé selon la diagonale p r
QUAD_SUB = [tri
            for i in range(2) for j in range(2)
            for p, q, r, s in [(_Q_GRID[i][j], _Q_GRID[i][j + 1], _Q_GRID[i + 1][j + 1], _Q_GRID[i + 1][j])]
            for tri in ([p, q, r], [p, r, s])]


def plot_triangulation(mesh):
    """Sous-triangles P1 utilisant les nœuds milieux (4 par triangle P2, 8 par carré Q2)."""
    parts = [mesh.tri[:, s] for s in TRI_SUB] + [mesh.quad[:, s] for s in QUAD_SUB]
    return Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], np.vstack(parts))


def draw_beta(ax, mesh):
    """beta en gris : chaque courbe de dbeta est orientée selon la parité du nombre de courbes
    qui l'entourent (bord extérieur direct, bord de trou indirect), et le chemin composé est
    rempli avec la règle non nulle, ce qui laisse les cavités vides."""
    loops = [np.vstack((xy, xy[:1])) for xy in mesh.beta_points]
    parts = []
    for k, pts in enumerate(loops):
        depth = sum(Path(other).contains_point(pts[0]) for j, other in enumerate(loops) if j != k)
        area = pts[:-1, 0] @ pts[1:, 1] - pts[:-1, 1] @ pts[1:, 0]
        parts.append(pts if (area > 0) == (depth % 2 == 0) else pts[::-1])
        ax.plot(pts[:, 0], pts[:, 1], "k", lw=1.2, zorder=4)
    ax.add_patch(PathPatch(Path.make_compound_path(*map(Path, parts)), facecolor="0.8",
                           edgecolor="none", zorder=3))


def mesh_figure(mesh):
    fig, ax = plt.subplots(figsize=(8, 6))
    P = mesh.coords
    ax.add_collection(LineCollection(P[mesh.quad[:, [0, 1, 2, 3, 0]]], colors="0.55", lw=0.6))
    # Triangles tracés avec leurs milieux : les arêtes posées sur beta apparaissent courbes
    ax.add_collection(LineCollection(P[mesh.tri[:, [0, 3, 1, 4, 2, 5, 0]]], colors="tab:blue", lw=0.6))
    ax.autoscale_view()
    draw_beta(ax, mesh)
    ax.set_aspect("equal")
    ax.set_title(f"Maillage : {len(mesh.quad)} quadrilatères Q2, {len(mesh.tri)} triangles P2")
    return fig


def plot_mesh(mesh, path):
    fig = mesh_figure(mesh)
    save_figure(fig, path)
    plt.close(fig)


def plot_solution(mesh, U, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.tripcolor(plot_triangulation(mesh), U, shading="gouraud", cmap="viridis")
    draw_beta(ax, mesh)
    fig.colorbar(im, ax=ax, shrink=0.8, label="$u_h$")
    ax.set_aspect("equal")
    ax.set_title(title)
    save_figure(fig, path)
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
    save_figure(fig, path)
    plt.close(fig)
