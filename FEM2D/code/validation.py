"""Validation : solution manufacturée, étude de convergence, erreur de chaque élément (section 6)."""
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.widgets import RadioButtons
import numpy as np

from assembly import element_errors, total_errors
from cases import CASES, DISK, application
from config import K, LX, LY, MeshParams, k_suffix, out_path
from geometry import disk
from mesh import build_mesh
from plots import draw_beta, has_display, plot_convergence, plot_solution, save_figure
from solver import solve

# u_ex = G S sur dbeta, G donnée avec son gradient et son laplacien
# (G = 1 : cas historique, G = 0 : condition homogène sur dbeta)
G_BETA = dict(G=lambda x, y: 1 + 3 * np.cos(x) * np.sin(y),
              grad=lambda x, y: (-3 * np.sin(x) * np.sin(y), 3 * np.cos(x) * np.cos(y)),
              lap=lambda x, y: -6 * np.cos(x) * np.sin(y))


# ----------------------------------------------------------------------------
# Solution manufacturée (section 6.1)
# ----------------------------------------------------------------------------
def manufactured(cx, cy, R, G, grad_G, lap_G, k=K):
    """u_ex = S (phi + G) : nulle sur dOmega, égale à G S sur dbeta (g_beta = u_ex).

    Avec p = phi + G : -Δu + k u = (k + a² + b²) S p - 2 ∇S.∇p - S Δp, où Δp = 4 + ΔG.
    """
    a, b = np.pi / LX, np.pi / LY

    def S(x, y):
        return np.sin(a * x) * np.sin(b * y)

    def grad_S(x, y):
        return a * np.cos(a * x) * np.sin(b * y), b * np.sin(a * x) * np.cos(b * y)

    def p(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 - R**2 + G(x, y)

    def grad_p(x, y):
        Gx, Gy = grad_G(x, y)
        return 2 * (x - cx) + Gx, 2 * (y - cy) + Gy

    def u_ex(x, y):
        return S(x, y) * p(x, y)

    def grad_u_ex(x, y):
        (Sx, Sy), (px, py) = grad_S(x, y), grad_p(x, y)
        return p(x, y) * Sx + S(x, y) * px, p(x, y) * Sy + S(x, y) * py

    def f(x, y):
        (Sx, Sy), (px, py) = grad_S(x, y), grad_p(x, y)
        return ((k + a**2 + b**2) * S(x, y) * p(x, y)
                - 2 * (Sx * px + Sy * py)
                - S(x, y) * (4 + lap_G(x, y)))

    return u_ex, grad_u_ex, f


def validation_problem(h, k=K):
    """Disque de validation résolu pour le pas h : (maillage, U, erreurs locales)."""
    obstacle = disk(DISK["center"], DISK["R"])
    u_ex, grad_u_ex, f = manufactured(*DISK["center"], DISK["R"], G_BETA["G"], G_BETA["grad"],
                                      G_BETA["lap"], k)
    mesh = build_mesh(obstacle, LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)
    U = solve(mesh, f, g_beta=u_ex, k=k)
    return mesh, U, element_errors(mesh, U, u_ex, grad_u_ex)


# ----------------------------------------------------------------------------
# Erreurs par élément
# ----------------------------------------------------------------------------
TYPE_LABEL = {"tri": "T", "quad": "Q"}
STATS_HEADER = f"{'type':>4} {'nombre':>7} {'moy. L2':>11} {'éc.-t. L2':>11} {'moy. H1':>11} {'éc.-t. H1':>11}"


def element_densities(area, l2, h1_semi):
    """Densités e_K / sqrt(|K|) (erreur moyenne quadratique sur K) en normes L2 et H1.

    Les triangles sont bien plus petits que les carrés : leurs erreurs brutes e_K ne sont
    pas comparables, leurs densités le sont.
    """
    return np.sqrt(l2 / area), np.sqrt((l2 + h1_semi) / area)


def element_error_stats(errs):
    """Moyenne et écart-type des densités d'erreur, par type d'élément.

    Renvoie {type: (nombre, moy. L2, éc.-type L2, moy. H1, éc.-type H1)}.
    """
    stats = {}
    for name, (area, l2, h1_semi, _) in errs.items():
        d_l2, d_h1 = element_densities(area, l2, h1_semi)
        stats[name] = (len(area), d_l2.mean(), d_l2.std(), d_h1.mean(), d_h1.std())
    return stats


def stats_row(name, stats):
    """Ligne du tableau STATS_HEADER pour un type d'élément."""
    n, m_l2, s_l2, m_h1, s_h1 = stats
    return f"{TYPE_LABEL[name]:>4} {n:7d} {m_l2:11.3e} {s_l2:11.3e} {m_h1:11.3e} {s_h1:11.3e}"


# ----------------------------------------------------------------------------
# Étude de convergence
# ----------------------------------------------------------------------------
# Chaque pas h tourne dans son propre processus : les fonctions ci-dessous renvoient
# leur résultat au lieu de l'afficher, pour que les sorties ne s'entremêlent pas.
HS = [0.5, 0.25, 0.125, 0.0625]


def validation_step(i, h, k=K):
    """Pas h numéro i de l'étude de convergence :
    (nombre de nœuds, erreur L2, erreur H1, statistiques par type d'élément)."""
    mesh, U, errs = validation_problem(h, k)
    l2, h1 = total_errors(errs)
    if i == 0:
        plot_solution(mesh, U, out_path("validation", f"solution{k_suffix(k)}.png"),
                      f"Validation : $u_h$ pour $h = {h}$, $k = {k:g}$")
    return mesh.n_nodes, l2, h1, element_error_stats(errs)


def validation_report(results, k=K):
    print(f"=== Validation : disque, solution manufacturée, k = {k:g} ===")
    e_l2, e_h1 = [], []
    print(f"{'h':>8} {'nœuds':>8} {'erreur L2':>12} {'ordre':>6} {'erreur H1':>12} {'ordre':>6}")
    for i, (h, (n_nodes, l2, h1, _)) in enumerate(zip(HS, results)):
        e_l2.append(l2)
        e_h1.append(h1)
        ratio_h = np.log(HS[i - 1] / h) if i else None
        p_l2 = f"{np.log(e_l2[-2] / l2) / ratio_h:6.2f}" if i else f"{'-':>6}"
        p_h1 = f"{np.log(e_h1[-2] / h1) / ratio_h:6.2f}" if i else f"{'-':>6}"
        print(f"{h:8.4f} {n_nodes:8d} {l2:12.4e} {p_l2} {h1:12.4e} {p_h1}")
    print("\nErreur par élément e_K / sqrt(|K|) : moyenne et écart-type, T = triangles, Q = carrés")
    print(f"{'h':>8} {STATS_HEADER}")
    for h, (*_, stats) in zip(HS, results):
        for name, s in stats.items():
            print(f"{h:8.4f} {stats_row(name, s)}")
    plot_convergence(HS, e_l2, e_h1, out_path("validation", f"convergence{k_suffix(k)}.png"))


def _worker_init():
    matplotlib.use("Agg")  # pas de fenêtre dans les processus de calcul


def run_validation(ex, k=K):
    # Le pas h le plus fin est le plus long : lancé en premier
    return [ex.submit(validation_step, i, h, k) for i, h in reversed(list(enumerate(HS)))][::-1]


def run_all(params=MeshParams(), k=K):
    """Validation et tous les cas prédéfinis en parallèle, sans validation du maillage."""
    with ProcessPoolExecutor(initializer=_worker_init) as ex:
        val = run_validation(ex, k)
        apps = [ex.submit(application, case, params, k) for case in CASES.values()]
        validation_report([fut.result() for fut in val], k)
        for fut in apps:
            print(fut.result())


def run_convergence(k=K):
    with ProcessPoolExecutor(initializer=_worker_init) as ex:
        validation_report([fut.result() for fut in run_validation(ex, k)], k)


# ----------------------------------------------------------------------------
# Erreur de chaque élément : carte cliquable
# ----------------------------------------------------------------------------
def element_polygons(mesh, name):
    """Contour de chaque élément passant par les nœuds milieux (arêtes courbes suivies)."""
    if name == "tri":
        return mesh.coords[mesh.tri[:, [0, 3, 1, 4, 2, 5]]]
    return mesh.coords[mesh.quad[:, [0, 4, 1, 5, 2, 6, 3, 7]]]


def zscore(d):
    """Écart à la moyenne en écarts-types ; 0 si l'écart-type est nul (un seul élément, erreur uniforme)."""
    std = d.std()
    return (d - d.mean()) / std if std > 0 else np.zeros_like(d)


def error_table(errs):
    """Grandeurs par élément, triangles puis carrés, dans un dict de tableaux (n_T + n_Q,).

    L2, H1 : densités e_K / sqrt(|K|) ; zL2, zH1 : écart à la moyenne du type de K, en
    écarts-types ; sd : écart-type de u_ex - u_h aux points de quadrature de K (pondéré).
    """
    cols = {k: [] for k in ("type", "index", "area", "eL2", "eH1", "L2", "H1", "zL2", "zH1", "sd")}
    for name, (area, l2, h1_semi, mean) in errs.items():
        d_l2, d_h1 = element_densities(area, l2, h1_semi)
        cols["type"].append(np.full(len(area), TYPE_LABEL[name]))
        cols["index"].append(np.arange(len(area)))
        cols["area"].append(area)
        cols["eL2"].append(np.sqrt(l2))
        cols["eH1"].append(np.sqrt(l2 + h1_semi))
        cols["L2"].append(d_l2)
        cols["H1"].append(d_h1)
        cols["zL2"].append(zscore(d_l2))
        cols["zH1"].append(zscore(d_h1))
        cols["sd"].append(np.sqrt(np.maximum(l2 / area - (mean / area) ** 2, 0)))
    return {k: np.concatenate(v) for k, v in cols.items()}


def describe_element(tab, k, center):
    t = tab["type"][k]
    return (f"{'triangle' if t == 'T' else 'carré'} {t}{tab['index'][k]}, centre "
            f"({center[0]:.3f}, {center[1]:.3f}), |K| = {tab['area'][k]:.3e}\n"
            f"e_K : L2 = {tab['eL2'][k]:.3e}, H1 = {tab['eH1'][k]:.3e}    "
            f"e_K / sqrt(|K|) : L2 = {tab['L2'][k]:.3e}, H1 = {tab['H1'][k]:.3e}\n"
            f"écart à la moyenne des {t} : L2 {tab['zL2'][k]:+.2f} σ, H1 {tab['zH1'][k]:+.2f} σ    "
            f"écart-type de u_ex - u_h dans K : {tab['sd'][k]:.3e}")


def show_errors(h, k=K):
    """Carte des densités d'erreur e_K / sqrt(|K|) du cas de validation pour le pas h.

    Un clic sur un élément affiche ses erreurs et son écart à la moyenne de son type.
    """
    mesh, _, errs = validation_problem(h, k)
    print(f"=== Erreur par élément : disque de validation, h = {h:g}, k = {k:g} ===")
    print(STATS_HEADER)
    for name, s in element_error_stats(errs).items():
        print(stats_row(name, s))
    tab = error_table(errs)
    polys = [p for name in errs for p in element_polygons(mesh, name)]
    centers = np.array([p.mean(axis=0) for p in polys])
    print("\nÉléments les plus éloignés de la moyenne de leur type (H1) :")
    for j in np.argsort(-tab["zH1"])[:5]:
        print(f"  {tab['type'][j]}{tab['index'][j]:<5} centre ({centers[j, 0]:.3f}, {centers[j, 1]:.3f})"
              f"  e_K/sqrt(|K|) = {tab['H1'][j]:.3e}  ({tab['zH1'][j]:+.2f} σ)")

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.2)
    coll = PolyCollection(polys, cmap="magma", norm=LogNorm(), edgecolors="0.35", linewidths=0.3)
    coll.set_array(tab["H1"])
    coll.set_picker(True)
    ax.add_collection(coll)
    ax.autoscale_view()
    draw_beta(ax, mesh)
    ax.set_aspect("equal")
    ax.set_title(f"Erreur par élément, h = {h:g} (cliquer sur un élément)")
    cbar = fig.colorbar(coll, ax=ax, shrink=0.8, label="$e_K / \\sqrt{|K|}$, norme $H^1$")
    marker = ax.add_patch(MplPolygon(polys[0], fill=False, ec="cyan", lw=2, zorder=5, visible=False))
    info = fig.text(0.16, 0.02, "", fontsize=8, family="monospace")

    path = out_path("validation", f"erreurs_elements{k_suffix(k)}.png")
    save_figure(fig, path)
    if not has_display():
        plt.close(fig)
        print(f"\nPas d'affichage disponible : carte enregistrée dans {path}")
        return

    def on_pick(event):
        if event.artist is not coll or not len(event.ind):
            return
        # La tolérance de sélection renvoie aussi les voisins : on garde l'élément cliqué
        xy = event.mouseevent.xdata, event.mouseevent.ydata
        inside = [k for k in event.ind if Path(polys[k]).contains_point(xy)]
        k = inside[0] if inside else event.ind[0]
        marker.set_xy(polys[k])
        marker.set_visible(True)
        info.set_text(describe_element(tab, k, centers[k]))
        fig.canvas.draw_idle()

    def on_norm(label):
        values = tab[label]
        coll.set_array(values)
        coll.norm.vmin, coll.norm.vmax = values.min(), values.max()
        cbar.set_label(f"$e_K / \\sqrt{{|K|}}$, norme ${label[0]}^{label[1]}$")
        cbar.update_normal(coll)
        fig.canvas.draw_idle()

    radio = RadioButtons(fig.add_axes([0.02, 0.03, 0.1, 0.1]), ("H1", "L2"))
    radio.on_clicked(on_norm)
    fig.canvas.mpl_connect("pick_event", on_pick)
    print("\nCarte affichée : cliquer sur un élément pour voir son erreur.")
    plt.show()
