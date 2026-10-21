"""Cas d'application : obstacles prédéfinis, obstacles composés, maillage et résolution d'un cas."""
import hashlib
import os
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Optional

from matplotlib.path import Path
import numpy as np

from config import HERE, K, SIGMA, k_suffix, out_path
from geometry import (MultiObstacle, Polygon, annulus, disk, domain_from_obstacle, from_step,
                      peanut, union)
from mesh import build_mesh
from plots import plot_mesh, plot_solution
from solver import solve

DISK = dict(center=(3.5, 2.5), R=1.0)
PEANUT = dict(center=(3.1, 2.63), r0=0.8, eps=0.35)
# Obstacle en L (non étoilé, avec un coin rentrant), sommets dans l'ordre du contour
POLYGON = [(2.0, 1.5), (2.0, 3.5), (3.0, 3.5), (3.0, 2.5), (5.0, 2.5), (5.0, 1.5)]
# Obstacles importés de fichiers CAO (obstacle.step, deux_obstacles.step : cao/make_obstacle.py)
STEP = os.path.join(HERE, "cao", "obstacle.step")
PUZZLE = os.path.join(HERE, "cao", "Puzzle.step")
DEUX = os.path.join(HERE, "cao", "deux_obstacles.step")  # beta non connexe, composantes proches
TANGENTS = os.path.join(HERE, "cao", "disques_tangents.step")  # deux disques à 0.05 l'un de l'autre


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable                     # () -> obstacle, en coordonnées quelconques
    folder: str                        # sous-dossier de resultats/ (un par obstacle)
    suffix: str = ""                   # suffixe des figures solution*.png (variante du cas)
    g_beta: Optional[Callable] = None  # (Lx, Ly) -> g_beta ; None : u = 0 sur dbeta
    f: Optional[Callable] = None       # (Lx, Ly) -> (f, titre LaTeX) ; None : f = 1


# Fonctions du module (et non des lambda) : les cas passent d'un processus à l'autre
def g_dirichlet(Lx, Ly):
    return lambda x, y: 1 + x / Lx


def g_polygon(Lx, Ly):
    return lambda x, y: np.exp(0.1 * (x - y))


def f_gauss(Lx, Ly, sigma=SIGMA, center=None):
    """Source gaussienne f(x0, y0) = 1, centrée en center (coordonnées de Omega = [0, Lx] x [0, Ly]),
    ou par défaut en (Lx/4, 3Ly/4), hors de beta pour les cas centrés."""
    x0, y0 = center if center is not None else (Lx / 4, 3 * Ly / 4)
    return (lambda x, y: np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)),
            rf"$f = e^{{-|x - x_0|^2 / 2\sigma^2}}$, $x_0 = ({x0:g}, {y0:g})$, $\sigma = {sigma:g}$")


def with_gauss(case, x0, y0, sigma=SIGMA):
    """Le même cas avec f gaussienne centrée en (x0, y0) ; suffixe de la solution complété."""
    return replace(case, f=partial(f_gauss, sigma=sigma, center=(x0, y0)),
                   suffix=f"{case.suffix}_gauss_{x0:g}_{y0:g}_s{sigma:g}")


def gauss_center(case):
    """Centre (x0, y0) choisi pour la source gaussienne du cas, None s'il n'y en a pas."""
    return case.f.keywords.get("center") if isinstance(case.f, partial) else None


def center_error(mesh, Lx, Ly, x0, y0):
    """Message d'erreur si (x0, y0) n'est pas dans D = Omega - beta (beta tel que maillé), None sinon."""
    if not (0 <= x0 <= Lx and 0 <= y0 <= Ly):
        return f"le point ({x0:g}, {y0:g}) est hors de Omega = [0, {Lx:g}] x [0, {Ly:g}]"
    # Dans beta : à l'intérieur d'un nombre impair de courbes de dbeta (les cavités sont dans D)
    if sum(Path(xy).contains_point((x0, y0)) for xy in mesh.beta_points) % 2:
        return f"le point ({x0:g}, {y0:g}) est dans l'obstacle beta"
    return None


def obstacles_eloignes():
    """beta non connexe, composantes assez éloignées pour être dans des zones T distinctes."""
    return MultiObstacle([disk((0.0, 0.0), 0.8),
                          peanut((6.0, 1.0), 0.7, 0.35),
                          Polygon([(1.0, 4.0), (1.0, 6.0), (2.0, 6.0), (2.0, 5.0), (4.0, 5.0), (4.0, 4.0)])])


def couronnes_secantes():
    """Deux couronnes dont les cavités se coupent : leur réunion a trois cavités."""
    return union([annulus((0.0, 0.0), 2.0, 1.4), annulus((1.5, 0.0), 2.0, 1.4)])


_PEANUT = partial(peanut, PEANUT["center"], PEANUT["r0"], PEANUT["eps"])
CASES = {
    "disque": Case("disque", partial(disk, DISK["center"], DISK["R"]), "disque"),
    "cacahuete": Case("cacahuète", _PEANUT, "cacahuete"),
    "cacahuete_dirichlet": Case("cacahuète, u = g_beta = 1 + x/Lx sur dbeta", _PEANUT,
                                "cacahuete", "_dirichlet", g_dirichlet),
    "polygone": Case("polygone", partial(Polygon, POLYGON), "polygone", g_beta=g_polygon),
    "step": Case("fichier STEP", partial(from_step, STEP), "step"),
    "puzzle": Case("pièce de puzzle (Puzzle.step)", partial(from_step, PUZZLE), "puzzle"),
    "puzzle_gauss": Case("pièce de puzzle (Puzzle.step), f gaussienne", partial(from_step, PUZZLE),
                         "puzzle", "_gauss", f=f_gauss),
    "non_connexe": Case("beta non connexe (deux_obstacles.step)", partial(from_step, DEUX),
                        "non_connexe"),
    "tangents": Case("deux disques presque tangents (disques_tangents.step)", partial(from_step, TANGENTS),
                     "tangents"),
    "couronne": Case("couronne (beta troué : cavité de D séparée du reste)",
                     partial(annulus, DISK["center"], 1.5, 0.8), "couronne"),
    "fusion": Case("deux couronnes sécantes (réunion : trois cavités)", couronnes_secantes, "fusion"),
    "eloignes": Case("trois obstacles éloignés, u = g_beta = 1 + x/Lx sur dbeta",
                     obstacles_eloignes, "eloignes", g_beta=g_dirichlet),
}


def step_file_case(path, scale=1.0):
    """Dossier = nom du fichier en minuscules (Puzzle.step -> puzzle, comme le cas prédéfini),
    suivi de l'échelle si elle n'est pas 1."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    folder = stem + (f"_x{scale:g}" if scale != 1 else "")
    return Case(f"fichier {os.path.basename(path)}", partial(from_step, path, scale), folder)


# ----------------------------------------------------------------------------
# beta composé : type et paramètres choisis pour chaque composante
# ----------------------------------------------------------------------------
# type -> (description, paramètres numériques, valeurs par défaut)
KINDS = {
    "disque": ("disque de centre (x, y) et de rayon R", ["x", "y", "R"], [0.0, 0.0, 1.0]),
    "cacahuete": ("cacahuète r = r0 (1 + eps cos 2t), centre (x, y)", ["x", "y", "r0", "eps"],
                  [0.0, 0.0, 0.8, 0.35]),
    "polygone": ("polygone (en L par défaut) translaté de (x, y)", ["x", "y"], [0.0, 0.0]),
    "couronne": ("couronne r <= |x - (x, y)| <= R (beta troué)", ["x", "y", "R", "r"],
                 [0.0, 0.0, 1.5, 0.8]),
    "fichier": ("fichier STEP / IGES / BREP translaté de (x, y)", ["x", "y", "echelle"], [0.0, 0.0, 1.0]),
}
L_SHAPE = tuple(v for xy in np.array(POLYGON) - POLYGON[0] for v in xy)


def polygon_vertices_ok(vertices):
    """Sommets x1 y1 x2 y2 ... d'un polygone : aucun (forme en L) ou au moins trois sommets."""
    return not vertices or (len(vertices) % 2 == 0 and len(vertices) >= 6)


def check_spec(spec):
    """Vérifie le type et le nombre de paramètres d'une composante, avant tout maillage.

    Lève ValueError avec un message explicite (au lieu d'une erreur de dépaquetage
    qui apparaîtrait plus tard comme un échec du maillage).
    """
    kind, args = spec[0], list(spec[1:])
    if kind not in KINDS:
        raise ValueError(f"type d'obstacle inconnu : {kind} (choix : {', '.join(KINDS)})")
    names = KINDS[kind][1]
    if kind == "fichier":
        if not args or not os.path.isfile(os.path.expanduser(args[0])):
            raise ValueError(f"fichier : chemin introuvable : {args[0] if args else '(absent)'}")
        args = args[1:]
    try:
        values = [float(a) for a in args]
    except ValueError:
        raise ValueError(f"{kind} : paramètres numériques attendus, reçu {' '.join(map(str, args))}")
    if kind == "fichier":
        ok = len(names) - 1 <= len(values) <= len(names)
    elif kind == "polygone":
        ok = len(values) >= 2 and polygon_vertices_ok(values[2:])
    else:
        ok = len(values) == len(names)
    if not ok:
        usage = {"fichier": "chemin x y [echelle]", "polygone": "x y [x1 y1 x2 y2 x3 y3 ...]"}
        raise ValueError(f"{kind} : paramètres attendus {usage.get(kind, ' '.join(names))}, "
                         f"reçu {' '.join(map(str, args)) or 'aucun'}")


def make_component(kind, *args):
    """Composantes de beta décrites par (type, paramètres) : liste d'obstacles connexes.

    disque x y R | cacahuete x y r0 eps | couronne x y R r | polygone x y [x1 y1 x2 y2 ...] |
    fichier chemin x y [echelle]
    """
    if kind == "fichier":
        path, args = args[0], [float(a) for a in args[1:]]
        x, y, scale = args + KINDS["fichier"][2][len(args):]
        obstacle = from_step(os.path.expanduser(path), scale).translated(np.array([x, y]))
        return getattr(obstacle, "components", [obstacle])
    args = [float(a) for a in args]
    if kind == "disque":
        x, y, R = args
        return [disk((x, y), R)]
    if kind == "cacahuete":
        x, y, r0, eps = args
        return [peanut((x, y), r0, eps)]
    if kind == "couronne":
        x, y, R, r = args
        return [annulus((x, y), R, r)]
    if kind == "polygone":
        (x, y), vertices = args[:2], args[2:]
        if not polygon_vertices_ok(vertices):
            raise ValueError("Polygone : au moins trois sommets x1 y1 x2 y2 x3 y3.")
        return [Polygon(np.reshape(vertices or L_SHAPE, (-1, 2)) + (x, y))]
    raise ValueError(f"Type d'obstacle inconnu : {kind} (choix : {', '.join(KINDS)}).")


def compose(specs):
    """beta = réunion des composantes décrites par specs, une suite de (type, *paramètres) ;
    celles qui se chevauchent sont fusionnées (union)."""
    return union([c for spec in specs for c in make_component(*spec)])


def composed_case(specs):
    """Dossier compose/<types>_<empreinte des paramètres> : deux compositions distinctes
    n'écrivent pas dans les mêmes figures."""
    specs = tuple(tuple(spec) for spec in specs)
    kinds = [spec[0] for spec in specs]
    digest = hashlib.sha1(repr(specs).encode()).hexdigest()[:6]
    return Case(f"obstacles composés ({', '.join(kinds)})", partial(compose, specs),
                os.path.join("compose", f"{'_'.join(kinds)}_{digest}"))


# ----------------------------------------------------------------------------
# Maillage et résolution d'un cas
# ----------------------------------------------------------------------------
def prepare(case, params):
    """Omega construit autour de beta, puis maillage : (mesh, Lx, Ly)."""
    obstacle, Lx, Ly, _ = domain_from_obstacle(case.make(), params.h, params.d_min, params.m_beta)
    mesh = build_mesh(obstacle, Lx, Ly, params.h, params.h_beta, params.delta, params.m_beta,
                      params.raffinement)
    return mesh, Lx, Ly


def mesh_summary(mesh, Lx, Ly):
    return (f"Omega = [0, {Lx:g}] x [0, {Ly:g}] : {mesh.n_nodes} nœuds, {len(mesh.tri)} triangles, "
            f"{len(mesh.quad)} quadrilatères, {len(mesh.bnd)} nœuds de bord")


def compute(case, mesh, Lx, Ly, k=K):
    """Assemblage, résolution de -Δu + k u = f et figures, sur un maillage déjà construit."""
    g_beta = case.g_beta(Lx, Ly) if case.g_beta else None
    f, f_title = case.f(Lx, Ly) if case.f else (lambda x, y: np.ones_like(x), "$f = 1$")
    U = solve(mesh, f, g_beta, k)
    plot_mesh(mesh, out_path(case.folder, "maillage.png"))
    title = (f"Solution $u_h$ pour {f_title}" + (f", $k = {k:g}$" if k != K else "")
             + (r", $u = g_\beta$ sur $\partial\beta$" if g_beta else ""))
    plot_solution(mesh, U, out_path(case.folder, f"solution{case.suffix}{k_suffix(k)}.png"), title)
    return f"min u_h = {U.min():.4e}, max u_h = {U.max():.4e}"


def application(case, params, k=K):
    """Cas complet sans validation du maillage (mode « tous »)."""
    mesh, Lx, Ly = prepare(case, params)
    return "\n".join([f"\n=== Application : {case.name}, k = {k:g} ===",
                      mesh_summary(mesh, Lx, Ly), compute(case, mesh, Lx, Ly, k)])
