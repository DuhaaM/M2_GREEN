"""Raffinement a priori du maillage près de beta, d'après la géométrie seule.

Taille visée en un point de dbeta, bornée par h_beta (r > 0 : intensité du raffinement,
paramètre raffinement ; r = 0 : pas de raffinement, maillage inchangé) :
    interstice : d / (N_GAP r), d distance à la partie de dbeta qui fait face à travers D
                 (autre composante, ou même composante après un détour, ex. fond d'une fente) ;
    courbure   : THETA / (r |kappa|), angle sous-tendu par un élément de dbeta <= THETA / r
                 (désactivé par défaut, CURVATURE = False : sur les cas testés, en norme
                 d'énergie, moins rentable que de réduire h) ;
    coin       : h_beta / (CORNER r) aux coins où l'angle de D dépasse pi (u y est singulière,
                 en rho^(pi/omega)).
Ces tailles sont des sources ponctuelles (p_j, s_j). La taille en x vaut
    min_j (s_j + GRADING |x - p_j|) :
elle croît d'au plus GRADING par unité de longueur en s'éloignant des sources. Elle place les
nœuds de dbeta (graded_params) et, dans la zone T, s'ajoute au champ de gmsh (mesh.triangulate).
"""
import numpy as np
from scipy.spatial import cKDTree

N_GAP = 3                       # éléments dans la largeur d'un interstice (r = 1)
CURVATURE = False               # critère de courbure (désactivé par défaut)
THETA = 0.3                     # angle (rad) sous-tendu par un élément de dbeta (r = 1)
CORNER = 4                      # h_beta / taille au sommet d'un coin singulier (r = 1)
CORNER_ANGLE = np.radians(15)   # déviation minimale de la tangente pour qu'une jonction soit un coin
GRADING = 0.25                  # croissance maximale de la taille par unité de longueur
FACING = 2.0                    # p et q se font face si leur distance le long de dbeta dépasse
                                # FACING |p - q| (jamais sur un disque : au plus pi/2 |p - q|)
CURVATURE_STENCIL = 2           # kappa calculé entre les échantillons i - 2 et i + 2
K_NEAREST = 8                   # sources consultées pour évaluer la taille en un point


def _piece(comp, ta, tb, n=2000):
    """Échantillons fins de gamma sur [ta, tb] : (t, points, abscisse curviligne)."""
    t = np.linspace(ta, tb, n + 1)
    p = comp.gamma(t)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    return t, p, s


def graded_params(comp, size):
    """Paramètres t_i des nœuds de dbeta, au pas local size(gamma(t)) ; les points de cassure
    (sommets, extrémités des courbes CAO) restent des nœuds. Avec size = h_beta constant :
    même répartition que comp.boundary_params(h_beta) pour un polygone ou une courbe CAO."""
    t_all = []
    bp = comp.breakpoints
    for ta, tb in zip(bp[:-1], bp[1:]):
        t, p, s = _piece(comp, ta, tb)
        N = np.concatenate(([0.0], np.cumsum(np.diff(s) / size(0.5 * (p[1:] + p[:-1])))))
        n = max(1, int(np.ceil(N[-1] - 1e-9)))
        t_all.append(np.interp(np.arange(n) * N[-1] / n, N, t))
    return np.concatenate(t_all)


def _loop_samples(comp, ds):
    """Échantillons de dbeta au pas ~ds, chaque morceau commençant par son point de cassure :
    (points (n, 2), indices des points de cassure)."""
    pieces, starts = [], [0]
    bp = comp.breakpoints
    for ta, tb in zip(bp[:-1], bp[1:]):
        t, _, s = _piece(comp, ta, tb)
        n = max(4, int(np.ceil(s[-1] / ds)))
        pieces.append(comp.gamma(np.interp(np.linspace(0, s[-1], n + 1), s, t))[:-1])
        starts.append(starts[-1] + n)
    return np.vstack(pieces), np.array(starts[:-1])


def _turn(a, b):
    """Angle signé de a vers b (> 0 : virage à gauche)."""
    return np.arctan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], np.einsum("...d,...d->...", a, b))


class SizeField:
    """Taille x -> min_j (s_j + GRADING |x - p_j|), parmi les K_NEAREST sources les plus proches.

    Les s_j sont d'abord gradués entre eux (s_j <- min_i (s_i + GRADING |p_i - p_j|)) : la
    restriction aux sources proches ne sous-estime alors la vraie taille de pas plus que
    2 GRADING fois la distance à la source la plus proche.
    """

    def __init__(self, points, sizes):
        graded = np.empty_like(sizes)
        for a in range(0, len(points), 1000):
            d = np.linalg.norm(points[a:a + 1000, None] - points[None], axis=-1)
            graded[a:a + 1000] = (sizes[None] + GRADING * d).min(axis=1)
        self.points, self.sizes = points, graded
        self.tree = cKDTree(points)
        self.k = min(K_NEAREST, len(points))

    def __call__(self, pts):
        d, j = self.tree.query(np.atleast_2d(pts), k=self.k)
        if self.k == 1:
            d, j = d[:, None], j[:, None]
        return (self.sizes[j] + GRADING * d).min(axis=1)


def size_field(comps, inside, h_beta, r):
    """Sources de raffinement de dbeta (composantes comps, indicatrice inside de beta).

    Renvoie un SizeField, ou None si r = 0 ou si aucune taille n'est inférieure à h_beta.
    """
    if r <= 0:
        return None
    ds = h_beta / 10
    points, sizes, comp_of, arc, loop_length = [], [], [], [], []
    for c, comp in enumerate(comps):
        P, starts = _loop_samples(comp, ds)
        n = len(P)
        step = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        s = np.concatenate(([0.0], np.cumsum(step[:-1])))
        x, y = P.T
        orientation = np.sign(x @ np.roll(y, -1) - y @ np.roll(x, -1))  # +1 : sens direct
        size = np.full(n, np.inf)

        # Courbure, hors du voisinage des points de cassure (traités comme coins éventuels)
        if CURVATURE:
            m = CURVATURE_STENCIL
            a, b = P - np.roll(P, m, axis=0), np.roll(P, -m, axis=0) - P
            kappa = np.abs(_turn(a, b)) / (0.5 * (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)))
            near_break = np.zeros(n, dtype=bool)
            for k in range(-m, m + 1):
                near_break[(starts + k) % n] = True
            kappa[near_break] = 0.0
            with np.errstate(divide="ignore"):
                size = np.minimum(size, THETA / (r * kappa))

        # Coins singuliers : virage vers beta (angle de D = pi + virage > pi)
        turn = orientation * _turn(P[starts] - P[starts - 1], P[(starts + 1) % n] - P[starts])
        size[starts[turn > CORNER_ANGLE]] = np.minimum(size[starts[turn > CORNER_ANGLE]],
                                                      h_beta / (CORNER * r))

        points.append(P)
        sizes.append(size)
        comp_of.append(np.full(n, c))
        arc.append(s)
        loop_length.append(np.full(n, s[-1] + step[-1]))
    P, size = np.vstack(points), np.concatenate(sizes)
    comp_of, arc, loop_length = map(np.concatenate, (comp_of, arc, loop_length))

    # Interstices : paires de points proches qui se font face à travers D
    pairs = cKDTree(P).query_pairs(N_GAP * r * h_beta, output_type="ndarray")
    if len(pairs):
        i, j = pairs.T
        dist = np.linalg.norm(P[i] - P[j], axis=1)
        along = np.abs(arc[i] - arc[j])
        along = np.minimum(along, loop_length[i] - along)
        facing = (comp_of[i] != comp_of[j]) | (along > FACING * dist)
        i, j, dist = i[facing], j[facing], dist[facing]
        across_d = ~inside(0.5 * (P[i] + P[j]))  # le segment pq traverse D, pas beta
        i, j, dist = i[across_d], j[across_d], dist[across_d]
        for side in (i, j):
            np.minimum.at(size, side, dist / (N_GAP * r))

    sources = size < h_beta
    return SizeField(P[sources], size[sources]) if sources.any() else None
