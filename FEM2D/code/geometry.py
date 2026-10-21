"""Obstacle beta décrit en polaire autour d'un centre c (section 3.1 de main.tex).

    beta = { x : rho(x) <= r(theta(x)) },   gamma(t) = c + r(t) (cos t, sin t)

ou par un polygone (classe Polygon), gamma étant alors paramétré par l'abscisse
curviligne ramenée à [0, 2pi).
"""
import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree


class Obstacle:
    def __init__(self, center, radius, n_samples=2000):
        self.c = np.asarray(center, dtype=float)
        self.radius = radius  # fonction theta -> r(theta), 2pi-périodique
        # Échantillonnage fin de gamma, utilisé pour la distance et l'abscisse curviligne
        self.t_fine = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        # r > 0 : beta est étoilé par rapport à c et gamma est une courbe simple
        if np.any(radius(self.t_fine) <= 0):
            raise ValueError("Obstacle polaire : r(theta) doit être strictement positif.")
        self.pts_fine = self.gamma(self.t_fine)
        self.vertices = self.pts_fine  # échantillons de gamma (boîte englobante de beta)
        self.tree = cKDTree(self.pts_fine)
        self.breakpoints = np.array([0.0, 2 * np.pi])  # gamma régulière : aucun coin

    def gamma(self, t):
        t = np.asarray(t, dtype=float)
        r = self.radius(t)
        return np.column_stack((self.c[0] + r * np.cos(t), self.c[1] + r * np.sin(t)))

    def polar(self, x, y):
        dx, dy = x - self.c[0], y - self.c[1]
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def inside(self, pts):
        rho, theta = self.polar(pts[:, 0], pts[:, 1])
        return rho <= self.radius(theta)

    def level(self, x, y):
        """phi_beta = rho^2 - r(theta)^2 : < 0 dans beta, = 0 sur le bord, > 0 dans D."""
        rho, theta = self.polar(x, y)
        return rho**2 - self.radius(theta) ** 2

    def boundary_distance(self, pts):
        """Distance à gamma (aux échantillons fins)."""
        return self.tree.query(pts)[0]

    def distance(self, pts):
        """d(x, beta), nulle dans beta."""
        d = self.boundary_distance(pts)
        d[self.inside(pts)] = 0.0
        return d

    def boundary_params(self, h_beta):
        """Paramètres t_i de n_beta points répartis selon l'abscisse curviligne."""
        closed = np.vstack((self.pts_fine, self.pts_fine[:1]))
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
        n_beta = int(np.ceil(s[-1] / h_beta))
        s_target = np.arange(n_beta) * s[-1] / n_beta
        t_closed = np.append(self.t_fine, 2 * np.pi)
        return np.interp(s_target, s, t_closed)

    def translated(self, shift):
        return Obstacle(self.c + shift, self.radius, len(self.t_fine))


def disk(center, R):
    return Obstacle(center, lambda t: R + 0.0 * np.asarray(t))


def peanut(center, r0, eps):
    return Obstacle(center, lambda t: r0 * (1.0 + eps * np.cos(2 * np.asarray(t))))


class Polygon:
    """Obstacle polygonal simple, donné par la liste de ses sommets (sans répéter le premier)."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        closed = np.vstack((self.vertices, self.vertices[:1]))
        self.a, self.b = closed[:-1], closed[1:]  # extrémités des segments
        lengths = np.linalg.norm(self.b - self.a, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("Deux sommets consécutifs du polygone sont confondus.")
        self.lengths = lengths
        # Paramètre t_k du sommet k : abscisse curviligne ramenée à [0, 2pi)
        self.t_vertices = 2 * np.pi * np.concatenate(([0.0], np.cumsum(lengths))) / lengths.sum()
        self.breakpoints = self.t_vertices  # sommets : toujours des nœuds de dbeta
        self.path = Path(closed, closed=True)

    def gamma(self, t):
        t = np.mod(np.asarray(t, dtype=float), 2 * np.pi)
        closed = np.vstack((self.a, self.a[:1]))
        return np.column_stack((np.interp(t, self.t_vertices, closed[:, 0]),
                                np.interp(t, self.t_vertices, closed[:, 1])))

    def inside(self, pts):
        return self.path.contains_points(pts)

    def boundary_distance(self, pts):
        """Distance exacte aux segments."""
        pts = np.asarray(pts, dtype=float)
        ab = self.b - self.a
        ap = pts[:, None, :] - self.a[None, :, :]
        s = np.clip(np.einsum("nmk,mk->nm", ap, ab) / self.lengths**2, 0.0, 1.0)
        return np.linalg.norm(ap - s[:, :, None] * ab[None, :, :], axis=2).min(axis=1)

    def distance(self, pts):
        """d(x, beta), nulle dans beta (distance exacte aux segments)."""
        d = self.boundary_distance(pts)
        d[self.inside(pts)] = 0.0
        return d

    def boundary_params(self, h_beta):
        """Paramètres t_i : chaque segment est découpé séparément, les sommets sont des nœuds."""
        t = []
        for k, L in enumerate(self.lengths):
            n = int(np.ceil(L / h_beta))
            t.append(np.linspace(self.t_vertices[k], self.t_vertices[k + 1], n, endpoint=False))
        return np.concatenate(t)

    def translated(self, shift):
        return Polygon(self.vertices + shift)


class CurveObstacle:
    """Obstacle dont le bord est une suite fermée de courbes (importée d'un fichier CAO).

    Chaque courbe est échantillonnée finement ; gamma est paramétré par l'abscisse
    curviligne ramenée à [0, 2pi) et interpolé linéairement entre les échantillons
    (erreur géométrique ~ kappa ds^2 / 8, négligeable devant h_beta). Les extrémités
    des courbes (coins éventuels) sont toujours des nœuds de dbeta.
    """

    def __init__(self, curves):
        # curves : liste de tableaux (n_k, 2), orientés et enchaînés (fin de k = début de k+1)
        self.curves = curves
        pts = np.vstack([c[:-1] for c in curves])
        self.vertices = pts  # échantillons de gamma (sans répéter le premier)
        closed = np.vstack((pts, pts[:1]))
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
        self.t_fine = 2 * np.pi * s / s[-1]  # paramètre des échantillons, dernier = 2pi
        self.closed = closed
        n_pts = np.cumsum([0] + [len(c) - 1 for c in curves])
        self.t_corners = self.t_fine[n_pts]  # début de chaque courbe, puis 2pi
        self.lengths = np.diff(s[n_pts])
        self.breakpoints = self.t_corners  # extrémités des courbes : toujours des nœuds de dbeta
        self.path = Path(closed, closed=True)
        self.tree = cKDTree(pts)

    def gamma(self, t):
        t = np.mod(np.asarray(t, dtype=float), 2 * np.pi)
        return np.column_stack((np.interp(t, self.t_fine, self.closed[:, 0]),
                                np.interp(t, self.t_fine, self.closed[:, 1])))

    def inside(self, pts):
        return self.path.contains_points(pts)

    def boundary_distance(self, pts):
        """Distance aux échantillons de gamma."""
        return self.tree.query(pts)[0]

    def distance(self, pts):
        """d(x, beta), nulle dans beta (distance aux échantillons de gamma)."""
        d = self.boundary_distance(pts)
        d[self.inside(pts)] = 0.0
        return d

    def boundary_params(self, h_beta):
        """Paramètres t_i : chaque courbe est découpée séparément, ses extrémités sont des nœuds."""
        t = []
        for k, L in enumerate(self.lengths):
            n = int(np.ceil(L / h_beta))
            t.append(np.linspace(self.t_corners[k], self.t_corners[k + 1], n, endpoint=False))
        return np.concatenate(t)

    def translated(self, shift):
        return CurveObstacle([c + shift for c in self.curves])


class HoledObstacle:
    """Composante connexe trouée : beta = outer privé de l'intérieur des trous holes.

    outer et chaque trou sont des obstacles simples (Obstacle, Polygon, CurveObstacle) ; les
    trous sont deux à deux disjoints et strictement intérieurs à outer. dbeta est fait de
    plusieurs courbes fermées (loops : outer puis les trous) ; chaque trou est une cavité de D,
    séparée du reste de D et maillée à part (mesh.triangulate).
    """

    def __init__(self, outer, holes):
        self.outer, self.holes = outer, list(holes)
        if not self.holes:
            raise ValueError("Obstacle troué sans trou : utiliser l'obstacle extérieur seul.")
        outer_path = Path(np.vstack((outer.vertices, outer.vertices[:1])), closed=True)
        for k, hole in enumerate(self.holes):
            if not outer_path.contains_path(Path(np.vstack((hole.vertices, hole.vertices[:1])))):
                raise ValueError(f"Le trou {k + 1} n'est pas strictement intérieur au bord extérieur.")
        for i, j in _overlapping_pairs(self.holes):
            raise ValueError(f"Les trous {i + 1} et {j + 1} sont imbriqués ou se chevauchent.")
        self.loops = [outer] + self.holes
        self.vertices = outer.vertices  # boîte englobante et tests de chevauchement : bord extérieur

    def in_hole(self, pts):
        """Numéro du trou contenant chaque point, -1 hors des trous."""
        k = np.full(len(pts), -1)
        for i, hole in enumerate(self.holes):
            k[hole.inside(pts)] = i
        return k

    def inside(self, pts):
        return self.outer.inside(pts) & (self.in_hole(pts) < 0)

    def distance(self, pts):
        """d(x, beta) : distance au bord extérieur, ou au bord du trou contenant x."""
        pts = np.asarray(pts, dtype=float)
        d = self.outer.distance(pts)
        k = self.in_hole(pts)
        for i, hole in enumerate(self.holes):
            d[k == i] = hole.boundary_distance(pts[k == i])
        return d

    def translated(self, shift):
        return HoledObstacle(self.outer.translated(shift), [c.translated(shift) for c in self.holes])


def annulus(center, R, r):
    """Couronne r <= |x - c| <= R : disque de rayon R troué d'un disque de rayon r."""
    if not 0 < r < R:
        raise ValueError("Couronne : il faut 0 < r < R.")
    return HoledObstacle(disk(center, R), [disk(center, r)])


def loops_of(comp):
    """Courbes fermées de dbeta pour une composante : bord extérieur, puis bords des trous."""
    return getattr(comp, "loops", [comp])


class MultiObstacle:
    """beta non connexe : réunion de composantes disjointes (Obstacle, Polygon, CurveObstacle,
    HoledObstacle).

    build_mesh traite chaque composante comme un trou de la zone T. Des composantes
    proches partagent une même zone T ; des composantes éloignées donnent plusieurs
    trous dans Q, chacun triangulé séparément.
    """

    def __init__(self, components):
        self.components = list(components)
        for i, j in _overlapping_pairs(self.components):
            raise ValueError(f"Les composantes {i + 1} et {j + 1} de beta sont imbriquées "
                             "ou se chevauchent (union() les fusionne).")
        self.vertices = np.vstack([c.vertices for c in self.components])

    def inside(self, pts):
        return np.any([c.inside(pts) for c in self.components], axis=0)

    def distance(self, pts):
        """d(x, beta) = min des distances aux composantes."""
        return np.min([c.distance(pts) for c in self.components], axis=0)

    def translated(self, shift):
        return MultiObstacle([c.translated(shift) for c in self.components])


def _overlapping_pairs(components):
    """Paires (i, j), i < j, de composantes qui se touchent, se chevauchent ou sont imbriquées.

    Bords polygonaux (sommets ou échantillons de gamma) : test exact de croisement des bords
    ou d'imbrication, seulement si les boîtes englobantes se rencontrent.
    """
    verts = [c.vertices for c in components]
    boxes = [(v.min(axis=0), v.max(axis=0)) for v in verts]
    paths = [Path(np.vstack((v, v[:1])), closed=True) for v in verts]
    pairs = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
            if np.all(lo_i <= hi_j) and np.all(lo_j <= hi_i) \
                    and paths[i].intersects_path(paths[j], filled=True):
                pairs.append((i, j))
    return pairs


def union(components, n_samples=4000):
    """beta = réunion de composantes quelconques (Obstacle, Polygon, CurveObstacle, HoledObstacle).

    Les composantes qui se chevauchent sont fusionnées en un CurveObstacle, dont le bord est
    fait des morceaux de leurs bords situés hors des autres ; les points d'intersection et les
    coins des composantes sont des extrémités de courbes, donc des nœuds de dbeta. Une
    composante contenue dans une autre disparaît ; une réunion qui enferme des cavités (anneau
    d'obstacles, obstacle troué) donne un HoledObstacle. Renvoie une composante seule ou un
    MultiObstacle.
    """
    components = list(components)
    group = list(range(len(components)))  # représentant du groupe de chaque composante

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i, j in _overlapping_pairs(components):
        group[root(j)] = root(i)
    groups = {}
    for i in range(len(components)):
        groups.setdefault(root(i), []).append(components[i])
    merged = [g[0] if len(g) == 1 else _merge(g, n_samples) for g in groups.values()]
    return merged[0] if len(merged) == 1 else MultiObstacle(merged)


def _is_corner(comp, t, angle=np.radians(1.0), dt=1e-6):
    """gamma présente un coin en t (déviation de la tangente supérieure à angle)."""
    p = comp.gamma([t - dt, t, t + dt])
    a, b = p[1] - p[0], p[2] - p[1]
    return abs(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)) > angle


def _boundary_samples(comp, n_samples):
    """Paramètres t des échantillons de gamma (≈ n_samples, répartis selon la longueur des
    morceaux entre points de cassure) et indicateur de coin de chaque échantillon."""
    bp = comp.breakpoints
    lengths = [np.linalg.norm(np.diff(comp.gamma(np.linspace(ta, tb, 201)), axis=0), axis=1).sum()
               for ta, tb in zip(bp[:-1], bp[1:])]
    t, corner = [], []
    for ta, tb, L in zip(bp[:-1], bp[1:], lengths):
        m = max(8, int(np.ceil(n_samples * L / sum(lengths))))
        t.append(np.linspace(ta, tb, m, endpoint=False))
        corner.append(np.zeros(m, dtype=bool))
        corner[-1][0] = _is_corner(comp, ta)  # point de cassure lisse (ex. t = 0) : ignoré
    return np.concatenate(t), np.concatenate(corner)


def _signed_area(pts):
    """Aire algébrique du polygone fermé de sommets pts (> 0 : sens direct)."""
    return 0.5 * (pts[:, 0] @ np.roll(pts[:, 1], -1) - pts[:, 1] @ np.roll(pts[:, 0], -1))


def _merge(comps, n_samples):
    """Réunion connexe de composantes qui se chevauchent : un CurveObstacle, ou un HoledObstacle
    si la réunion enferme des cavités.

    Le bord de la réunion est fait des parties des bords (extérieurs et de trous) qui ne sont
    dans aucune autre composante. Chaque bord est orienté beta à gauche (sens direct pour un
    bord extérieur, indirect pour un trou) : les morceaux gardés s'enchaînent dans ce sens, et
    l'orientation des contours obtenus distingue le bord extérieur (direct) des trous.
    """
    curves, contributors = [], []
    for k, comp in enumerate(comps):
        others = comps[:k] + comps[k + 1:]
        kept, intact = [], []
        for n_loop, loop in enumerate(loops_of(comp)):

            def covered(t, loop=loop):
                pts = loop.gamma(np.atleast_1d(t))
                return np.any([o.inside(pts) for o in others], axis=0)

            t, corner = _boundary_samples(loop, n_samples)
            if (_signed_area(loop.gamma(t)) > 0) == (n_loop > 0):
                # beta à gauche du bord ; le premier échantillon (point de cassure) reste en tête
                t, corner = np.roll(t[::-1], 1), np.roll(corner[::-1], 1)
            out = ~covered(t)
            kept.append(out.any())
            if not out.any():
                continue  # bord contenu dans les autres composantes
            intact.append(out.all())
            period = 2 * np.pi * np.sign(t[1] - t[0])  # gamma est 2pi-périodique
            if out.all():
                # Bord intact (autres composantes dans une cavité, ou tangentes : refusé plus bas)
                arcs = [(np.append(t, t[0] + period), np.append(corner, True))]
                arcs[0][1][0] = True
            else:
                # Décalage cyclique : le premier échantillon est couvert, aucun arc ne fait le
                # tour ; t reste monotone et t[n] (voisin du dernier) est le premier échantillon
                s = np.argmin(out)
                t = np.concatenate((t[s:], t[:s] + period, [t[s] + period]))
                corner, out = np.roll(corner, -s), np.roll(out, -s)
                arcs = []
                starts = np.flatnonzero(out & ~np.roll(out, 1))
                ends = np.flatnonzero(out & ~np.roll(out, -1))
                for a, b in zip(starts, ends):
                    ta = _bisect(covered, t[a - 1], t[a])
                    tb = _bisect(covered, t[b + 1], t[b])
                    arcs.append((np.concatenate(([ta], t[a:b + 1], [tb])),
                                 np.concatenate(([True], corner[a:b + 1], [True]))))
            # Découpage aux coins : un CurveObstacle garde les extrémités de ses courbes
            for arc_t, arc_corner in arcs:
                cut = np.flatnonzero(arc_corner)
                curves += [loop.gamma(arc_t[i:j + 1]) for i, j in zip(cut[:-1], cut[1:])]
        if any(kept):
            contributors.append((comp, all(kept) and all(intact)))

    if len(contributors) == 1 and contributors[0][1]:
        return contributors[0][0]  # une composante contient toutes les autres : gamma exact
    size = max(np.ptp(np.vstack(curves), axis=0))
    loops = [CurveObstacle(loop) for loop in _chain(curves, 1e-7 * size)]
    areas = np.array([_signed_area(L.vertices) for L in loops])
    outer = loops[int(np.argmax(areas))]
    if np.sum(areas > 0) > 1:
        if all(outer.inside(L.vertices[:1])[0] for L in loops):
            raise ValueError("Une composante de beta est dans une cavité d'une autre : "
                             "cas non pris en charge.")
        raise ValueError("Des composantes de beta se touchent sans se chevaucher franchement "
                         "(contact tangent) : les écarter ou les faire se chevaucher.")
    holes = [L for L, a in zip(loops, areas) if a < 0]
    return HoledObstacle(outer, holes) if holes else outer


def _bisect(covered, t_in, t_out, n_iter=60):
    """Paramètre du point où gamma sort des autres composantes, entre t_in (couvert) et t_out."""
    for _ in range(n_iter):
        t_mid = 0.5 * (t_in + t_out)
        if covered(t_mid)[0]:
            t_in = t_mid
        else:
            t_out = t_mid
    return t_out


def from_step(path, scale=1.0, n_samples=4000):
    """Obstacle beta lu dans un fichier STEP (ou IGES, BREP) via gmsh / OpenCASCADE.

    Le fichier doit décrire une géométrie plane contenue dans un plan z = cte :
    une face (son bord extérieur est utilisé) ou un contour fermé de courbes.
    Les contours fermés sont classés par imbrication : un contour contenu dans un nombre impair
    d'autres est le bord d'un trou (face percée), et les autres sont des bords extérieurs.
    Le résultat est la réunion (union) des composantes ainsi formées : fusion de celles qui se
    chevauchent, MultiObstacle sinon.
    scale : facteur appliqué aux coordonnées (OpenCASCADE convertit en mm par défaut,
    scale=1e-3 ramène alors en mètres).
    """
    import gmsh  # import local : gmsh n'est nécessaire que pour lire un fichier CAO

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("beta")
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()

        if gmsh.model.getEntities(3):
            raise ValueError("Le fichier contient un volume : exporter une face ou un contour plan.")
        faces = gmsh.model.getEntities(2)
        if faces:
            bnd = gmsh.model.getBoundary(faces, combined=True, oriented=False)
            tags = [abs(t) for _, t in bnd]
        else:
            tags = [t for _, t in gmsh.model.getEntities(1)]
        if not tags:
            raise ValueError(f"Aucune courbe trouvée dans {path}.")

        def sample(tag, n):
            lo, hi = gmsh.model.getParametrizationBounds(1, tag)
            u = np.linspace(lo[0], hi[0], n + 1)
            return np.asarray(gmsh.model.getValue(1, tag, u)).reshape(-1, 3)

        # Premier passage grossier pour répartir les échantillons selon la longueur
        rough = {t: sample(t, 200) for t in tags}
        lengths = {t: np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for t, p in rough.items()}
        total = sum(lengths.values())
        curves = [sample(t, max(50, int(np.ceil(n_samples * L / total))))
                  for t, L in lengths.items() if L > 1e-12 * total]
    finally:
        gmsh.finalize()

    z = np.concatenate([c[:, 2] for c in curves])
    if np.ptp(z) > 1e-8 * total:
        raise ValueError("La géométrie n'est pas contenue dans un plan z = cte.")
    curves = [scale * c[:, :2] for c in curves]
    loops = [CurveObstacle(loop) for loop in _chain(curves, 1e-6 * scale * total)]
    if len(loops) == 1:
        return loops[0]
    paths = [Path(L.closed) for L in loops]
    parents = [[j for j in range(len(loops)) if j != i and paths[j].contains_path(paths[i])]
               for i in range(len(loops))]
    depth = [len(p) for p in parents]
    holes = {i: [] for i in range(len(loops)) if depth[i] % 2 == 0}
    for i, p in enumerate(parents):
        if depth[i] % 2:  # trou : rattaché au bord extérieur qui le contient directement
            holes[max(p, key=lambda j: depth[j])].append(loops[i])
    return union([HoledObstacle(loops[i], hs) if hs else loops[i] for i, hs in holes.items()])


def _chain(curves, tol):
    """Ordonne et oriente les courbes en contours fermés (un par composante connexe)."""
    remaining = list(curves)
    loops = []
    while remaining:
        loop = [remaining.pop(0)]
        while not np.linalg.norm(loop[-1][-1] - loop[0][0]) < tol:
            end = loop[-1][-1]
            for k, c in enumerate(remaining):
                if np.linalg.norm(c[0] - end) < tol:
                    loop.append(remaining.pop(k))
                    break
                if np.linalg.norm(c[-1] - end) < tol:
                    loop.append(remaining.pop(k)[::-1])
                    break
            else:
                raise ValueError("Les courbes du fichier ne forment pas un contour fermé.")
        for k in range(len(loop)):  # recolle exactement les extrémités
            loop[k][-1] = loop[(k + 1) % len(loop)][0]
        loops.append(loop)
    return loops


def domain_from_obstacle(obstacle, h, d_min, m_beta=None):
    """Omega = [0, Lx] x [0, Ly] aligné sur la grille de pas h, à distance >= d_min de beta.

    La distance d'un segment de beta à un côté de Omega est atteinte en un sommet :
    il suffit d'imposer la marge d_min autour de la boîte englobante des sommets
    (sommets du polygone, ou échantillons de gamma pour un Obstacle, un
    CurveObstacle ou un MultiObstacle).
    beta est centré dans Omega puis translaté pour que Omega commence en (0, 0).
    Renvoie (obstacle translaté, Lx, Ly, décalage appliqué aux sommets).
    """
    if m_beta is not None and d_min < m_beta + h:
        raise ValueError("d_min doit vérifier d_min >= m_beta + h.")
    lo, hi = obstacle.vertices.min(axis=0), obstacle.vertices.max(axis=0)
    n = np.ceil((hi - lo + 2 * d_min) / h - 1e-9).astype(int)
    origin = 0.5 * (lo + hi) - 0.5 * n * h
    Lx, Ly = n * h
    return obstacle.translated(-origin), Lx, Ly, -origin
