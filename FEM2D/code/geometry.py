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

    def distance(self, pts):
        """d(x, beta), nulle dans beta."""
        d, _ = self.tree.query(pts)
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

    def distance(self, pts):
        """d(x, beta), nulle dans beta (distance exacte aux segments)."""
        pts = np.asarray(pts, dtype=float)
        ab = self.b - self.a
        ap = pts[:, None, :] - self.a[None, :, :]
        s = np.clip(np.einsum("nmk,mk->nm", ap, ab) / self.lengths**2, 0.0, 1.0)
        d = np.linalg.norm(ap - s[:, :, None] * ab[None, :, :], axis=2).min(axis=1)
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

    def distance(self, pts):
        """d(x, beta), nulle dans beta (distance aux échantillons de gamma)."""
        d, _ = self.tree.query(pts)
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


class MultiObstacle:
    """beta non connexe : réunion de composantes disjointes (Obstacle, Polygon, CurveObstacle).

    build_mesh traite chaque composante comme un trou de la zone T. Des composantes
    proches partagent une même zone T ; des composantes éloignées donnent plusieurs
    trous dans Q, chacun triangulé séparément.
    """

    def __init__(self, components):
        self.components = list(components)
        # Bords polygonaux (sommets ou échantillons de gamma) : test exact de croisement des
        # bords ou d'imbrication, seulement si les boîtes englobantes se rencontrent
        verts = [c.vertices for c in self.components]
        boxes = [(v.min(axis=0), v.max(axis=0)) for v in verts]
        paths = [Path(np.vstack((v, v[:1])), closed=True) for v in verts]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                (lo_i, hi_i), (lo_j, hi_j) = boxes[i], boxes[j]
                if np.all(lo_i <= hi_j) and np.all(lo_j <= hi_i) \
                        and paths[i].intersects_path(paths[j], filled=True):
                    raise ValueError(f"Les composantes {i + 1} et {j + 1} de beta sont imbriquées "
                                     "ou se chevauchent (beta troué ?).")
        self.vertices = np.vstack(verts)

    def inside(self, pts):
        return np.any([c.inside(pts) for c in self.components], axis=0)

    def distance(self, pts):
        """d(x, beta) = min des distances aux composantes."""
        return np.min([c.distance(pts) for c in self.components], axis=0)

    def translated(self, shift):
        return MultiObstacle([c.translated(shift) for c in self.components])


def from_step(path, scale=1.0, n_samples=4000):
    """Obstacle beta lu dans un fichier STEP (ou IGES, BREP) via gmsh / OpenCASCADE.

    Le fichier doit décrire une géométrie plane contenue dans un plan z = cte :
    une face (son bord extérieur est utilisé) ou un contour fermé de courbes.
    Chaque composante connexe doit être bordée par une seule courbe fermée (pas de
    trou) ; plusieurs composantes disjointes donnent un MultiObstacle.
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
    return loops[0] if len(loops) == 1 else MultiObstacle(loops)


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
