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
        self.pts_fine = self.gamma(self.t_fine)
        self.tree = cKDTree(self.pts_fine)

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
        """g_beta = rho^2 - r(theta)^2 : < 0 dans beta, = 0 sur le bord, > 0 dans D."""
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
