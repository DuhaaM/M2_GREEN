"""Construction du maillage mixte Q2 (grille) / P2 (gmsh autour de beta).

Sections 3.2 et 3.3 de main.tex. Le résultat est un objet Mesh contenant :
    coords (N, 2), tri (n_T, 6), quad (n_Q, 9), bnd (indices des nœuds de bord),
    bnd_beta (indices des nœuds de bord situés sur dbeta, sous-ensemble de bnd).
"""
from dataclasses import dataclass

import gmsh
import numpy as np
from scipy import ndimage


@dataclass
class Mesh:
    coords: np.ndarray
    tri: np.ndarray
    quad: np.ndarray
    bnd: np.ndarray
    bnd_beta: np.ndarray  # nœuds de dbeta (y compris les milieux P2 placés sur la courbe)
    beta_points: list  # points de discrétisation de gamma, un tableau par composante de beta

    @property
    def n_nodes(self):
        return len(self.coords)


# ----------------------------------------------------------------------------
# Étape 1 : sélection des cellules de la grille
# ----------------------------------------------------------------------------
def select_cells(obstacle, nx, ny, h, m_beta):
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    keep = np.ones((nx, ny), dtype=bool)
    for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]:
        pts = np.column_stack((((i + dx) * h).ravel(), ((j + dy) * h).ravel()))
        keep &= (obstacle.distance(pts) >= m_beta).reshape(nx, ny)

    while True:
        # Îlots de cellules isolés du bord de Omega (entourés par des composantes de beta) :
        # retirés, pour que chaque trou de Q soit bordé par un seul contour
        labels, _ = ndimage.label(keep)
        border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        keep &= np.isin(labels, border[border > 0])

        padded = np.pad(keep, 1)
        n_neighbors = (padded[:-2, 1:-1].astype(int) + padded[2:, 1:-1]
                       + padded[1:-1, :-2] + padded[1:-1, 2:])
        bad = keep & (n_neighbors <= 1)
        # Sommet intérieur entouré de deux cellules gardées en diagonale seulement
        a, b, c, d = keep[:-1, :-1], keep[1:, :-1], keep[:-1, 1:], keep[1:, 1:]
        pinch1 = a & d & ~b & ~c
        pinch2 = b & c & ~a & ~d
        bad[:-1, :-1] |= pinch1
        bad[1:, 1:] |= pinch1
        bad[1:, :-1] |= pinch2
        bad[:-1, 1:] |= pinch2
        if not bad.any():
            break
        keep &= ~bad

    if not (keep[0, :].all() and keep[-1, :].all() and keep[:, 0].all() and keep[:, -1].all()):
        raise ValueError("beta est trop proche de dOmega : le contour intérieur touche le bord.")
    return keep


# ----------------------------------------------------------------------------
# Étape 2 : contours intérieurs de Q (un par trou, suite fermée de sommets de la grille)
# ----------------------------------------------------------------------------
def inner_contours(keep):
    """Renvoie (loops, holes) : un contour par trou de Q et le numéro de trou de chaque cellule.

    Les trous sont les composantes connexes (par les arêtes) des cellules non gardées ;
    holes vaut 0 sur les cellules de Q et k + 1 sur celles du trou k.
    """
    nx, ny = keep.shape
    holes, n_holes = ndimage.label(~keep)
    padded = np.pad(holes, 1)  # 0 hors de la grille : comme une cellule de Q
    adjacency = [{} for _ in range(n_holes)]

    def add_edge(k, p, q):
        adjacency[k - 1].setdefault(p, []).append(q)
        adjacency[k - 1].setdefault(q, []).append(p)

    for i in range(nx):
        for j in range(1, ny):
            if keep[i, j - 1] != keep[i, j]:
                add_edge(max(holes[i, j - 1], holes[i, j]), (i, j), (i + 1, j))
    for i in range(1, nx):
        for j in range(ny):
            if keep[i - 1, j] != keep[i, j]:
                add_edge(max(holes[i - 1, j], holes[i, j]), (i, j), (i, j + 1))

    loops = []
    for adj in adjacency:
        if any(len(v) != 2 for v in adj.values()):
            raise ValueError("Le contour intérieur de Q n'est pas une courbe simple.")
        start = next(iter(adj))
        loop, prev, cur = [start], None, start
        while True:
            a, b = adj[cur]
            nxt = b if a == prev else a
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) != len(adj):
            raise ValueError("Un trou de Q est bordé par plusieurs contours fermés.")
        loops.append(loop)
    return loops, holes


# ----------------------------------------------------------------------------
# Étapes 3-4 : triangulation de la zone T avec gmsh (éléments d'ordre 1)
# ----------------------------------------------------------------------------
def triangulate(contour_xy, betas_xy, h, h_beta, delta):
    """Renvoie (node_xy, triangles, contour_nodes, beta_nodes) en indices locaux gmsh.

    betas_xy : liste des points de dbeta, un tableau par composante connexe de beta
    (chacune devient un trou de la zone T). beta_nodes suit la même structure.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("T")
        geo = gmsh.model.geo

        def closed_polygon(points):
            tags = [geo.addPoint(x, y, 0.0) for x, y in points]
            n = len(tags)
            lines = [geo.addLine(tags[k], tags[(k + 1) % n]) for k in range(n)]
            return tags, lines

        contour_pts, contour_lines = closed_polygon(contour_xy)
        betas = [closed_polygon(xy) for xy in betas_xy]
        beta_lines = [line for _, lines in betas for line in lines]
        loop_out = geo.addCurveLoop(contour_lines)
        loops_beta = [geo.addCurveLoop(lines) for _, lines in betas]
        geo.addPlaneSurface([loop_out] + loops_beta)
        # Aucun nœud ajouté sur les contours : leurs nœuds sont imposés
        for line in contour_lines + beta_lines:
            geo.mesh.setTransfiniteCurve(line, 2)
        geo.synchronize()

        field = gmsh.model.mesh.field
        # Trou de Q sans composante de beta : taille h partout
        size_min = h_beta if beta_lines else h
        dist = field.add("Distance")
        field.setNumbers(dist, "CurvesList", beta_lines or contour_lines)
        field.setNumber(dist, "Sampling", 10)
        thr = field.add("Threshold")
        field.setNumber(thr, "InField", dist)
        field.setNumber(thr, "SizeMin", size_min)
        field.setNumber(thr, "SizeMax", h)
        field.setNumber(thr, "DistMin", 0.0)
        field.setNumber(thr, "DistMax", delta)
        field.setAsBackgroundMesh(thr)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        gmsh.model.mesh.generate(2)

        node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
        index = {int(t): k for k, t in enumerate(node_tags)}
        node_xy = node_xyz.reshape(-1, 3)[:, :2]
        _, tri_nodes = gmsh.model.mesh.getElementsByType(2)
        triangles = np.array([index[int(t)] for t in tri_nodes]).reshape(-1, 3)

        def point_node(ptag):
            tags, _, _ = gmsh.model.mesh.getNodes(0, ptag)
            return index[int(tags[0])]

        contour_nodes = [point_node(p) for p in contour_pts]
        beta_nodes = [[point_node(p) for p in pts] for pts, _ in betas]
    finally:
        gmsh.finalize()
    return node_xy, triangles, contour_nodes, beta_nodes


# ----------------------------------------------------------------------------
# Assemblage du maillage global (section 3.3)
# ----------------------------------------------------------------------------
def build_mesh(obstacle, Lx, Ly, h, h_beta, delta, m_beta):
    nx, ny = int(round(Lx / h)), int(round(Ly / h))
    if abs(nx * h - Lx) > 1e-9 or abs(ny * h - Ly) > 1e-9:
        raise ValueError("Lx / h et Ly / h doivent être entiers.")
    # Tout point d'une cellule est à moins de h/2 de l'un des 5 points testés par select_cells
    # (sommets et centre) : m_beta > h/2 garantit que les cellules gardées ne touchent pas beta
    if m_beta <= h / 2:
        raise ValueError(f"m_beta = {m_beta:g} trop petit : il faut m_beta > h/2 = {h / 2:g}.")

    keep = select_cells(obstacle, nx, ny, h, m_beta)
    loops, holes = inner_contours(keep)
    # Composantes connexes de beta (une seule pour Obstacle, Polygon, CurveObstacle)
    comps = getattr(obstacle, "components", [obstacle])
    t_beta = [c.boundary_params(h_beta) for c in comps]
    beta_xy = [c.gamma(t) for c, t in zip(comps, t_beta)]
    # Trou de Q contenant chaque composante (repéré par la cellule d'un point de son bord)
    members = [[] for _ in loops]
    for c, xy in enumerate(beta_xy):
        i, j = np.clip((xy // h).astype(int), 0, [nx - 1, ny - 1]).T
        hole = np.unique(holes[i, j])
        if hole[0] == 0:
            raise ValueError(f"La composante {c + 1} de beta traverse une cellule de Q.")
        if len(hole) > 1:
            raise ValueError(f"La composante {c + 1} de beta est à cheval sur plusieurs trous de Q.")
        members[hole[0] - 1].append(c)

    coords = []            # coordonnées de tous les nœuds
    on_beta = []           # nœud situé sur dbeta
    beta_index = {}        # numéro global -> (composante c, indice k du point gamma_c(t_k))

    def new_node(xy, is_beta=False):
        coords.append(xy)
        on_beta.append(is_beta)
        return len(coords) - 1

    # 1. Sommets de la grille utilisés par Q
    used = np.zeros((nx + 1, ny + 1), dtype=bool)
    ci, cj = np.nonzero(keep)
    for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        used[ci + di, cj + dj] = True
    vid = -np.ones((nx + 1, ny + 1), dtype=int)
    for i, j in zip(*np.nonzero(used)):
        vid[i, j] = new_node((i * h, j * h))

    # 2. Sommets des triangles, une zone T par trou de Q : ceux des contours
    #    reprennent les numéros existants
    tri_parts = []
    for loop, cs in zip(loops, members):
        contour_xy = [(i * h, j * h) for i, j in loop]
        node_xy, tri3, contour_nodes, beta_nodes = triangulate(
            contour_xy, [beta_xy[c] for c in cs], h, h_beta, delta)
        g2glob = -np.ones(len(node_xy), dtype=int)
        for (i, j), g in zip(loop, contour_nodes):
            g2glob[g] = vid[i, j]
        for c, nodes in zip(cs, beta_nodes):
            for k, g in enumerate(nodes):
                g2glob[g] = new_node(tuple(beta_xy[c][k]), is_beta=True)
                beta_index[g2glob[g]] = (c, k)
        for g in np.nonzero(g2glob < 0)[0]:
            g2glob[g] = new_node(tuple(node_xy[g]))
        tri_parts.append(g2glob[tri3])
    tri3 = np.vstack(tri_parts)

    # Orientation directe des triangles
    P = np.array(coords)
    e1 = P[tri3[:, 1]] - P[tri3[:, 0]]
    e2 = P[tri3[:, 2]] - P[tri3[:, 0]]
    neg = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
    tri3[neg] = tri3[neg][:, [0, 2, 1]]

    # 3. Milieux d'arêtes (dictionnaire d'arêtes)
    edge_mid = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in edge_mid:
            return edge_mid[key]
        if a in beta_index and b in beta_index and beta_index[a][0] == beta_index[b][0]:
            (c, ka), (_, kb) = beta_index[a], beta_index[b]
            n_beta = len(t_beta[c])
            if (ka - kb) % n_beta == 1:
                ka, kb = kb, ka
            if (kb - ka) % n_beta == 1:  # arête posée sur dbeta
                ta, tb = t_beta[c][ka], t_beta[c][kb]
                if kb == 0:
                    tb += 2 * np.pi
                m = new_node(tuple(comps[c].gamma([0.5 * (ta + tb)])[0]), is_beta=True)
                edge_mid[key] = m
                return m
        m = new_node(tuple(0.5 * (np.array(coords[a]) + np.array(coords[b]))))
        edge_mid[key] = m
        return m

    tri = np.empty((len(tri3), 6), dtype=int)
    tri[:, :3] = tri3
    for e, (a, b, c) in enumerate(tri3):
        tri[e, 3:] = midpoint(a, b), midpoint(b, c), midpoint(c, a)

    # 4. Quadrilatères Q2 : sommets, milieux (bas, droite, haut, gauche), centre
    quad = np.empty((len(ci), 9), dtype=int)
    for e, (i, j) in enumerate(zip(ci, cj)):
        v = [vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]]
        quad[e, :4] = v
        quad[e, 4:8] = [midpoint(v[k], v[(k + 1) % 4]) for k in range(4)]
        quad[e, 8] = new_node(((i + 0.5) * h, (j + 0.5) * h))

    # 5. Nœuds de bord
    coords = np.array(coords)
    tol = 1e-10 * h
    x, y = coords[:, 0], coords[:, 1]
    on_omega = (np.abs(x) < tol) | (np.abs(x - Lx) < tol) | (np.abs(y) < tol) | (np.abs(y - Ly) < tol)
    on_beta = np.array(on_beta)
    bnd = np.nonzero(on_omega | on_beta)[0]
    bnd_beta = np.nonzero(on_beta)[0]

    return Mesh(coords=coords, tri=tri, quad=quad, bnd=bnd, bnd_beta=bnd_beta, beta_points=beta_xy)
