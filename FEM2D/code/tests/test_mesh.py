"""Conformité du maillage mixte Q2 / P2 et garde-fous sur les paramètres et la géométrie."""
import numpy as np
import pytest

from assembly import check_jacobians, element_errors
from cases import CASES, DISK, POLYGON, prepare
from config import LX, LY, MeshParams
from geometry import MultiObstacle, Obstacle, Polygon, disk, domain_from_obstacle, peanut
from mesh import build_mesh

# Arêtes locales (sommet, sommet, milieu)
TRI_EDGES = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
QUAD_EDGES = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]


@pytest.fixture(scope="module", params=["disque", "polygone", "eloignes", "non_connexe"])
def case_mesh(request):
    return prepare(CASES[request.param], MeshParams())


def test_edges_conforming(case_mesh):
    """Chaque arête : au plus 2 éléments, un seul nœud milieu ; arêtes isolées sur le bord."""
    mesh, _, _ = case_mesh
    mids = {}
    for conn, edges in ((mesh.tri, TRI_EDGES), (mesh.quad, QUAD_EDGES)):
        for a, b, m in edges:
            for key, mid in zip(np.sort(conn[:, [a, b]], axis=1), conn[:, m]):
                mids.setdefault(tuple(key), []).append(mid)
    assert all(len(v) <= 2 for v in mids.values())
    assert all(len(set(v)) == 1 for v in mids.values())
    on_bnd = np.zeros(mesh.n_nodes, dtype=bool)
    on_bnd[mesh.bnd] = True
    for (a, b), v in mids.items():
        if len(v) == 1:
            assert on_bnd[[a, b, v[0]]].all()


def test_every_node_used(case_mesh):
    mesh, _, _ = case_mesh
    used = np.zeros(mesh.n_nodes, dtype=bool)
    used[mesh.tri.ravel()] = used[mesh.quad.ravel()] = True
    assert used.all()


def test_polygon_area():
    """|D| = Lx Ly - |beta|, exact pour un obstacle polygonal (et Jacobiens > 0)."""
    mesh, Lx, Ly = prepare(CASES["polygone"], MeshParams())
    errs = element_errors(mesh, np.zeros(mesh.n_nodes), lambda x, y: 0 * x,
                          lambda x, y: (0 * x, 0 * x))
    x, y = np.array(POLYGON).T
    beta_area = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    area = sum(a.sum() for a, *_ in errs.values())
    assert np.isclose(area, Lx * Ly - beta_area, rtol=1e-12)


def test_disk_area():
    """Arêtes courbes P2 isoparamétriques : |D| proche de Lx Ly - pi R^2."""
    h = 0.25
    mesh = build_mesh(disk(DISK["center"], DISK["R"]), LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)
    errs = element_errors(mesh, np.zeros(mesh.n_nodes), lambda x, y: 0 * x,
                          lambda x, y: (0 * x, 0 * x))
    area = sum(a.sum() for a, *_ in errs.values())
    assert np.isclose(area, LX * LY - np.pi * DISK["R"] ** 2, rtol=1e-6)


# ----------------------------------------------------------------------------
# Garde-fous
# ----------------------------------------------------------------------------
def test_m_beta_too_small():
    with pytest.raises(ValueError, match="m_beta"):
        build_mesh(disk(DISK["center"], DISK["R"]), LX, LY, 0.5, 0.15, 0.8, 0.25)


def test_polar_radius_positive():
    with pytest.raises(ValueError, match="strictement positif"):
        peanut((0, 0), 1.0, 1.2)
    with pytest.raises(ValueError):
        Obstacle((0, 0), lambda t: np.cos(t))


@pytest.mark.parametrize("components", [
    [disk((0, 0), 1), disk((1.5, 0), 1)],                        # chevauchement
    [disk((0, 0), 1), disk((0.1, 0), 0.3)],                      # imbrication
    [Polygon([(-2, -0.1), (2, -0.1), (2, 0.1), (-2, 0.1)]),     # croix : aucun sommet
     Polygon([(-0.1, -2), (0.1, -2), (0.1, 2), (-0.1, 2)])],    # dans l'autre composante
])
def test_overlapping_components(components):
    with pytest.raises(ValueError, match="imbriquées ou se chevauchent"):
        MultiObstacle(components)


def test_disjoint_components():
    multi = MultiObstacle([disk((0, 0), 1), disk((2.5, 0), 1), Polygon([(5, 0), (6, 0), (6, 1)])])
    assert len(multi.components) == 3


def close_disks_mesh(gap, h_beta):
    multi = MultiObstacle([disk((0, 0), 1), disk((2 + gap, 0), 1)])
    obstacle, Lx, Ly, _ = domain_from_obstacle(multi, 0.5, 3.0, 1.0)
    return build_mesh(obstacle, Lx, Ly, 0.5, h_beta, 0.8, 1.0)


def test_close_components_inverted_elements():
    """Deux disques à 0.005 l'un de l'autre : triangles courbes retournés, refusés au maillage ;
    un h_beta plus petit règle le problème, comme l'indique le message."""
    with pytest.raises(ValueError, match="Jacobien négatif.*réduire h_beta"):
        close_disks_mesh(0.005, 0.15)
    close_disks_mesh(0.005, 0.075)


def test_check_jacobians_flipped_element():
    mesh, _, _ = prepare(CASES["disque"], MeshParams())
    mesh.quad[7] = mesh.quad[7][[1, 0, 3, 2, 4, 7, 6, 5, 8]]  # carré retourné (symétrie x)
    with pytest.raises(ValueError, match="quadrilatère n° 7"):
        check_jacobians(mesh)
