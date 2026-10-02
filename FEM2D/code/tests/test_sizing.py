"""Raffinement a priori près de beta (sizing.py)."""
import numpy as np
import pytest

from cases import CASES, DISK, POLYGON, prepare
from config import LX, LY, MeshParams
from geometry import MultiObstacle, Polygon, disk, domain_from_obstacle
from mesh import build_mesh
from sizing import CORNER, graded_params, size_field


def beta_spacing(mesh):
    """Longueur des arêtes de dbeta, avec le milieu de chaque arête."""
    out = []
    for xy in mesh.beta_points:
        nxt = np.roll(xy, -1, axis=0)
        out.append((np.linalg.norm(nxt - xy, axis=1), 0.5 * (xy + nxt)))
    return out


def test_constant_size_matches_uniform_params():
    """Taille constante h_beta : mêmes nœuds de dbeta que sans raffinement."""
    poly = Polygon(POLYGON)
    np.testing.assert_allclose(graded_params(poly, lambda p: np.full(len(p), 0.15)),
                               poly.boundary_params(0.15), atol=1e-12)


def test_disk_unchanged():
    """Disque de rayon 1 : ni coin, ni interstice, courbure faible : aucune source."""
    assert size_field([disk(DISK["center"], DISK["R"])], disk(DISK["center"], DISK["R"]).inside,
                      0.15, 1.0) is None
    a, _, _ = prepare(CASES["disque"], MeshParams(raffinement=0.0))
    b, _, _ = prepare(CASES["disque"], MeshParams(raffinement=1.0))
    np.testing.assert_array_equal(a.coords, b.coords)


def test_polygon_singular_corners_only():
    """L : arêtes de dbeta raccourcies aux 5 coins saillants de beta (angle de D = 3pi/2),
    pas au coin rentrant (angle de D = pi/2, u régulière)."""
    h_beta = 0.15
    mesh, _, _ = prepare(CASES["polygone"], MeshParams(h_beta=h_beta))
    (length, mid), = beta_spacing(mesh)
    shift = mesh.beta_points[0][0] - np.array(POLYGON[0])  # translation de beta dans Omega
    corners = np.array(POLYGON) + shift
    d = np.linalg.norm(mid[:, None] - corners[None], axis=-1)
    nearest = d.argmin(axis=1)
    reentrant = 3  # sommet (3.0, 2.5) du L
    for c in range(len(POLYGON)):
        shortest = length[(nearest == c) & (d.min(axis=1) < 0.5)].min()
        if c == reentrant:
            assert shortest > 0.5 * h_beta
        else:
            assert shortest < 1.5 * h_beta / CORNER


def test_gap_resolved():
    """Deux disques à 0.05 : au moins trois éléments dans la largeur de l'interstice."""
    mesh, Lx, Ly = prepare(CASES["tangents"], MeshParams())
    for length, mid in beta_spacing(mesh):
        near_gap = np.linalg.norm(mid - [Lx / 2, Ly / 2], axis=1) < 0.05
        assert length[near_gap].max() < 0.05 / 3 * 1.2


def test_refinement_fixes_inverted_elements():
    """Disques à 0.005 : refusés sans raffinement (test_mesh), maillés avec."""
    multi = MultiObstacle([disk((0, 0), 1), disk((2.005, 0), 1)])
    obstacle, Lx, Ly, _ = domain_from_obstacle(multi, 0.5, 3.0, 1.0)
    with pytest.raises(ValueError, match="Jacobien"):
        build_mesh(obstacle, Lx, Ly, 0.5, 0.15, 0.8, 1.0)
    build_mesh(obstacle, Lx, Ly, 0.5, 0.15, 0.8, 1.0, refine=1.0)


def test_thin_obstacle_not_a_gap():
    """Plaque fine : ses deux faces sont proches à travers beta, pas à travers D."""
    plate = Polygon([(0, 0), (3, 0), (3, 0.05), (0, 0.05)])
    field = size_field([plate], plate.inside, 0.15, 1.0)
    far_from_ends = np.abs(field.points[:, 0] - 1.5) < 0.5
    assert not far_from_ends.any()


def test_curvature_criterion(monkeypatch):
    """Critère de courbure désactivé par défaut ; activé, il raccourcit les arêtes de dbeta
    dans les coins arrondis (rayon 0.2) de deux_obstacles.step."""
    import sizing
    assert not sizing.CURVATURE
    a, _, _ = prepare(CASES["non_connexe"], MeshParams(raffinement=1.0))
    b, _, _ = prepare(CASES["non_connexe"], MeshParams(raffinement=0.0))
    assert a.n_nodes == b.n_nodes
    monkeypatch.setattr(sizing, "CURVATURE", True)
    c, _, _ = prepare(CASES["non_connexe"], MeshParams(raffinement=1.0))
    assert c.n_nodes > a.n_nodes
    shortest = min(length.min() for length, _ in beta_spacing(c))
    assert shortest < 1.2 * sizing.THETA * 0.2
