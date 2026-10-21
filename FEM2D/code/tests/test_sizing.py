"""Raffinement a priori près de beta (sizing.py)."""
import numpy as np
import pytest

from cases import CASES, DISK, POLYGON, prepare
from config import LX, LY, MeshParams
from geometry import HoledObstacle, MultiObstacle, Polygon, disk, domain_from_obstacle, union
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


def test_cavity_corners():
    """Cavité en L dans un carré : dans la cavité, seul le coin rentrant du L (angle de D = 3pi/2)
    est singulier ; ses 5 coins saillants (angle de D = pi/2) ne le sont pas. Les coins du carré
    extérieur (angle de D = 3pi/2) sont singuliers."""
    h_beta = 0.15
    hole = np.array(POLYGON) - POLYGON[0] + (1.0, 1.0)
    square = [(0, 0), (6, 0), (6, 4.5), (0, 4.5)]
    obstacle, Lx, Ly, shift = domain_from_obstacle(HoledObstacle(Polygon(square), [Polygon(hole)]),
                                                   0.5, 2.5, 0.9)
    mesh = build_mesh(obstacle, Lx, Ly, 0.5, h_beta, 0.8, 0.9, refine=1.0)
    for (length, mid), corners, singular in zip(beta_spacing(mesh), (square, hole),
                                                ([0, 1, 2, 3], [3])):
        d = np.linalg.norm(mid[:, None] - (np.array(corners) + shift)[None], axis=-1)
        for c in range(len(corners)):
            shortest = length[(d.argmin(axis=1) == c) & (d.min(axis=1) < 0.5)].min()
            assert (shortest < 1.5 * h_beta / CORNER) == (c in singular)


def test_gap_resolved():
    """Deux disques à 0.05 : au moins trois éléments dans la largeur de l'interstice."""
    mesh, Lx, Ly = prepare(CASES["tangents"], MeshParams())
    for length, mid in beta_spacing(mesh):
        near_gap = np.linalg.norm(mid - [Lx / 2, Ly / 2], axis=1) < 0.05
        assert length[near_gap].max() < 0.05 / 3 * 1.2


def test_acute_corner_not_a_gap():
    """Disque et carré fusionnés : aux points d'intersection, D a un coin aigu (≈ 41°) dont les
    deux côtés sont proches près du sommet ; ce n'est pas un interstice (u y est régulière)."""
    merged = union([disk((0, 0), 2), Polygon([(1.5, -1.5), (4.5, -1.5), (4.5, 1.5), (1.5, 1.5)])])
    field = size_field([merged], merged.inside, 0.15, 1.0)
    crossings = np.array([[1.5, np.sqrt(4 - 1.5**2)], [1.5, -np.sqrt(4 - 1.5**2)]])
    near = np.linalg.norm(field.points[:, None] - crossings[None], axis=-1).min(axis=1) < 0.15
    assert not near.any()


def test_slot_still_a_gap():
    """Fente de largeur 0.1 à fond plat (deux coins saillants de D sur le chemin d'un bord à
    l'autre) : toujours un interstice, au moins trois éléments dans la largeur."""
    slot = Polygon([(0, 0), (3, 0), (3, 2), (1.55, 2), (1.55, 0.8), (1.45, 0.8), (1.45, 2), (0, 2)])
    field = size_field([slot], slot.inside, 0.15, 1.0)
    in_slot = (np.abs(field.points[:, 0] - 1.5) < 0.06) & (field.points[:, 1] > 1.0)
    assert in_slot.sum() > 10
    assert field.sizes[in_slot].max() <= 0.1 / 3 + 1e-12


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
