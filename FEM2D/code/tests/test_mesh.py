"""Conformité du maillage mixte Q2 / P2 et garde-fous sur les paramètres et la géométrie."""
import numpy as np
import pytest

from assembly import check_jacobians, element_errors
from cases import CASES, DISK, POLYGON, center_error, prepare
from config import LX, LY, MeshParams
from geometry import (CurveObstacle, HoledObstacle, MultiObstacle, Obstacle, Polygon, annulus, disk,
                      domain_from_obstacle, from_step, peanut, union)
from mesh import build_mesh
from solver import solve

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


def shoelace(vertices):
    x, y = np.asarray(vertices).T
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


CROSS = [Polygon([(-2, -0.1), (2, -0.1), (2, 0.1), (-2, 0.1)]),
         Polygon([(-0.1, 2), (0.1, 2), (0.1, -2), (-0.1, -2)])]  # orientations opposées


def test_union_disks():
    """Deux disques sécants : un seul contour, aire pi R^2 x 2 - lunule, 2 coins aux intersections."""
    merged = union([disk((0, 0), 1), disk((1, 0), 1)])
    assert isinstance(merged, CurveObstacle) and len(merged.curves) == 2
    lens = 2 * np.arccos(0.5) - 0.5 * np.sqrt(3)
    assert np.isclose(shoelace(merged.vertices), 2 * np.pi - lens, rtol=1e-5)
    corners = merged.gamma(merged.breakpoints[:-1])
    assert np.allclose(np.sort(corners[:, 1]), [-np.sqrt(3) / 2, np.sqrt(3) / 2])
    assert np.allclose(corners[:, 0], 0.5)


def test_union_polygons_exact():
    merged = union(CROSS)
    assert len(merged.curves) == 12  # 8 sommets des bras + 4 coins rentrants
    assert np.isclose(shoelace(merged.vertices), 1.56, rtol=1e-12)


def test_union_contained():
    """Composante contenue dans une autre : la plus grande est gardée telle quelle."""
    big = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    assert union([disk((0.1, 0), 0.3), big]) is big


def test_union_ring_cavity():
    """Anneau de disques sécants : beta troué, la cavité centrale reste dans D."""
    ring = union([disk((2 * np.cos(a), 2 * np.sin(a)), 1.2) for a in np.arange(6) * np.pi / 3])
    assert isinstance(ring, HoledObstacle) and len(ring.holes) == 1
    assert list(ring.inside(np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]))) == [False, True, False]
    assert np.isclose(ring.distance(np.zeros((1, 2)))[0], 2 - 1.2, rtol=1e-5)


def test_union_with_holed_component():
    """Couronne coupée par un disque qui entre dans la cavité : la cavité est réduite, pas bouchée ;
    un disque qui recouvre toute la cavité la bouche ; un îlot dans la cavité est refusé."""
    crown = annulus((0, 0), 2, 1)
    merged = union([crown, disk((1.5, 0), 0.8)])
    assert isinstance(merged, HoledObstacle)
    assert list(merged.inside(np.array([[0.0, 0.0], [0.9, 0.0], [-1.5, 0.0]]))) == [False, True, True]
    assert not isinstance(union([crown, disk((0, 0), 1.3)]), HoledObstacle)
    with pytest.raises(ValueError, match="cavité"):
        union([crown, disk((0, 0), 0.5)])


def test_union_of_holed_obstacles():
    """Couronnes dont les cavités se coupent : 3 cavités (deux croissants et une lentille) ;
    bord de trou intact d'un polygone : découpé à ses sommets seulement (4 courbes)."""
    lens = union([annulus((0, 0), 2, 1.4), annulus((1.5, 0), 2, 1.4)])
    assert isinstance(lens, HoledObstacle) and len(lens.holes) == 3
    pts = np.array([[-0.5, 0.0], [0.75, 0.0], [2.0, 0.0], [0.75, 1.3]])
    assert list(lens.inside(pts)) == [False, False, False, True]
    square = HoledObstacle(Polygon([(1.5, -1.5), (4.5, -1.5), (4.5, 1.5), (1.5, 1.5)]),
                           [Polygon([(2.2, -0.8), (3.8, -0.8), (3.8, 0.8), (2.2, 0.8)])])
    merged = union([annulus((0, 0), 2, 1), square])
    assert len(merged.holes) == 2
    hole = next(h for h in merged.holes if len(h.curves) != 1)
    assert len(hole.curves) == 4
    np.testing.assert_allclose(hole.gamma(hole.breakpoints[:-1])[:, 0] % 1.6, 0.6, atol=1e-12)
    obstacle, Lx, Ly, _ = domain_from_obstacle(lens, 0.5, 2.0, 0.5)
    mesh = build_mesh(obstacle, Lx, Ly, 0.5, 0.1, 0.8, 0.5, refine=1.0)
    assert len(mesh.beta_points) == 4


def test_holed_obstacle_checks():
    with pytest.raises(ValueError, match="strictement intérieur"):
        HoledObstacle(disk((0, 0), 1), [disk((0.8, 0), 0.5)])
    with pytest.raises(ValueError, match="trous 1 et 2"):
        HoledObstacle(disk((0, 0), 3), [disk((0, 0), 1), disk((1, 0), 1)])
    with pytest.raises(ValueError, match="0 < r < R"):
        annulus((0, 0), 1, 2)


def test_annulus_mesh():
    """Couronne : |D| exact au bord près, et dans la cavité (disque de rayon 1, u = 0 au bord)
    u_h proche de la solution exacte de -Δu + u = 1 : u = 1 - I0(r) / I0(1)."""
    from scipy.special import i0
    obstacle, Lx, Ly, shift = domain_from_obstacle(annulus((0, 0), 2, 1), 0.5, 2.0, 0.5)
    mesh = build_mesh(obstacle, Lx, Ly, 0.5, 0.1, 0.8, 0.5)
    assert len(mesh.beta_points) == 2
    errs = element_errors(mesh, np.zeros(mesh.n_nodes), lambda x, y: 0 * x,
                          lambda x, y: (0 * x, 0 * x))
    area = sum(a.sum() for a, *_ in errs.values())
    assert np.isclose(area, Lx * Ly - 3 * np.pi, rtol=1e-6)
    U = solve(mesh, lambda x, y: np.ones_like(x), k=1.0)
    r = np.linalg.norm(mesh.coords - shift, axis=1)
    cavity = r < 1 - 1e-9
    assert cavity.sum() > 100
    assert np.abs(U[cavity] - (1 - i0(r[cavity]) / i0(1))).max() < 1e-4


def test_holed_polygon_mesh_area():
    """Carré percé d'un carré : |D| = Lx Ly - |beta| exact (bords polygonaux)."""
    square = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    hole = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    obstacle, Lx, Ly, _ = domain_from_obstacle(HoledObstacle(square, [hole]), 0.5, 2.0, 0.5)
    mesh = build_mesh(obstacle, Lx, Ly, 0.5, 0.1, 0.8, 0.5, refine=1.0)
    errs = element_errors(mesh, np.zeros(mesh.n_nodes), lambda x, y: 0 * x,
                          lambda x, y: (0 * x, 0 * x))
    area = sum(a.sum() for a, *_ in errs.values())
    assert np.isclose(area, Lx * Ly - 8, rtol=1e-12)


def test_center_in_cavity():
    """Un point de la cavité est dans D, un point de la couronne est dans beta."""
    mesh, Lx, Ly = prepare(CASES["couronne"], MeshParams())
    c = mesh.beta_points[1].mean(axis=0)  # centre de la cavité
    assert center_error(mesh, Lx, Ly, *c) is None
    assert "dans l'obstacle" in center_error(mesh, Lx, Ly, c[0] + 1.1, c[1])


def test_from_step_holed_face(tmp_path):
    """Face percée de deux trous, plus une face séparée : MultiObstacle(HoledObstacle, disque)."""
    import gmsh
    path = str(tmp_path / "troue.step")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        plate = occ.addRectangle(0, 0, 0, 4, 3)
        occ.cut([(2, plate)], [(2, occ.addDisk(1, 1.5, 0, 0.6, 0.6)),
                               (2, occ.addRectangle(2.3, 0.8, 0, 1, 1.4))])
        occ.addDisk(6, 1.5, 0, 1, 1)
        occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()
    beta = from_step(path)
    holed = [c for c in beta.components if isinstance(c, HoledObstacle)]
    assert len(beta.components) == 2 and len(holed) == 1 and len(holed[0].holes) == 2
    assert not beta.inside(np.array([[1.0, 1.5], [2.8, 1.5]])).any()  # dans les trous
    assert beta.inside(np.array([[0.2, 0.2], [6.0, 1.5]])).all()


def test_union_keeps_disjoint():
    a, b, c = disk((0, 0), 1), disk((4, 0), 1), disk((4.5, 0), 1)
    multi = union([a, b, c])
    assert isinstance(multi, MultiObstacle) and multi.components[0] is a
    assert isinstance(multi.components[1], CurveObstacle)


def test_union_tangent_rejected():
    with pytest.raises(ValueError, match="tangent"):
        union([disk((0, 0), 1), disk((2, 0), 1)])


def test_union_mesh_area():
    """Maillage de la croix fusionnée : |D| = Lx Ly - |beta| exact (bord polygonal)."""
    obstacle, Lx, Ly, _ = domain_from_obstacle(union(CROSS), 0.5, 2.0, 0.5)
    mesh = build_mesh(obstacle, Lx, Ly, 0.5, 0.05, 0.8, 0.5)
    errs = element_errors(mesh, np.zeros(mesh.n_nodes), lambda x, y: 0 * x,
                          lambda x, y: (0 * x, 0 * x))
    area = sum(a.sum() for a, *_ in errs.values())
    assert np.isclose(area, Lx * Ly - 1.56, rtol=1e-12)


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
