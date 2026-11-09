"""Coefficients variables, conditions de Neumann et de Robin, flux, expressions et exports."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from assembly import edge_points, element_errors, total_errors
from cases import DISK
from config import LX, LY
from expressions import check, expression, number_or_expression
from geometry import annulus, disk
from mesh import build_mesh
from post import boundary_fluxes, domain_integrals, export_csv, export_vtu
from solver import BC, build_system, solve, solve_system

(CX, CY), R = DISK["center"], DISK["R"]


def disk_mesh(h):
    return build_mesh(disk(DISK["center"], R), LX, LY, h, 0.3 * h, 1.6 * h, 0.9 * h)


# u, a, k de -div(a grad u) + k u = f ; k et alpha (Robin) donnés par des expressions
def u_ex(x, y):
    return np.cos(0.7 * x) * np.exp(0.2 * y) + 0.1 * x * y


def grad_u(x, y):
    return (-0.7 * np.sin(0.7 * x) * np.exp(0.2 * y) + 0.1 * y,
            0.2 * np.cos(0.7 * x) * np.exp(0.2 * y) + 0.1 * x)


def a(x, y):
    return 1 + 0.5 * np.sin(x) * np.cos(y)


K_EXPR, ALPHA_EXPR = "1 + x/Lx", "1 + y"
k, alpha = expression(K_EXPR, LX, LY), expression(ALPHA_EXPR, LX, LY)


def f(x, y):
    ux, uy = grad_u(x, y)
    ax, ay = 0.5 * np.cos(x) * np.cos(y), -0.5 * np.sin(x) * np.sin(y)
    lap = -0.45 * np.cos(0.7 * x) * np.exp(0.2 * y)
    return -a(x, y) * lap - (ax * ux + ay * uy) + k(x, y) * u_ex(x, y)


def flux_n(x, y, nx, ny):
    """a du/dn pour la normale (nx, ny)."""
    ux, uy = grad_u(x, y)
    return a(x, y) * (ux * nx + uy * ny)


BCS = [  # haut, gauche : Dirichlet u = u_ex (défaut dirichlet, g remplacé)
    ("omega", BC("dirichlet", u_ex)),
    ("beta", BC("neumann", lambda x, y: flux_n(x, y, -(x - CX) / R, -(y - CY) / R))),
    ("bas", BC("robin", lambda x, y: flux_n(x, y, 0, -1) + 2 * u_ex(x, y), 2.0)),
    ("droite", BC("robin", lambda x, y: flux_n(x, y, 1, 0) + alpha(x, y) * u_ex(x, y), alpha)),
]


def trapezoid(values, s):
    """Règle des trapèzes (np.trapezoid n'existe qu'à partir de numpy 2)."""
    return np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(s))


def solve_manufactured(h):
    mesh = disk_mesh(h)
    system = build_system(mesh, f, k, a, BCS)
    U = solve_system(system)
    return mesh, system, U


def test_convergence_mixed_conditions():
    """a, k variables, Neumann sur dbeta courbe, Robin sur deux côtés : ordres 3 (L2) et 2 (H1)."""
    hs = [0.5, 0.25, 0.125]
    errs = []
    for h in hs:
        mesh, _, U = solve_manufactured(h)
        errs.append(total_errors(element_errors(mesh, U, u_ex, lambda x, y: grad_u(x, y))))
    errs = np.array(errs)
    orders = np.log(errs[:-1] / errs[1:]) / np.log(2)
    assert (orders[:, 0] > 2.7).all(), orders
    assert (orders[:, 1] > 1.8).all(), orders


def test_fluxes_and_balance():
    """Bilan discret exact ; flux de Dirichlet (résidu) et de Neumann proches des valeurs exactes."""
    mesh, system, U = solve_manufactured(0.25)
    fluxes = boundary_fluxes(mesh, system, U)
    _, _, int_f, int_ku = domain_integrals(mesh, U, f, k)
    assert sum(fluxes.values()) == pytest.approx(int_f - int_ku, rel=1e-10)

    s = np.linspace(0, 1, 200001)
    haut = -trapezoid(flux_n(LX * s, LY + 0 * s, 0, 1), LX * s)
    t = 2 * np.pi * s
    x, y = CX + R * np.cos(t), CY + R * np.sin(t)
    beta = -trapezoid(flux_n(x, y, -np.cos(t), -np.sin(t)), R * t)
    assert fluxes["haut"] == pytest.approx(haut, rel=1e-4)  # coin haut-gauche partagé
    assert fluxes["beta1"] == pytest.approx(beta, rel=1e-4)


def test_boundary_edges():
    """Arêtes de bord : longueurs des côtés de Omega et de dbeta, nœuds de bord tous couverts."""
    mesh = disk_mesh(0.5)
    length = {name: sum(wl.sum() for *_, wl in edge_points(mesh, mesh.edges[mesh.edge_zone == z]))
              for z, name in enumerate(mesh.zones)}
    assert length["bas"] == pytest.approx(LX) and length["haut"] == pytest.approx(LX)
    assert length["gauche"] == pytest.approx(LY) and length["droite"] == pytest.approx(LY)
    assert length["beta1"] == pytest.approx(2 * np.pi * R, rel=1e-5)
    assert set(np.unique(mesh.edges)) == set(mesh.bnd)


def test_default_matches_dirichlet():
    """Sans conditions : même solution qu'avec dirichlet 0 donné explicitement sur toutes les zones."""
    mesh = disk_mesh(0.5)
    one = lambda x, y: np.ones_like(x)
    U0 = solve(mesh, one)
    U1 = solve(mesh, one, bcs=[("omega", BC()), ("beta", BC("dirichlet", 0.0))])
    assert np.allclose(U0, U1)


def test_pure_neumann_rejected():
    mesh = disk_mesh(0.5)
    with pytest.raises(ValueError, match="constante"):
        solve(mesh, 1.0, k=0.0, bcs=[("omega", BC("neumann")), ("beta", BC("neumann"))])


def annulus_mesh():
    """Couronne de centre (5, 5), cavité (beta2) séparée du reste de D."""
    return build_mesh(annulus((5.0, 5.0), 2.0, 1.2), 10.0, 10.0, 0.5, 0.15, 0.8, 1.0)


def in_cavity(x, y):
    return (np.hypot(x - 5, y - 5) < 1.0).astype(float)


@pytest.mark.parametrize("bcs, k", [
    ([("beta2", BC("neumann"))], 0.0),                                   # cavité seule
    ([("beta2", BC("robin", 0.0, 0.0))], 0.0),                           # Robin avec alpha = 0
    ([("beta2", BC("neumann"))], lambda x, y: (np.hypot(x - 5, y - 5) > 1.5) * 1.0),  # k nul dans la cavité
    ([("omega", BC("neumann")), ("beta1", BC("neumann"))], 0.0),         # reste de D seul
    ([("beta2", BC("neumann"))], lambda x, y: np.zeros_like(x)),         # k fonction, nul partout
])
def test_cavity_without_control_rejected(bcs, k):
    """Coercivité vérifiée pour chaque composante connexe de D, et pas seulement globalement."""
    with pytest.raises(ValueError, match="constante"):
        solve(annulus_mesh(), 1.0, k=k, bcs=bcs)


@pytest.mark.parametrize("bc, k", [
    (BC("neumann"), in_cavity),          # k > 0 dans la cavité seulement
    (BC("robin", 0.0, 1.0), 0.0),        # alpha > 0
    (BC("dirichlet"), 0.0),
    (BC("neumann"), 1.0),
])
def test_cavity_with_control_accepted(bc, k):
    """Solution bornée : f = 1, aucune valeur démesurée dans la cavité."""
    U = solve(annulus_mesh(), 1.0, k=k, bcs=[("beta2", bc)])
    assert np.abs(U).max() < 1e3


def test_bad_coefficients_rejected():
    mesh = disk_mesh(0.5)
    with pytest.raises(ValueError, match="a > 0"):
        solve(mesh, 1.0, a=lambda x, y: x - 3)
    with pytest.raises(ValueError, match="positif ou nul"):
        solve(mesh, 1.0, bcs=[("bas", BC("robin", 0.0, -1.0))])
    with pytest.raises(ValueError, match="zone de bord inconnue"):
        solve(mesh, 1.0, bcs=[("beta2", BC())])


def test_expressions():
    fn = expression("x^2 + sin(pi*y/Ly) + (x > 1)", 2.0, 1.0)
    x, y = np.array([0.5, 2.0]), np.array([0.5, 0.0])
    assert np.allclose(fn(x, y), x**2 + np.sin(np.pi * y) + (x > 1))
    assert np.allclose(expression("3", 1.0, 1.0)(x, y), 3.0)  # constante : forme de x
    assert number_or_expression("2.5") == 2.5
    for bad in ["__import__('os')", "x.real", "open('f')", "z + 1", "sin(x, y)", "[x]", "'a'"]:
        with pytest.raises(ValueError):
            check(bad)


def test_exports(tmp_path):
    mesh = disk_mesh(0.5)
    U = solve(mesh, 1.0)
    export_vtu(mesh, U, tmp_path / "u.vtu")
    piece = ET.parse(tmp_path / "u.vtu").find(".//Piece")
    assert int(piece.get("NumberOfPoints")) == mesh.n_nodes
    assert int(piece.get("NumberOfCells")) == len(mesh.tri) + len(mesh.quad)
    values = np.array(piece.find(".//PointData/DataArray").text.split(), dtype=float)
    assert np.allclose(values, U)
    export_csv(mesh, U, tmp_path / "u.csv")
    data = np.loadtxt(tmp_path / "u.csv", delimiter=",", skiprows=1)
    assert np.allclose(data, np.column_stack((mesh.coords, U)))
