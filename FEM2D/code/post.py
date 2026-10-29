"""Post-traitement : grandeurs intégrales, flux à travers les zones de bord, exports VTU et CSV.

Flux sortant de D à travers la zone Z (flux de q = -a grad u) : Phi_Z = -int_Z a du/dn.
    Neumann, Robin : a du/dn = g - alpha u est connu, intégré sur les arêtes de Z ;
    Dirichlet      : méthode du résidu, r = A U - F (système complet) vérifie
                     r_i = int_{dD Dirichlet} a du/dn phi_i ; la somme des r_i sur les nœuds de Z
                     donne le flux. Un nœud commun à deux zones de Dirichlet (coin de Omega)
                     n'a qu'un résidu pour deux côtés : chaque côté Z reçoit l'estimation
                     directe int_Z a du_h/dn phi_i, calculée dans le carré du coin, plus une
                     part égale de l'écart entre r_i et la somme des estimations.
Bilan discret exact (aux erreurs d'arrondi près) : sum_Z Phi_Z = int_D f - int_D k u.
"""
import numpy as np

from assembly import edge_points, element_blocks, evaluate, quadrature_points
from reference import q2_gradients, q2_values

# Côtés du carré de référence [-1, 1]^2 (bas, droite, haut, gauche) : nœuds locaux des extrémités,
# points (x^, y^) en fonction de s dans [-1, 1], normale sortante de référence
_QUAD_SIDES = [((0, 1), lambda s: (s, -np.ones_like(s)), (0, -1)),
               ((1, 2), lambda s: (np.ones_like(s), s), (1, 0)),
               ((2, 3), lambda s: (s, np.ones_like(s)), (0, 1)),
               ((3, 0), lambda s: (-np.ones_like(s), s), (-1, 0))]
_GAUSS_S, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def domain_integrals(mesh, U, f, k):
    """(|D|, int u, int f, int k u) par la quadrature de l'assemblage."""
    out = np.zeros(4)
    for _, conn, ref in element_blocks(mesh):
        Ue = U[conn]
        for psi, xq, wd, _ in quadrature_points(mesh, conn, ref):
            uq = Ue @ psi
            out += [wd.sum(), wd @ uq, wd @ evaluate(f, *xq.T), wd @ (evaluate(k, *xq.T) * uq)]
    return out


def _corner_flux(mesh, U, a, node, edge):
    """int_edge a du_h/dn phi_node, edge arête de bord d'un carré Q2 (n normale sortante), ou
    None si l'arête n'est pas un côté de carré."""
    rows = np.flatnonzero(np.isin(mesh.quad[:, :4], edge[:2]).sum(axis=1) == 2)
    if len(rows) != 1:
        return None
    quad = mesh.quad[rows[0]]
    for (i, j), ref_point, n_ref in _QUAD_SIDES:
        if {quad[i], quad[j]} == set(edge[:2]):
            break
    xh, yh = ref_point(_GAUSS_S)
    G, psi = q2_gradients(xh, yh), q2_values(xh, yh)  # (2, 9, nq), (9, nq)
    X = mesh.coords[quad]
    total = 0.0
    for q in range(len(_GAUSS_S)):
        J = X.T @ G[:, :, q].T  # J[d, b] = dx_d / dx^_b
        B = np.linalg.inv(J).T @ G[:, :, q]  # gradients physiques (2, 9)
        normal = np.linalg.inv(J).T @ n_ref
        tangent = J @ np.array([-n_ref[1], n_ref[0]])  # dérivée de x(s) le long du côté
        x = X.T @ psi[:, q]
        total += _GAUSS_W[q] * np.linalg.norm(tangent) * evaluate(a, x[0], x[1]) \
            * (B @ U[quad]) @ normal / np.linalg.norm(normal) * psi[list(quad).index(node), q]
    return float(total)


def boundary_fluxes(mesh, system, U):
    """Flux sortant Phi_Z de chaque zone de bord : {nom de zone : flux}."""
    residual = system.A @ U - system.F
    shared_part = {}  # (nœud partagé, zone) -> part du résidu attribuée à la zone
    dirichlet = [z for z, name in enumerate(mesh.zones) if system.bcs[name].kind == "dirichlet"]
    # Nombre de zones de Dirichlet qui contiennent chaque nœud
    share = np.zeros(mesh.n_nodes)
    nodes_of = {z: np.unique(mesh.edges[mesh.edge_zone == z]) for z in dirichlet}
    for nodes in nodes_of.values():
        share[nodes] += 1
    for node in np.flatnonzero(share > 1):
        zones = [z for z in nodes_of if node in nodes_of[z]]
        est = []
        for z in zones:
            edges = mesh.edges[mesh.edge_zone == z]
            parts = [_corner_flux(mesh, U, system.a, node, e) for e in edges[(edges[:, :2] == node).any(axis=1)]]
            est.append(None if None in parts else sum(parts))
        if None in est:
            est = [0.0] * len(zones)  # nœud hors d'un carré : parts égales
        correction = (residual[node] - sum(est)) / len(zones)
        for z, e in zip(zones, est):
            shared_part[node, z] = e + correction
    fluxes = {}
    for z, name in enumerate(mesh.zones):
        bc = system.bcs[name]
        if z in nodes_of:
            nodes = nodes_of[z]
            own = nodes[share[nodes] == 1]
            fluxes[name] = -(residual[own].sum()
                             + sum(shared_part[i, z] for i in nodes[share[nodes] > 1]))
            continue
        edges, total = mesh.edges[mesh.edge_zone == z], 0.0
        alpha = bc.alpha if bc.kind == "robin" else 0.0
        for psi, xq, wl in edge_points(mesh, edges):
            total += wl @ (evaluate(bc.g, *xq.T) - evaluate(alpha, *xq.T) * (U[edges] @ psi))
        fluxes[name] = -total + 0.0  # + 0.0 : pas de -0 affiché pour un flux nul
    return fluxes


def report(mesh, system, U, f, k):
    """Tableau des grandeurs : intégrales sur D, flux par zone et bilan."""
    area, int_u, int_f, int_ku = domain_integrals(mesh, U, f, k)
    fluxes = boundary_fluxes(mesh, system, U)
    lines = [f"|D| = {area:.6e}, int_D u = {int_u:.6e}, moyenne de u = {int_u / area:.6e}",
             f"int_D f = {int_f:.6e}, int_D k u = {int_ku:.6e}",
             "Flux sortant de D, Phi_Z = -int_Z a du/dn :",
             f"  {'zone':<8} {'condition':<10} {'flux':>14}"]
    for name, flux in fluxes.items():
        lines.append(f"  {name:<8} {system.bcs[name].kind:<10} {flux:14.6e}")
    total, balance = sum(fluxes.values()), int_f - int_ku
    lines += [f"  {'dOmega':<8} {'':<10} {sum(fluxes[s] for s in mesh.zones if not s.startswith('beta')):14.6e}",
              f"  {'dbeta':<8} {'':<10} {sum(v for s, v in fluxes.items() if s.startswith('beta')):14.6e}",
              f"Bilan : somme des flux = {total:.6e}, int_D (f - k u) = {balance:.6e} "
              f"(écart relatif {abs(total - balance) / max(abs(balance), abs(total), 1e-300):.1e})"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------
VTK_QUADRATIC_TRIANGLE, VTK_BIQUADRATIC_QUAD = 22, 28  # mêmes numérotations locales que tri, quad


def export_vtu(mesh, U, path):
    """Maillage P2 / Q2 et u_h au format VTK XML (ParaView, VisIt...), en ASCII."""
    cells = [(mesh.tri, VTK_QUADRATIC_TRIANGLE), (mesh.quad, VTK_BIQUADRATIC_QUAD)]
    conn = np.concatenate([c.ravel() for c, _ in cells])
    offsets = np.cumsum(np.concatenate([np.full(len(c), c.shape[1]) for c, _ in cells]))
    types = np.concatenate([np.full(len(c), t) for c, t in cells])
    points = np.column_stack((mesh.coords, np.zeros(mesh.n_nodes)))

    def array(values, dtype, name=None, components=None, fmt="{:d}"):
        attrs = f' type="{dtype}"' + (f' Name="{name}"' if name else "") \
            + (f' NumberOfComponents="{components}"' if components else "")
        return (f'<DataArray{attrs} format="ascii">\n'
                + " ".join(map(fmt.format, np.ravel(values))) + "\n</DataArray>\n")

    with open(path, "w") as out:
        out.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" '
                  'byte_order="LittleEndian">\n<UnstructuredGrid>\n'
                  f'<Piece NumberOfPoints="{mesh.n_nodes}" NumberOfCells="{len(types)}">\n'
                  '<Points>\n' + array(points, "Float64", components=3, fmt="{:.12g}") + '</Points>\n'
                  '<Cells>\n' + array(conn, "Int64", "connectivity") + array(offsets, "Int64", "offsets")
                  + array(types, "UInt8", "types") + '</Cells>\n'
                  '<PointData Scalars="u">\n' + array(U, "Float64", "u", fmt="{:.12g}")
                  + '</PointData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


def export_csv(mesh, U, path):
    """Valeurs nodales x, y, u (un nœud par ligne)."""
    np.savetxt(path, np.column_stack((mesh.coords, U)), delimiter=",", header="x,y,u", comments="",
               fmt="%.12g")
