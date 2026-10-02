"""Mode interactif : menu, composition des obstacles, validation du maillage, choix de la source."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.widgets import Button, TextBox
import numpy as np

from cases import (KINDS, center_error, composed_case, compute, gauss_center, mesh_summary,
                   polygon_vertices_ok, prepare, with_gauss)
from config import K, SIGMA, MeshParams, out_path
from plots import has_display, mesh_figure, save_figure


def ask(prompt, default=""):
    try:
        answer = input(prompt).strip()
    except EOFError:
        if not default:  # entrée fermée et pas de réponse par défaut : la question boucherait
            raise SystemExit("\nEntrée standard fermée.")
        answer = ""
    return answer or default


def ask_float(prompt, default, lo=-np.inf, hi=np.inf):
    while True:
        answer = ask(f"{prompt} [{default:g}] : ")
        try:
            value = float(answer) if answer else default
        except ValueError:
            print("  Nombre attendu.")
            continue
        if lo <= value <= hi:
            return value
        print(f"  Valeur attendue dans [{lo:g}, {hi:g}].")


# ----------------------------------------------------------------------------
# beta composé : obstacles ajoutés un par un
# ----------------------------------------------------------------------------
def ask_component(k):
    """Type et paramètres de l'obstacle k (Entrée : valeur par défaut)."""
    while True:
        kind = ask(f"Obstacle {k + 1} : type ({', '.join(KINDS)}) [disque] : ", "disque").lower()
        if kind in KINDS:
            break
        print("Type inconnu.")
    print(f"  {KINDS[kind][0]}")
    spec = [kind]
    if kind == "fichier":
        while not os.path.isfile(path := os.path.expanduser(ask("  Chemin du fichier : "))):
            print(f"  Fichier introuvable : {path}")
        spec.append(path)
    names, defaults = KINDS[kind][1], list(KINDS[kind][2])
    defaults[0] = 4.0 * k  # obstacles placés côte à côte par défaut
    for name, value in zip(names, defaults):
        spec.append(ask_float(f"  {name}", value))
    if kind == "polygone":
        while True:
            answer = ask("  Sommets x1 y1 x2 y2 ... [L] : ")
            try:
                vertices = [float(v) for v in answer.replace(",", " ").split()]
            except ValueError:
                print("  Nombres attendus.")
                continue
            if polygon_vertices_ok(vertices):
                spec += vertices
                break
            print("  Au moins trois sommets (nombre pair de valeurs).")
    return spec


def preview_mesh(mesh, specs, previous=None):
    """Affiche le maillage en cours de composition, sans bloquer (ferme l'aperçu précédent)."""
    if previous is not None:
        plt.close(previous)
    fig = mesh_figure(mesh)
    fig.axes[0].set_title(f"{len(specs)} obstacle(s) : {', '.join(spec[0] for spec in specs)}\n"
                          + fig.axes[0].get_title())
    if not has_display():
        path = out_path("compose", f"apercu_{len(specs)}.png")
        save_figure(fig, path)
        plt.close(fig)
        print(f"  Maillage enregistré dans {path}")
        return None
    plt.show(block=False)
    plt.pause(0.1)
    return fig


def ask_composed(params=MeshParams()):
    """Obstacles ajoutés un par un ; le maillage est affiché après chaque ajout ou retrait."""
    specs, fig = [], None
    try:
        while True:
            answer = "a"
            if specs:
                answer = ask("[a] ajouter un obstacle, [r] retirer le dernier, [t] terminer [a] : ",
                             "a").lower()[:1]
            if answer == "t":
                return composed_case(specs)
            if answer == "r":
                print(f"Obstacle {len(specs)} retiré.")
                candidate = specs[:-1]
            elif answer == "a":
                candidate = specs + [ask_component(len(specs))]
            else:
                print("Choix invalide.")
                continue
            if not candidate:
                specs = []
                continue
            try:
                mesh, Lx, Ly = prepare(composed_case(candidate), params)
            except ValueError as err:
                print(f"Échec du maillage avec cet obstacle : {err}\nObstacle non ajouté.")
                continue
            specs = candidate
            print(mesh_summary(mesh, Lx, Ly))
            fig = preview_mesh(mesh, specs, fig)
    finally:
        if fig is not None:
            plt.close(fig)


# ----------------------------------------------------------------------------
# Menu et paramètres
# ----------------------------------------------------------------------------
MENU = {
    "composer": "obstacles ajoutés un par un (disque, cacahuète, polygone, fichier CAO)",
    "validation": "étude de convergence (disque, solution manufacturée)",
    "tous": "tous les cas prédéfinis, sans validation du maillage",
    "erreurs": "erreur de chaque élément (disque de validation), au clic",
}


def choose_case(params=MeshParams()):
    """Menu : renvoie un Case, ou l'une des chaînes « validation », « tous », « erreurs »."""
    choices = list(MENU)
    for i, key in enumerate(choices, 1):
        print(f"  {i:2d}. {key:<12} {MENU[key]}")
    while True:
        answer = ask("Choix (numéro ou nom) : ").lower()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            answer = choices[int(answer) - 1]
        if answer == "composer":
            return ask_composed(params)
        if answer in MENU:
            return answer
        print("Choix invalide.")


def ask_params(params):
    """Nouveaux paramètres de maillage (Entrée : valeur conservée)."""
    return MeshParams(**{name: ask_float(f"  {name}", value) for name, value in vars(params).items()})


# ----------------------------------------------------------------------------
# Validation du maillage et choix de la source
# ----------------------------------------------------------------------------
def review_mesh(mesh, case, pick=False, sigma=SIGMA):
    """Affiche le maillage et renvoie (validé, centre). Si pick, le centre (x0, y0) de la source
    gaussienne se choisit d'un clic gauche dans Omega ou en saisissant x0, y0 dans les champs sous
    la figure (Entrée) ; clic droit : annulé. centre = None sinon."""
    path = out_path(case.folder, "maillage.png")
    fig = mesh_figure(mesh)
    save_figure(fig, path)
    if not has_display():
        plt.close(fig)
        print(f"Pas d'affichage disponible : maillage enregistré dans {path}")
        return ask("Valider ce maillage ? [o/n] : ", "n").lower().startswith("o"), None

    decision = {}

    def decide(accepted):
        decision["ok"] = accepted
        plt.close(fig)

    ax = fig.axes[0]
    hi_x, hi_y = mesh.coords.max(axis=0)  # Omega = [0, Lx] x [0, Ly]
    title = ax.get_title()
    mark = ax.plot([], [], "+", color="tab:red", ms=14, mew=2, zorder=5)[0]
    disc = MplCircle((0, 0), sigma, fill=False, ls="--", color="tab:red", visible=False, zorder=5)
    ax.add_patch(disc)

    def show_error(err):
        print(f"Erreur : {err}, centre non retenu.")
        ax.set_title(title + f"\nErreur : {err}", color="tab:red")
        fig.canvas.draw_idle()

    def fill_boxes(text_x, text_y):
        """Champs x0, y0 mis à jour sans déclencher leur validation."""
        for box, text in ((box_x, text_x), (box_y, text_y)):
            box.eventson = False
            box.set_val(text)
            box.eventson = True

    def set_center(x0, y0):
        err = center_error(mesh, hi_x, hi_y, x0, y0)
        if err:
            show_error(err)
            return
        decision["center"] = (x0, y0)
        mark.set_data([x0], [y0])
        disc.center = (x0, y0)
        disc.set_visible(True)
        ax.set_title(title + f"\nf gaussienne, f({x0:.6g}, {y0:.6g}) = 1, σ = {sigma:g} "
                             "(clic droit : annuler)", color="black")
        fig.canvas.draw_idle()

    def clear_center():
        decision.pop("center", None)
        mark.set_data([], [])
        disc.set_visible(False)
        fill_boxes("", "")
        ax.set_title(title + "\nsource du cas (clic gauche ou x0, y0 : centre de la gaussienne)",
                     color="black")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
            return  # boutons, champs, ou zoom / déplacement en cours
        if event.button == 3:
            clear_center()
        elif event.button == 1:
            fill_boxes(f"{event.xdata:.4f}", f"{event.ydata:.4f}")
            set_center(event.xdata, event.ydata)

    def on_submit(_):
        """Centre saisi : appliqué dès que x0 et y0 sont tous deux remplis."""
        if not (box_x.text.strip() and box_y.text.strip()):
            return
        try:
            x0, y0 = float(box_x.text), float(box_y.text)
        except ValueError:
            show_error(f"« {box_x.text} », « {box_y.text} » : nombres attendus")
            return
        set_center(x0, y0)

    fig.subplots_adjust(bottom=0.22 if pick else 0.14)
    b_ok = Button(fig.add_axes([0.25, 0.02, 0.2, 0.06]), "Valider", color="#b8e0b8", hovercolor="#8fd08f")
    b_no = Button(fig.add_axes([0.55, 0.02, 0.2, 0.06]), "Refuser", color="#f0b8b8", hovercolor="#e89090")
    b_ok.on_clicked(lambda _: decide(True))
    b_no.on_clicked(lambda _: decide(False))
    if pick:
        box_x = TextBox(fig.add_axes([0.25, 0.10, 0.2, 0.05]), "x0 ")
        box_y = TextBox(fig.add_axes([0.55, 0.10, 0.2, 0.05]), "y0 ")
        box_x.on_submit(on_submit)
        box_y.on_submit(on_submit)
        ax.set_title(title + "\nclic gauche ou x0, y0 : centre de la gaussienne (sinon source du cas)")
        fig.canvas.mpl_connect("button_press_event", on_click)
        print(f"Maillage affiché : cliquer dans Omega = [0, {hi_x:g}] x [0, {hi_y:g}] ou saisir x0, y0 "
              "sous la figure (Entrée) pour centrer une source gaussienne (facultatif), "
              "puis cliquer sur « Valider » ou « Refuser ».")
    else:
        print("Maillage affiché : cliquer sur « Valider » ou « Refuser ».")
    plt.show()
    if "ok" not in decision:  # fenêtre fermée sans cliquer
        decision["ok"] = ask("Valider ce maillage ? [o/n] : ", "n").lower().startswith("o")
    return decision["ok"], decision.get("center")


def ask_gauss(case, mesh, Lx, Ly):
    """Choix de la source : celle du cas, ou une gaussienne f(x0, y0) = 1 de centre choisi dans Omega."""
    current = "gaussienne" if case.f else "f = 1"
    answer = ask(f"Source : [Entrée] celle du cas ({current}), [g] gaussienne centrée en (x0, y0) : ")
    if not answer.lower().startswith("g"):
        return case
    print(f"  Omega = [0, {Lx:g}] x [0, {Ly:g}]")
    while True:
        x0 = ask_float("  x0", Lx / 2, 0, Lx)
        y0 = ask_float("  y0", Ly / 2, 0, Ly)
        err = center_error(mesh, Lx, Ly, x0, y0)
        if not err:
            break
        print(f"  Erreur : {err}, choisir un autre point.")
    sigma = ask_float("  sigma", SIGMA, 1e-12)
    return with_gauss(case, x0, y0, sigma)


def interactive(case, params, auto_accept=False, k=K, ask_source=True, sigma=SIGMA):
    """Maillage -> validation -> calcul. En cas de refus : autres paramètres ou autre obstacle.

    Renvoie None une fois le calcul fait, « quitter », ou le choix fait dans le menu
    (« validation », « tous », « erreurs ») si l'utilisateur change d'obstacle.
    """
    while True:
        print(f"\n=== {case.name} : h = {params.h:g}, h_beta = {params.h_beta:g}, "
              f"delta = {params.delta:g}, m_beta = {params.m_beta:g}, k = {k:g} ===")
        try:
            mesh, Lx, Ly = prepare(case, params)
        except ValueError as err:
            print(f"Échec du maillage : {err}")
            accepted = False
        else:
            print(mesh_summary(mesh, Lx, Ly))
            if gauss_center(case) is not None:
                err = center_error(mesh, Lx, Ly, *gauss_center(case))
                if err:
                    raise SystemExit(f"Erreur : --gauss : {err}.")
            accepted, center = (True, None) if auto_accept else review_mesh(mesh, case, ask_source, sigma)
        if accepted:
            if center is not None:  # centre cliqué sur le maillage
                case = with_gauss(case, *center, sigma)
                print(f"Source gaussienne : f({center[0]:.4g}, {center[1]:.4g}) = 1, sigma = {sigma:g}")
            elif ask_source and not has_display():
                case = ask_gauss(case, mesh, Lx, Ly)  # pas de fenêtre où cliquer : question au terminal
            print("Maillage validé : assemblage et résolution...")
            print(compute(case, mesh, Lx, Ly, k))
            return None
        answer = ask("Maillage refusé. [p] modifier les paramètres, [o] changer d'obstacle, "
                     "[q] quitter : ", "q").lower()
        if answer.startswith("p"):
            params = ask_params(params)
        elif answer.startswith("o"):
            case = choose_case(params)
            if isinstance(case, str):
                return case
        else:
            return "quitter"
