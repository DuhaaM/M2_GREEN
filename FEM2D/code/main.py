"""Validation (solution manufacturée, étude de convergence) et cas d'application.

Problème : -Δu + k u = f dans D = Omega - beta, u = 0 sur dOmega, u = g_beta sur dbeta
(k >= 0 constant, option --k, 1 par défaut).

Point d'entrée (sections 6 et 7 de main.tex). Le code est réparti en modules :
    config.py (paramètres), cases.py (obstacles et cas), plots.py (figures),
    validation.py (convergence, erreur par élément), ui.py (mode interactif).
Lancer :
    python3 main.py                       menu : obstacles composés, validation ou tous les cas
    python3 main.py puzzle                obstacle choisi directement (voir --help)
    python3 main.py --step fichier.step   obstacle lu dans un fichier CAO
    python3 main.py --obstacle disque 0 0 1 --obstacle polygone 4 0
                                          plusieurs obstacles, type choisi pour chacun
    python3 main.py puzzle --gauss 2 3    f gaussienne avec f(2, 3) = 1 (option --sigma)
    python3 main.py tous                  tous les cas en parallèle, sans validation du maillage
    python3 main.py erreurs --h 0.25      erreur de chaque élément (cas de validation), au clic
Le maillage est affiché et doit être validé avant l'assemblage et la résolution ; la source f
(celle du cas, ou une gaussienne f(x0, y0) = 1) se choisit en cliquant (x0, y0) sur le maillage
ou en saisissant x0, y0 sous la figure, ou à défaut au terminal après validation.
Les figures sont écrites dans le dossier resultats/ à côté de ce fichier, un sous-dossier par
obstacle (maillage.png, solution*.png), plus resultats/validation/ ; suffixe _k<valeur> pour les
solutions et la validation si k n'est pas 1.
"""
import argparse
import os

from cases import CASES, Case, check_spec, composed_case, step_file_case, with_gauss
from config import K, OUT, SIGMA, MeshParams
from ui import choose_case, interactive
from validation import run_all, run_convergence, show_errors

MODES = ["validation", "tous", "erreurs"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("obstacle", nargs="?", choices=list(CASES) + MODES,
                        help="obstacle (sans argument : menu)")
    parser.add_argument("--step", metavar="FICHIER", help="obstacle lu dans un fichier STEP / IGES / BREP")
    parser.add_argument("--obstacle", nargs="+", action="append", metavar="TYPE", dest="components",
                        help="composante de beta, à répéter : disque x y R | cacahuete x y r0 eps | "
                             "couronne x y R r | "
                             "polygone x y [x1 y1 ...] | fichier chemin x y [echelle]")
    parser.add_argument("--echelle", type=float, default=1.0, help="facteur d'échelle du fichier CAO")
    parser.add_argument("--k", type=float, default=K, help=f"coefficient k >= 0 de -Δu + k u = f [{K:g}]")
    parser.add_argument("--gauss", nargs=2, type=float, metavar=("X0", "Y0"),
                        help="f gaussienne avec f(X0, Y0) = 1, centre dans Omega = [0, Lx] x [0, Ly] "
                             "(sans cette option, la source est demandée après validation du maillage)")
    parser.add_argument("--sigma", type=float, default=SIGMA, help=f"écart-type de la gaussienne [{SIGMA:g}]")
    parser.add_argument("-y", "--oui", action="store_true", help="valider le maillage sans l'afficher")
    for name, value in vars(MeshParams()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value, dest=name)
    args = parser.parse_args()
    if args.k < 0:
        parser.error("--k : k doit être positif ou nul (sinon a n'est plus coercive)")
    if args.sigma <= 0:
        parser.error("--sigma : sigma doit être strictement positif")
    if min(args.h, args.h_beta, args.delta) <= 0:
        parser.error("--h, --h-beta, --delta : valeurs strictement positives attendues")
    if args.raffinement < 0:
        parser.error("--raffinement : valeur positive ou nulle attendue (0 : pas de raffinement)")
    if args.m_beta <= args.h / 2:
        parser.error(f"--m-beta : il faut m_beta > h/2 = {args.h / 2:g} "
                     "(sinon des cellules de la grille peuvent toucher beta)")
    for spec in args.components or []:
        try:
            check_spec(spec)
        except ValueError as err:
            parser.error(f"--obstacle {err}")
    if args.step and not os.path.isfile(args.step):
        parser.error(f"--step : fichier introuvable : {args.step}")
    return args


def main():
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    params = MeshParams(args.h, args.h_beta, args.delta, args.m_beta, args.raffinement)
    if args.components:
        choice = composed_case(args.components)
    elif args.step:
        choice = step_file_case(args.step, args.echelle)
    elif args.obstacle:
        choice = CASES.get(args.obstacle, args.obstacle)
    else:
        choice = choose_case(params)
    if isinstance(choice, Case):
        if args.gauss:
            choice = with_gauss(choice, *args.gauss, args.sigma)
        choice = interactive(choice, params, args.oui, args.k, ask_source=not (args.gauss or args.oui),
                             sigma=args.sigma)
    if choice == "validation":
        run_convergence(args.k)
    elif choice == "tous":
        run_all(params, args.k)
    elif choice == "erreurs":
        show_errors(params.h, args.k)
    if choice != "quitter":
        print(f"\nFigures écrites dans {OUT}")


if __name__ == "__main__":
    main()
