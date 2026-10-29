"""Expressions saisies par l'utilisateur (--f, --a, --k, --cl) : fonctions de (x, y) évaluées avec numpy.

Syntaxe Python restreinte : nombres, x, y, Lx, Ly (taille de Omega), pi, e, opérateurs
+ - * / ** (ou ^), comparaisons (valent 0 ou 1, ex. 1 + 9 * (x > 3)) et les fonctions de
FUNCTIONS. Tout le reste (attributs, indices, noms inconnus...) est refusé avant l'évaluation.
"""
import ast

import numpy as np

FUNCTIONS = {name: getattr(np, name) for name in [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
    "exp", "log", "log10", "sqrt", "abs", "hypot", "minimum", "maximum", "where", "sign",
    "floor", "ceil"]}
# Nombre d'arguments : les fonctions universelles de numpy acceptent un argument de plus (out),
# qui écrirait le résultat dans x ou y
ARITY = dict.fromkeys(FUNCTIONS, 1) | dict.fromkeys(["arctan2", "hypot", "minimum", "maximum"], 2) \
    | {"where": 3}
CONSTANTS = {"pi": np.pi, "e": np.e}
VARIABLES = ("x", "y", "Lx", "Ly")
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
          ast.Constant, ast.operator, ast.unaryop, ast.cmpop)


def _compile(text):
    """Code compilé de l'expression, après vérification de chaque nœud de l'arbre syntaxique."""
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ValueError(f"expression invalide : {text}")
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise ValueError(f"« {text} » : construction non autorisée ({type(node).__name__})")
        if isinstance(node, ast.Name) and node.id not in (*VARIABLES, *FUNCTIONS, *CONSTANTS):
            raise ValueError(f"« {text} » : nom inconnu {node.id} (variables : {', '.join(VARIABLES)} ; "
                             f"fonctions : {', '.join(FUNCTIONS)})")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                           or node.func.id not in FUNCTIONS or node.keywords):
            raise ValueError(f"« {text} » : seules les fonctions {', '.join(FUNCTIONS)} sont permises")
        if isinstance(node, ast.Call) and len(node.args) != ARITY[node.func.id]:
            raise ValueError(f"« {text} » : {node.func.id} attend {ARITY[node.func.id]} argument(s)")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"« {text} » : seuls les nombres sont permis comme constantes")
    return compile(tree, "<expression>", "eval")


def expression(text, Lx, Ly):
    """Fonction (x, y) -> tableau de flottants de la forme de x, pour Omega = [0, Lx] x [0, Ly]."""
    code = _compile(text)

    def fn(x, y):
        names = {**FUNCTIONS, **CONSTANTS, "x": x, "y": y, "Lx": Lx, "Ly": Ly}
        value = eval(code, {"__builtins__": {}}, names)
        return np.broadcast_to(np.asarray(value, dtype=float), np.shape(x)).copy()

    return fn


def check(text):
    """Lève ValueError si l'expression est refusée ou ne s'évalue pas sur des tableaux."""
    x = np.linspace(0.1, 0.9, 4)
    try:
        with np.errstate(all="ignore"):
            expression(text, 1.0, 1.0)(x, x)
    except ValueError:
        raise
    except Exception as err:
        raise ValueError(f"« {text} » : {err}")


def number_or_expression(text):
    """Valeur d'une option : flottant si le texte est un nombre, sinon expression vérifiée (str)."""
    try:
        return float(text)
    except ValueError:
        check(text)
        return text


def coefficient(value, Lx, Ly):
    """Constante (flottant) telle quelle, expression (str) changée en fonction (x, y)."""
    return expression(value, Lx, Ly) if isinstance(value, str) else value


def fmt(value):
    """Affichage d'une constante (%g) ou d'une expression (texte)."""
    return value if isinstance(value, str) else f"{value:g}"
