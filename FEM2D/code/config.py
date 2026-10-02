"""Paramètres par défaut (section 7.1 de main.tex) et dossier des résultats."""
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "resultats")


def out_path(folder, name):
    """Chemin de la figure name dans resultats/<folder>/, dossier créé au besoin."""
    os.makedirs(os.path.join(OUT, folder), exist_ok=True)
    return os.path.join(OUT, folder, name)


# Omega du cas de validation et paramètres de maillage par défaut
LX, LY = 7.0, 5.0
H, H_BETA, DELTA, M_BETA = 0.5, 0.15, 0.8, 1
K = 1.0  # coefficient de réaction de -Δu + k u = f
SIGMA = 0.5  # écart-type par défaut de la source gaussienne


@dataclass(frozen=True)
class MeshParams:
    h: float = H
    h_beta: float = H_BETA
    delta: float = DELTA
    m_beta: float = M_BETA

    @property
    def d_min(self):
        """Distance minimale entre beta et le bord de Omega (section 3.5)."""
        return self.m_beta + 4 * self.h


def k_suffix(k):
    """Suffixe des figures qui dépendent de k : vide pour la valeur par défaut."""
    return "" if k == K else f"_k{k:g}"
