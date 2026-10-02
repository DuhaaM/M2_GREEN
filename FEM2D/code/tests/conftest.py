"""Les modules du code sont importés depuis le dossier parent ; figures sans fenêtre."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
