"""Génère cao/obstacle.step : exemple d'obstacle CAO (rectangle arrondi + disque).

Le bord mêle segments, arcs et coins rentrants. Lancer :  python3 cao/make_obstacle.py
"""
import os

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
rect = occ.addRectangle(0.0, 0.0, 0.0, 3.0, 1.0, roundedRadius=0.3)
circ = occ.addDisk(0.8, 1.2, 0.0, 0.7, 0.7)
occ.fuse([(2, rect)], [(2, circ)])
occ.synchronize()
gmsh.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "obstacle.step"))
gmsh.finalize()
