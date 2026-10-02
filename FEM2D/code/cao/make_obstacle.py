"""Génère les exemples d'obstacles CAO de ce dossier.

- obstacle.step : rectangle arrondi + disque (segments, arcs et coins rentrants) ;
- deux_obstacles.step : beta non connexe, deux composantes proches ;
- disques_tangents.step : deux disques de rayon 1 presque tangents (écart EPS < h_beta).
Lancer :  python3 cao/make_obstacle.py
"""
import os

import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.05  # écart entre les deux disques de disques_tangents.step

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ

gmsh.model.add("obstacle")
rect = occ.addRectangle(0.0, 0.0, 0.0, 3.0, 1.0, roundedRadius=0.3)
circ = occ.addDisk(0.8, 1.2, 0.0, 0.7, 0.7)
occ.fuse([(2, rect)], [(2, circ)])
occ.synchronize()
gmsh.write(os.path.join(HERE, "obstacle.step"))

gmsh.model.add("deux_obstacles")
occ.addDisk(0.0, 0.0, 0.0, 0.8, 0.5)
occ.addRectangle(1.3, -0.6, 0.0, 1.0, 1.2, roundedRadius=0.2)
occ.synchronize()
gmsh.write(os.path.join(HERE, "deux_obstacles.step"))

gmsh.model.add("disques_tangents")
occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
occ.addDisk(2.0 + EPS, 0.0, 0.0, 1.0, 1.0)
occ.synchronize()
gmsh.write(os.path.join(HERE, "disques_tangents.step"))
gmsh.finalize()
