import numpy as np
import time
import Mesh
import matplotlib.pyplot as plt

tic = time.time()

Pol = np.array([[25, 25],
                [75, 25],
                [75, 75],
                [25, 75]])

e = 5

mesh = Mesh.Mesh(Pol, e)
mesh.showcontour()
mesh.generate()
mesh.showmesh()
mesh.findvertindex()

Geometry = {'coord': mesh.coord,
            'tri': mesh.Tri,
            'n_el': mesh.n_el,
            'n_nodes': mesh.n_nodes,
            'n_nodes_edges': mesh.n_nodes_edges,
            'vertice_index': mesh.vertices_index}

np.save('Geometry.npy', Geometry)

toc = time.time() - tic
print(toc)
plt.show()