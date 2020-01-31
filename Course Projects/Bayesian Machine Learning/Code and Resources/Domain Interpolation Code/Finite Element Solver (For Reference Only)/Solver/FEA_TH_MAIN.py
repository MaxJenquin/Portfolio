import numpy as np
import sympy as sy
import FEA_FUN as fea
import matplotlib.pyplot as plt
import matplotlib.tri as tr
import time

tic = time.time()

#------------------------------------------------ LOADING PARAMETERS --------------------------------------------------#

# PYTHON
Geometry = np.load('/Users/christophebonneville/Google Drive/Shared Python Project/MESH_GENERATOR/Geometry.npy').item()
coord = Geometry['coord']
tri = Geometry['tri']
n_el = Geometry['n_el']
n_nodes = Geometry['n_nodes']
n_nodes_edges = Geometry['n_nodes_edges']

#------------------------------------------------ PROBLEM PARAMETERS --------------------------------------------------#

k = 1
Q = 2
ImpT = fea.impTconvert(np.array([[1, 19, 10],
                                 [20, 39, 20],
                                 [40, 59, 10],
                                 [60, 79, 20],
                                 [80, 80, 10]]))

ImpT = fea.impTconvert(np.array([[1, 38, 0]]))

ImpDT, DTnodes = fea.impDTconvert(np.array([[0, 0, np.NaN]]))

disp_mesh = 0
cmap_res = 500

#--------------------------------------------------- SHAPE FUNCTIONS --------------------------------------------------#


xi = sy.symbols('xi')
eta = sy.symbols('eta')

N1 = 1 - xi - eta
N2 = xi
N3 = eta

N = np.array([[N1, N2, N3]])
B_loc = np.zeros([2,3])
for j in range(0,3):
    B_loc[0, j] = N[0, j].diff(xi)
    B_loc[1, j] = N[0, j].diff(eta)


#------------------------------------------------- STIFFNESS & FORCE --------------------------------------------------#

K, f = fea.FEAmatrix(n_nodes, ImpDT, n_el, coord, tri, B_loc, DTnodes, k, Q, N, N1, N2, N3, xi, eta)

#------------------------------------------------------- SOLVER -------------------------------------------------------#


U = fea.solverTH(n_nodes, ImpT, K, f)


#--------------------------------------------------- POST-PROCESS -----------------------------------------------------#


x = coord[:, 1]
y = coord[:, 2]
tri = (tri-1).astype(int)
U = np.reshape(U, (len(U),))

fig = plt.figure(1, figsize = (8, 6))
triang = tr.Triangulation(x, y, tri)
plt.tricontourf(triang, U, cmap_res, cmap = plt.cm.jet)
if disp_mesh == 1:
    plt.triplot(x, y, tri, color = 'k', linewidth = 0.2)
plt.colorbar()
plt.axis('equal')
plt.title('FEA Simulation (' + str(n_el) + ' elements & ' + str(n_nodes)+' nodes)')

toc = time.time() - tic
print(toc)

plt.show()