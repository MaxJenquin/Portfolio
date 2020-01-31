import numpy as np
import FEA_FUN as fea
import time
import MESH_GENERATOR.Mesh as Mesh
import caffeine

caffeine.on(display = True)

ntrain = 500

for iter in range(ntrain):

    tic = time.time()

    a = 0
    b = 0
    while a == 0 or b == 0:
        a = round(np.random.rand() * 100, 1)
        b = round(np.random.rand() * a, 1)

    if a / b > 25:
        continue

    Q = round(np.random.rand() * 100, 1)
    T_edges = round(np.random.rand() * 100, 1)
    k = 1

    print('Iter = ', iter + 1, ' | Time :', time.asctime(), ' | Omega = ', a, ' x ', b)

    # MESH GENERATION -------------------------------------------------------------------------------------------------#

    e = round(a / 15, 1)
    if 4 * e > b:
        e = round(b / 4, 1)

    if e == 0:
        continue

    Pol = np.array([[0, 0],
                    [a, 0],
                    [a, b],
                    [0, b]])

    mesh = Mesh.Mesh(Pol, e)
    mesh.generate()

    coord = mesh.coord
    tri = mesh.Tri
    n_el = mesh.n_el
    n_nodes = mesh.n_nodes
    n_nodes_edges = mesh.n_nodes_edges

    # BOUNDARY CONDITIONS PROCESSING ----------------------------------------------------------------------------------#

    ImpT = fea.impTconvert(np.array([[1, n_nodes_edges, T_edges]]))
    ImpDT, DTnodes = fea.impDTconvert(np.array([[1, 1, np.NaN]]))

    # SHAPE FUNCTIONS -------------------------------------------------------------------------------------------------#

    xi, eta, N, N1, N2, N3, B_loc = fea.shapefun()

    # STIFFNESS & FORCE MATRICES --------------------------------------------------------------------------------------#

    K, f = fea.FEAmatrix(n_nodes, ImpDT, n_el, coord, tri, B_loc, DTnodes, k, Q, N, N1, N2, N3, xi, eta)

    # SOLVER ----------------------------------------------------------------------------------------------------------#

    U = fea.solverTH(n_nodes, ImpT, K, f)

    # UPDATING DATASET ------------------------------------------------------------------------------------------------#

    Umax = np.max(U)
    toc = time.time() - tic

    Data_heat_FE0 = np.load('Data_heat_FE.npy').item()
    X0 = Data_heat_FE0['X']
    y0 = Data_heat_FE0['y']
    comp_time0 = Data_heat_FE0['comp_time']

    X = np.array([[a, b, Q, T_edges]])
    y = np.array([[Umax]])
    comp_time = np.array([[toc]])

    X = np.concatenate((X0, X), 0)
    y = np.concatenate((y0, y), 0)
    comp_time = np.concatenate((comp_time0, comp_time), 0)

    Data_heat_FE = {'X': X, 'y': y, 'comp_time': comp_time}

    np.save('Data_heat_FE.npy', Data_heat_FE)

