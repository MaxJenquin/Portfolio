import numpy as np
import FEA_FUN as fea
import time
import MESH_GENERATOR.Mesh as Mesh
import caffeine

caffeine.on(display = True)

ntrain = 1000
for iter in range(ntrain):

    tic = time.time()

    x1 = round(np.random.rand() * 50, 0)
    y1 = round(np.random.rand() * 50, 0)
    x2 = round(np.random.rand() * 50, 0) + 50
    y2 = round(np.random.rand() * 50, 0)
    x3 = round(np.random.rand() * 50, 0) + 50
    y3 = round(np.random.rand() * 50, 0) + 50
    x4 = round(np.random.rand() * 50, 0)
    y4 = round(np.random.rand() * 50, 0) + 50

    v12 = np.array([x2 - x1, y2 - y1])
    v23 = np.array([x3 - x2, y3 - y2])
    v34 = np.array([x4 - x3, y4 - y3])
    v41 = np.array([x1 - x4, y1 - y4])

    n12 = np.linalg.norm(v12)
    n23 = np.linalg.norm(v23)
    n34 = np.linalg.norm(v34)
    n41 = np.linalg.norm(v41)

    a412 = np.arccos(-np.sum(v12 * v41) / n12 / n41)
    a123 = np.arccos(-np.sum(v23 * v12) / n23 / n12)
    a234 = np.arccos(-np.sum(v23 * v34) / n34 / n23)
    a341 = np.arccos(-np.sum(v34 * v41) / n34 / n41)

    while a412 < np.pi / 6 or a123 < np.pi / 6 or a234 < np.pi / 6 or a341 < np.pi / 6 or \
            a412 > 8 * np.pi / 10 or a123 > 8 * np.pi / 10 or a234 > 8 * np.pi / 10 or a341 > 8 * np.pi / 10:

        x1 = round(np.random.rand() * 50, 0)
        y1 = round(np.random.rand() * 50, 0)
        x2 = round(np.random.rand() * 50, 0) + 50
        y2 = round(np.random.rand() * 50, 0)
        x3 = round(np.random.rand() * 50, 0) + 50
        y3 = round(np.random.rand() * 50, 0) + 50
        x4 = round(np.random.rand() * 50, 0)
        y4 = round(np.random.rand() * 50, 0) + 50

        v12 = np.array([x2 - x1, y2 - y1])
        v23 = np.array([x3 - x2, y3 - y2])
        v34 = np.array([x4 - x3, y4 - y3])
        v41 = np.array([x1 - x4, y1 - y4])

        n12 = np.linalg.norm(v12)
        n23 = np.linalg.norm(v23)
        n34 = np.linalg.norm(v34)
        n41 = np.linalg.norm(v41)

        a412 = np.arccos(-np.sum(v12 * v41) / n12 / n41)
        a123 = np.arccos(-np.sum(v23 * v12) / n23 / n12)
        a234 = np.arccos(-np.sum(v23 * v34) / n34 / n23)
        a341 = np.arccos(-np.sum(v34 * v41) / n34 / n41)

    Q = round(np.random.rand() * 100, 0)
    T_edges1 = round(np.random.rand() * 100, 0)
    T_edges2 = round(np.random.rand() * 100, 0)
    T_edges3 = round(np.random.rand() * 100, 0)
    T_edges4 = round(np.random.rand() * 100, 0)

    k = 1

    print('Iter = ', iter + 1, ' | Time :', time.asctime(), ' | x = [', x1, ', ', x2, ', ', x3, ', ', x4, '], y = [', y1, ', ', y2, ', ', y3, ', ', y4, '], T_edges = [', T_edges1, ', ', T_edges2, ', ', T_edges3, ', ', T_edges4, ']')

    # MESH GENERATION -------------------------------------------------------------------------------------------------#

    n_max = np.max([n12, n23, n34, n41])
    n_min = np.min([n12, n23, n34, n41])

    e = round(n_max / 15, 0)
    if 4 * e > n_min:
        e = round(n_min / 4, 0)

    if e == 0:
        continue

    Pol = np.array([[x1, y1],
                    [x2, y2],
                    [x3, y3],
                    [x4, y4]])

    mesh = Mesh.Mesh(Pol, e)
    mesh.generate()
    mesh.findvertindex()

    coord = mesh.coord
    tri = mesh.Tri
    n_el = mesh.n_el
    n_nodes = mesh.n_nodes
    n_nodes_edges = mesh.n_nodes_edges

    # BOUNDARY CONDITIONS PROCESSING ----------------------------------------------------------------------------------#

    vertices_index = mesh.vertices_index
    ImpT = fea.impTconvert(np.array([[1, vertices_index[1] - 1, T_edges1],
                                     [vertices_index[1], vertices_index[2] - 1, T_edges2],
                                     [vertices_index[2], vertices_index[3] - 1, T_edges3],
                                     [vertices_index[3], n_nodes_edges, T_edges4]]))

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

    Data_heat_FE0 = np.load('Data_heat_FE_Quadrilateral+BC.npy').item()
    X0 = Data_heat_FE0['X']
    y0 = Data_heat_FE0['y']
    comp_time0 = Data_heat_FE0['comp_time']

    X = np.array([[x1, y1, x2, y2, x3, y3, x4, y4, Q, T_edges1, T_edges2, T_edges3, T_edges4]])
    y = np.array([[Umax]])
    comp_time = np.array([[toc]])

    X = np.concatenate((X0, X), 0)
    y = np.concatenate((y0, y), 0)
    comp_time = np.concatenate((comp_time0, comp_time), 0)

    Data_heat_FE = {'X': X, 'y': y, 'comp_time': comp_time}

    np.save('Data_heat_FE_Quadrilateral+BC.npy', Data_heat_FE)

