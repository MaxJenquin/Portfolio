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
    x3 = round(np.random.rand() * 50, 0) + 100
    y3 = round(np.random.rand() * 50, 0)
    x4 = round(np.random.rand() * 50, 0) + 100
    y4 = round(np.random.rand() * 50, 0) + 50
    x5 = round(np.random.rand() * 50, 0) + 100
    y5 = round(np.random.rand() * 50, 0) + 100
    x6 = round(np.random.rand() * 50, 0) + 50
    y6 = round(np.random.rand() * 50, 0) + 100
    x7 = round(np.random.rand() * 50, 0)
    y7 = round(np.random.rand() * 50, 0) + 100
    x8 = round(np.random.rand() * 50, 0)
    y8 = round(np.random.rand() * 50, 0) + 50

    v12 = np.array([x2 - x1, y2 - y1])
    v23 = np.array([x3 - x2, y3 - y2])
    v34 = np.array([x4 - x3, y4 - y3])
    v45 = np.array([x5 - x4, y5 - y4])
    v56 = np.array([x6 - x5, y6 - y5])
    v67 = np.array([x7 - x6, y7 - y6])
    v78 = np.array([x8 - x7, y8 - y7])
    v81 = np.array([x1 - x8, y1 - y8])

    n12 = np.linalg.norm(v12)
    n23 = np.linalg.norm(v23)
    n34 = np.linalg.norm(v34)
    n45 = np.linalg.norm(v45)
    n56 = np.linalg.norm(v56)
    n67 = np.linalg.norm(v67)
    n78 = np.linalg.norm(v78)
    n81 = np.linalg.norm(v81)

    a123 = np.arccos(-np.sum(v23 * v12) / n23 / n12)
    a234 = np.arccos(-np.sum(v23 * v34) / n34 / n23)
    a345 = np.arccos(-np.sum(v34 * v45) / n34 / n45)
    a456 = np.arccos(-np.sum(v45 * v56) / n45 / n56)
    a567 = np.arccos(-np.sum(v56 * v67) / n56 / n67)
    a678 = np.arccos(-np.sum(v67 * v78) / n67 / n78)
    a781 = np.arccos(-np.sum(v78 * v81) / n78 / n81)
    a812 = np.arccos(-np.sum(v81 * v12) / n12 / n81)

    while a123 < np.pi / 5 or a234 < np.pi / 5 or a345 < np.pi / 5 or a456 < np.pi / 5 or a567 < np.pi / 5 or a678 < np.pi / 5 \
        or a781 < np.pi / 5 or a812 < np.pi / 5:

        x1 = round(np.random.rand() * 50, 0)
        y1 = round(np.random.rand() * 50, 0)
        x2 = round(np.random.rand() * 50, 0) + 50
        y2 = round(np.random.rand() * 50, 0)
        x3 = round(np.random.rand() * 50, 0) + 100
        y3 = round(np.random.rand() * 50, 0)
        x4 = round(np.random.rand() * 50, 0) + 100
        y4 = round(np.random.rand() * 50, 0) + 50
        x5 = round(np.random.rand() * 50, 0) + 100
        y5 = round(np.random.rand() * 50, 0) + 100
        x6 = round(np.random.rand() * 50, 0) + 50
        y6 = round(np.random.rand() * 50, 0) + 100
        x7 = round(np.random.rand() * 50, 0)
        y7 = round(np.random.rand() * 50, 0) + 100
        x8 = round(np.random.rand() * 50, 0)
        y8 = round(np.random.rand() * 50, 0) + 50

        v12 = np.array([x2 - x1, y2 - y1])
        v23 = np.array([x3 - x2, y3 - y2])
        v34 = np.array([x4 - x3, y4 - y3])
        v45 = np.array([x5 - x4, y5 - y4])
        v56 = np.array([x6 - x5, y6 - y5])
        v67 = np.array([x7 - x6, y7 - y6])
        v78 = np.array([x8 - x7, y8 - y7])
        v81 = np.array([x1 - x8, y1 - y8])

        n12 = np.linalg.norm(v12)
        n23 = np.linalg.norm(v23)
        n34 = np.linalg.norm(v34)
        n45 = np.linalg.norm(v45)
        n56 = np.linalg.norm(v56)
        n67 = np.linalg.norm(v67)
        n78 = np.linalg.norm(v78)
        n81 = np.linalg.norm(v81)

        a123 = np.arccos(-np.sum(v23 * v12) / n23 / n12)
        a234 = np.arccos(-np.sum(v23 * v34) / n34 / n23)
        a345 = np.arccos(-np.sum(v34 * v45) / n34 / n45)
        a456 = np.arccos(-np.sum(v45 * v56) / n45 / n56)
        a567 = np.arccos(-np.sum(v56 * v67) / n56 / n67)
        a678 = np.arccos(-np.sum(v67 * v78) / n67 / n78)
        a781 = np.arccos(-np.sum(v78 * v81) / n78 / n81)
        a812 = np.arccos(-np.sum(v81 * v12) / n12 / n81)

    Q = round(np.random.rand() * 5, 1)
    T_edges1 = round(np.random.rand() * 100, 0)
    T_edges2 = round(np.random.rand() * 100, 0)
    T_edges3 = round(np.random.rand() * 100, 0)
    T_edges4 = round(np.random.rand() * 100, 0)
    T_edges5 = round(np.random.rand() * 100, 0)
    T_edges6 = round(np.random.rand() * 100, 0)
    T_edges7 = round(np.random.rand() * 100, 0)
    T_edges8 = round(np.random.rand() * 100, 0)

    k = 1

    print('Iter = ', iter + 1, ' | Time :', time.asctime())

    # MESH GENERATION -------------------------------------------------------------------------------------------------#

    n_max = np.max([n12, n23, n34, n45, n67, n78, n81])
    n_min = np.min([n12, n23, n34, n45, n67, n78, n81])

    e = round(n_max / 5, 0)
    if 2 * e > n_min:
        e = round(n_min / 2, 0)

    if e == 0:
        continue

    Pol = np.array([[x1, y1],
                    [x2, y2],
                    [x3, y3],
                    [x4, y4],
                    [x5, y5],
                    [x6, y6],
                    [x7, y7],
                    [x8, y8]])

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
                                     [vertices_index[3], vertices_index[4] - 1, T_edges4],
                                     [vertices_index[4], vertices_index[5] - 1, T_edges5],
                                     [vertices_index[5], vertices_index[6] - 1, T_edges6],
                                     [vertices_index[6], vertices_index[7] - 1, T_edges7],
                                     [vertices_index[7], n_nodes_edges, T_edges8]]))

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

    Data_heat_FE0 = np.load('Data_heat_FE_Octagon+BC.npy').item()
    X0 = Data_heat_FE0['X']
    y0 = Data_heat_FE0['y']
    comp_time0 = Data_heat_FE0['comp_time']

    X = np.array([[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, Q, T_edges1, T_edges2, T_edges3, T_edges4, T_edges5, T_edges6, T_edges7, T_edges8]])
    y = np.array([[Umax]])
    comp_time = np.array([[toc]])

    X = np.concatenate((X0, X), 0)
    y = np.concatenate((y0, y), 0)
    comp_time = np.concatenate((comp_time0, comp_time), 0)

    Data_heat_FE = {'X': X, 'y': y, 'comp_time': comp_time}

    np.save('Data_heat_FE_Octagon+BC.npy', Data_heat_FE)

