import numpy as np
import FEA_FUN as fea
import time
import caffeine
import scipy.spatial
caffeine.on(display = True)

ntrain = 1000
for iter in range(ntrain):
    try:
        tic = time.time()

        x1 = round(np.random.rand() * 30, 0) + 10
        y1 = round(np.random.rand() * 30, 0) + 10
        x2 = round(np.random.rand() * 30, 0) + 60
        y2 = round(np.random.rand() * 30, 0) + 10
        x3 = round(np.random.rand() * 30, 0) + 60
        y3 = round(np.random.rand() * 30, 0) + 60
        x4 = round(np.random.rand() * 30, 0) + 10
        y4 = round(np.random.rand() * 30, 0) + 60

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

        while a412 < np.pi / 10 or a123 < np.pi / 10 or a234 < np.pi / 10 or a341 < np.pi / 10 or \
                a412 > 8 * np.pi / 10 or a123 > 8 * np.pi / 10 or a234 > 8 * np.pi / 10 or a341 > 8 * np.pi / 10:

            x1 = round(np.random.rand() * 30, 0) + 10
            y1 = round(np.random.rand() * 30, 0) + 10
            x2 = round(np.random.rand() * 30, 0) + 60
            y2 = round(np.random.rand() * 30, 0) + 10
            x3 = round(np.random.rand() * 30, 0) + 60
            y3 = round(np.random.rand() * 30, 0) + 60
            x4 = round(np.random.rand() * 30, 0) + 10
            y4 = round(np.random.rand() * 30, 0) + 60

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

        Q = round(np.random.rand() * 5, 1)
        T_edges = round(np.random.rand() * 50, 1)

        k = 1

        print('Iter = ', iter + 1, ' | Time :', time.asctime(), ' | x = [', x1, ', ', x2, ', ', x3, ', ', x4, '], y = [', y1, ', ', y2, ', ', y3, ', ', y4, ']')

        # MESH GENERATION -------------------------------------------------------------------------------------------------#

        Pol = np.array([[x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4]])

        coord = np.zeros([25, 2])
        for i in range(3):
            coord[i * 4, 0] = Pol[i, 0]
            coord[i * 4, 1] = Pol[i, 1]
            coord[i * 4 + 2, 0] = (Pol[i + 1, 0] + Pol[i, 0]) / 2
            coord[i * 4 + 2, 1] = (Pol[i + 1, 1] + Pol[i, 1]) / 2
            coord[i * 4 + 1, 0] = (coord[i * 4 + 2, 0] + Pol[i, 0]) / 2
            coord[i * 4 + 1, 1] = (coord[i * 4 + 2, 1] + Pol[i, 1]) / 2
            coord[i * 4 + 3, 0] = (coord[i * 4 + 2, 0] + Pol[i + 1, 0]) / 2
            coord[i * 4 + 3, 1] = (coord[i * 4 + 2, 1] + Pol[i + 1, 1]) / 2

        coord[12, 0] = Pol[3, 0]
        coord[12, 1] = Pol[3, 1]
        coord[14, 0] = (Pol[0, 0] + Pol[3, 0]) / 2
        coord[14, 1] = (Pol[0, 1] + Pol[3, 1]) / 2
        coord[13, 0] = (coord[14, 0] + Pol[3, 0]) / 2
        coord[13, 1] = (coord[14, 1] + Pol[3, 1]) / 2
        coord[15, 0] = (coord[14, 0] + Pol[0, 0]) / 2
        coord[15, 1] = (coord[14, 1] + Pol[0, 1]) / 2

        Var1_11 = (coord[11, 1] - coord[1, 1]) / (coord[11, 0] - coord[1, 0])
        Var2_10 = (coord[10, 1] - coord[2, 1]) / (coord[10, 0] - coord[2, 0])
        Var3_9 = (coord[9, 1] - coord[3, 1]) / (coord[9, 0] - coord[3, 0])

        Var15_5 = (coord[5, 1] - coord[15, 1]) / (coord[5, 0] - coord[15, 0])
        Var14_6 = (coord[6, 1] - coord[14, 1]) / (coord[6, 0] - coord[14, 0])
        Var13_7 = (coord[7, 1] - coord[13, 1]) / (coord[7, 0] - coord[13, 0])


        coord[16, 0] = (coord[1, 1] - coord[15, 1] - Var1_11 * coord[1, 0] + Var15_5 * coord[15, 0]) / (Var15_5 - Var1_11)
        coord[16, 1] = coord[15, 1] + Var15_5 * (coord[16, 0] - coord[15, 0])
        coord[17, 0] = (coord[2, 1] - coord[15, 1] - Var2_10 * coord[2, 0] + Var15_5 * coord[15, 0]) / (Var15_5 - Var2_10)
        coord[17, 1] = coord[15, 1] + Var15_5 * (coord[17, 0] - coord[15, 0])
        coord[18, 0] = (coord[3, 1] - coord[15, 1] - Var3_9 * coord[3, 0] + Var15_5 * coord[15, 0]) / (Var15_5 - Var3_9)
        coord[18, 1] = coord[15, 1] + Var15_5 * (coord[18, 0] - coord[15, 0])

        coord[19, 0] = (coord[1, 1] - coord[14, 1] - Var1_11 * coord[1, 0] + Var14_6 * coord[14, 0]) / (Var14_6 - Var1_11)
        coord[19, 1] = coord[14, 1] + Var14_6 * (coord[19, 0] - coord[14, 0])
        coord[20, 0] = (coord[2, 1] - coord[14, 1] - Var2_10 * coord[2, 0] + Var14_6 * coord[14, 0]) / (Var14_6 - Var2_10)
        coord[20, 1] = coord[14, 1] + Var14_6 * (coord[20, 0] - coord[14, 0])
        coord[21, 0] = (coord[3, 1] - coord[14, 1] - Var3_9 * coord[3, 0] + Var14_6 * coord[14, 0]) / (Var14_6 - Var3_9)
        coord[21, 1] = coord[14, 1] + Var14_6 * (coord[21, 0] - coord[14, 0])

        coord[22, 0] = (coord[1, 1] - coord[13, 1] - Var1_11 * coord[1, 0] + Var13_7 * coord[13, 0]) / (Var13_7 - Var1_11)
        coord[22, 1] = coord[13, 1] + Var13_7 * (coord[22, 0] - coord[13, 0])
        coord[23, 0] = (coord[2, 1] - coord[13, 1] - Var2_10 * coord[2, 0] + Var13_7 * coord[13, 0]) / (Var13_7 - Var2_10)
        coord[23, 1] = coord[13, 1] + Var13_7 * (coord[23, 0] - coord[13, 0])
        coord[24, 0] = (coord[3, 1] - coord[13, 1] - Var3_9 * coord[3, 0] + Var13_7 * coord[13, 0]) / (Var13_7 - Var3_9)
        coord[24, 1] = coord[13, 1] + Var13_7 * (coord[24, 0] - coord[13, 0])

        tri = scipy.spatial.Delaunay(coord).simplices + 1
        coord = np.concatenate((np.arange(1,26).reshape((25, 1)), coord), 1)
        n_el = 32
        n_nodes = 25
        n_nodes_edges = 16

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

        Ur = U[16:25]
        Ur = Uc.reshape((1, 9))
        toc = time.time() - tic

        Data_heat_FE0 = np.load('Data_heat_FE_Quadrilateral_Full_Updated9points.npy').item()
        X0 = Data_heat_FE0['X']
        y0 = Data_heat_FE0['y']
        comp_time0 = Data_heat_FE0['comp_time']

        X = np.array([[x1, y1, x2, y2, x3, y3, x4, y4, Q, T_edges]])
        y = Ur
        comp_time = np.array([[toc]])

        X = np.concatenate((X0, X), 0)
        y = np.concatenate((y0, y), 0)
        comp_time = np.concatenate((comp_time0, comp_time), 0)

        Data_heat_FE = {'X': X, 'y': y, 'comp_time': comp_time}
        np.save('Data_heat_FE_Quadrilateral_Full_Updated9points.npy', Data_heat_FE)

    except:
        continue

