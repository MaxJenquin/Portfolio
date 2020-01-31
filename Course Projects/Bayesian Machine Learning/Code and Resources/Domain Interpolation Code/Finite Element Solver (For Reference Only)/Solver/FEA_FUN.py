import numpy as np
import copy

def jacfqe(i, coord, tri, B_loc, DTnodes):
    nodes = tri[i, :]
    nodes = nodes.astype(int)
    nodes.sort()

    Case = 0
    coord1 = coord[nodes[0]-1, 1:3]
    coord2 = coord[nodes[1]-1, 1:3]
    coord3 = coord[nodes[2]-1, 1:3]

    if np.shape(DTnodes)[0] > 0:
        for j in range(0, np.shape(DTnodes)[0]):

            if DTnodes[j][0] <= nodes[0] < nodes[1] < nodes[2] <= DTnodes[j][-1]:
                coord1 = coord[nodes[1] - 1, 1:3] # nodes[1] becomes N1
                coord2 = coord[nodes[2] - 1, 1:3] # nodes[2] becomes N2
                coord3 = coord[nodes[0] - 1, 1:3] # nodes[0] becomes N3
                Case = 1
            elif (DTnodes[j][0] <= nodes[0] < nodes[1] < DTnodes[j][-1] or DTnodes[j][0] < nodes[0] < nodes[1] <= DTnodes[j][-1]) \
                    and nodes[0] + 1 == nodes[1]:
                coord1 = coord[nodes[1] - 1, 1:3]
                coord2 = coord[nodes[2] - 1, 1:3]
                coord3 = coord[nodes[0] - 1, 1:3]
                Case = 2
            elif DTnodes[j][0] == nodes[0] and DTnodes[j][-1] == nodes[1]:
                coord1 = coord[nodes[1] - 1, 1:3]
                coord2 = coord[nodes[2] - 1, 1:3]
                coord3 = coord[nodes[0] - 1, 1:3]
                Case = 2
            elif (DTnodes[j][0] <= nodes[1] < nodes[2] < DTnodes[j][-1] or DTnodes[j][0] < nodes[1] < nodes[2] <= DTnodes[j][-1]) \
                    and nodes[1] + 1 == nodes[2]:
                coord1 = coord[nodes[2] - 1, 1:3] # nodes[2] becomes N1
                coord2 = coord[nodes[0] - 1, 1:3] # nodes[0] becomes N2
                coord3 = coord[nodes[1] - 1, 1:3] # nodes[1] becomes N3
                Case = 3
            elif DTnodes[j][0] == nodes[1] and DTnodes[j][-1] == nodes[2]:
                coord1 = coord[nodes[2] - 1, 1:3]
                coord2 = coord[nodes[0] - 1, 1:3]
                coord3 = coord[nodes[1] - 1, 1:3]
                Case = 3
            elif (DTnodes[j][0] < nodes[0] < nodes[2] <= DTnodes[j][-1] or DTnodes[j][0] <= nodes[0] < nodes[2] < DTnodes[j][-1])\
                    and nodes[0] + 1 == nodes[2]:
                coord1 = coord[nodes[0] - 1, 1:3] # nodes[0] becomes N1
                coord2 = coord[nodes[1] - 1, 1:3] # nodes[1] becomes N2
                coord3 = coord[nodes[2] - 1, 1:3] # nodes[2] becomes N3
                Case = 4
            elif DTnodes[j][0] == nodes[0] and DTnodes[j][-1] == nodes[2]:
                coord1 = coord[nodes[0] - 1, 1:3]
                coord2 = coord[nodes[1] - 1, 1:3]
                coord3 = coord[nodes[2] - 1, 1:3]
                Case = 4

    dXdxi = B_loc[0, 0] * coord1[0] + B_loc[0, 1] * coord2[0] + B_loc[0,2] * coord3[0]
    dYdxi = B_loc[0, 0] * coord1[1] + B_loc[0, 1] * coord2[1] + B_loc[0, 2] * coord3[1]
    dXdeta = B_loc[1, 0] * coord1[0] + B_loc[1, 1] * coord2[0] + B_loc[1, 2] * coord3[0]
    dYdeta = B_loc[1, 0] * coord1[1] + B_loc[1, 1] * coord2[1] + B_loc[1, 2] * coord3[1]

    jx = (dXdxi ** 2 + dYdxi ** 2) ** (1/2)
    jy = (dXdeta ** 2 + dYdeta ** 2) ** (1/2)

    return nodes, jx, jy, Case


def fqecalc(N1, N2, N3, jx, jy, ImpDT, nodes, Case):

    if Case == 1:
        for i in range(0, np.shape(ImpDT)[0] - 1):
                if ImpDT[i, 0] == nodes[0] and ImpDT[i, 1] == nodes[1] and ImpDT[i + 1, 1] == nodes[2]:
                    q1 = ImpDT[i, 2]
                    q2 = ImpDT[i + 1, 2]

        N = np.array([[N3, N1, N2]])
        fqe_integrand = []
        fqe_integrand1 = np.transpose(N) * q1 * jy
        fqe_integrand2 = np.transpose(N) * q2 * jx

    elif Case > 1:
        for i in range(0, np.shape(ImpDT)[0]):
            if ImpDT[i, 0] == nodes[0] and ImpDT[i, 1] == nodes[1]:
                q = ImpDT[i, 2]
            elif ImpDT[i, 0] == nodes[1] and ImpDT[i, 1] == nodes[2]:
                q = ImpDT[i, 2]
            elif ImpDT[i, 0] == nodes[2] and ImpDT[i, 1] == nodes[0]:
                q = ImpDT[i, 2]

        if Case == 2:
            N = np.array([[N3, N1, N2]])
        elif Case == 3:
            N = np.array([[N2, N3, N1]])
        elif Case == 4:
            N = np.array([[N1, N2, N3]])

        fqe_integrand = np.transpose(N) * q * jy
        fqe_integrand1 = []
        fqe_integrand2 = []

    return fqe_integrand, fqe_integrand1, fqe_integrand2


def FEAmatrix(n_nodes, ImpDT, n_el, coord, tri, B_loc, DTnodes, k, Q, N, N1, N2, N3, xi, eta):

    K = np.zeros([n_nodes, n_nodes])
    f = np.zeros([n_nodes, 1])

    if np.shape(ImpDT)[0] > 0:
        Heat = 1
    else:
        Heat = 0


    for i in range(0, n_el):
        print(i)

        J, invJ, B, coord1, coord2, coord3, nodes = jacTH(i, coord, tri, B_loc)
        if Heat == 1:
            nodesfqe, jx, jy, Case = jacfqe(i, coord, tri, B_loc, DTnodes)
        else:
            Case = 0
            nodesfqe = nodes

        detJ = np.linalg.det(J)

        Ke_integrand = k * np.dot(np.transpose(B), B) * detJ
        fQe_integrand = Q * np.transpose(N) * detJ

        fqe = np.zeros([3, 1])
        if Case > 1:
            fqe_integrand, fqe_integrand1, fqe_integrand2 = fqecalc(N1, N2, N3, jx, jy, ImpDT, nodesfqe, Case)
            fqe = np.zeros([3, 1])
            for j in range(0, 3):
                fqe[j, 0] = fqe_integrand[j, 0].subs({xi: 0, eta: 0.5})

        elif Case == 1:
            fqe_integrand, fqe_integrand1, fqe_integrand2 = fqecalc(N1, N2, N3, jx, jy, ImpDT, nodesfqe, Case)
            fqe = np.zeros([3, 1])
            for j in range(0, 3):
                fqe[j, 0] = fqe_integrand1[j, 0].subs({xi: 0, eta: 0.5}) + fqe_integrand2[j, 0].subs({xi: 0.5, eta: 0})

        for ii in range(0,3):
            f[nodes[ii] - 1, 0] = f[nodes[ii] - 1, 0] + (fQe_integrand[ii, 0].subs({xi : 0.5, eta : 0}) +
                                                         fQe_integrand[ii, 0].subs({xi : 0, eta : 0.5}) +
                                                         fQe_integrand[ii, 0].subs({xi : 0.5, eta : 0.5})) * 1 / 6
            f[nodesfqe[ii] - 1, 0] = fqe[ii, 0] + f[nodesfqe[ii] - 1, 0]

            for jj in range(0,3):
                    K[nodes[ii] - 1, nodes[jj] - 1] = K[nodes[ii] - 1, nodes[jj] - 1] + Ke_integrand[ii, jj] * 1 / 2

    return K, f

def jacTH(i, coord, tri, B_loc):
    nodes = tri[i, :]
    nodes = nodes.astype(int)

    coord1 = coord[nodes[0]-1, 1:3]
    coord2 = coord[nodes[1]-1, 1:3]
    coord3 = coord[nodes[2]-1, 1:3]

    dXdxi = B_loc[0, 0] * coord1[0] + B_loc[0, 1] * coord2[0] + B_loc[0, 2] * coord3[0]
    dYdxi = B_loc[0, 0] * coord1[1] + B_loc[0, 1] * coord2[1] + B_loc[0, 2] * coord3[1]
    dXdeta = B_loc[1, 0] * coord1[0] + B_loc[1, 1] * coord2[0] + B_loc[1, 2] * coord3[0]
    dYdeta = B_loc[1, 0] * coord1[1] + B_loc[1, 1] * coord2[1] + B_loc[1, 2] * coord3[1]

    J = np.transpose(np.array([[dXdxi, dXdeta], [dYdxi, dYdeta]]))
    invJ = np.linalg.inv(J)
    B = np.dot(invJ, B_loc)

    return J, invJ, B, coord1, coord2, coord3, nodes


def solverTH(n_nodes, ImpT, K, f):
    U = np.zeros([n_nodes, 1])

    u_d_index = copy.deepcopy(ImpT[:,0] - 1)
    u_d_index = np.array([np.int(u_d_index[i]) for i in range(0, len(u_d_index))])
    u_free = np.arange(0,n_nodes)
    u_free = np.delete(u_free, u_d_index, 0)

    Kt = copy.deepcopy(K[:, u_d_index])
    ft = np.dot(Kt, ImpT[:, 1])
    ft = ft[u_free]
    ft = np.reshape(ft, (len(ft), 1))

    Kd = copy.deepcopy(K)
    fd = copy.deepcopy(f)

    Kd = np.delete(Kd, u_d_index, 0)
    Kd = np.delete(Kd, u_d_index, 1)
    fd = np.delete(fd, u_d_index, 0)

    U[u_free] = np.dot(np.linalg.inv(Kd), (fd - ft))
    U[u_d_index, 0] = ImpT[:, 1]

    return U


def impTconvert(ImpU):

    impU = np.array([[0, 0]])
    for i in range(0, np.shape(ImpU)[0]):
            n = (ImpU[i, 1] - ImpU[i, 0] + 1).astype(int)
            impU_i = np.transpose(np.concatenate(([np.arange(ImpU[i, 0], ImpU[i, 1] + 1)], np.ones([1, n]) * ImpU[i, 2]), 0))
            impU = np.concatenate((impU, impU_i), 0)

    ImpU = np.delete(impU, 0, 0)

    return ImpU


def impDTconvert(ImpU):

    if np.shape(ImpU)[0] == 0:
        ImpU = []
        DTnodes = []

    else:
        impU = np.array([[0, 0, 0]])
        for i in range(0, np.shape(ImpU)[0]):
                n = (ImpU[i, 1] - ImpU[i, 0]).astype(int)
                impU_i = np.transpose(np.concatenate(([np.arange(ImpU[i, 0], ImpU[i, 1])], [np.arange(ImpU[i, 0] + 1, ImpU[i, 1] + 1)],
                                                      np.ones([1, n]) * ImpU[i, 2]), 0))
                impU = np.concatenate((impU, impU_i), 0)

        DTnodes = list()
        for i in range(0, np.shape(ImpU)[0]):
            DTnodes.append(np.arange(ImpU[i, 0], ImpU[i, 1] + 1))

        ImpU = np.delete(impU, 0, 0)

    return ImpU, DTnodes








