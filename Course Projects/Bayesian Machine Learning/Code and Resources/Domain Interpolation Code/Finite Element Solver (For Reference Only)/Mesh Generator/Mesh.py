import numpy as np
import scipy.spatial as sp
import shapely.geometry as gm
import copy
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import scipy


class Mesh:

    def __init__(self, Pol, e, minAng = 33.8):

        self.Pol = Pol # Pol: Matrix of nodes coordinates defining the geometry polygon
        self.e = e # Element width along the polygon edges
        self.minAng = minAng * np.pi / 180 # Minimum acceptable angle within the mesh (default : 33.8 degrees)


    # MESH GENERATION METHOD ----------------------------------------------------------------------------------------- #
    def generate(self, ani = 0): # ani = 1 : real time animation

        Mesh.__polcontour(self)
        Mesh.meshtri(self)
        PoorQ = Mesh.poorqual(self)

        iter = 1
        #print(iter)
        new_coord = []

        if ani == 1:
            fig = plt.figure()
            plt.triplot(self.coord[:, 0], self.coord[:, 1], (self.Tri - 1).astype(int), color = 'k', linewidth = 0.4)
            plt.axis('equal')
            plt.show(scipy.block == False)

        while np.shape(PoorQ)[0] != 0:
            k = np.argmin(PoorQ[:, 3])
            tri_iter = PoorQ[k, 0:3]
            coord = self.coord
            tri_coord = coord[(tri_iter - 1).astype(int), :]
            CC = Mesh.__circumcenter(tri_coord)

            if gm.Point(CC[0].tolist()).intersects(gm.Polygon(self.Pol)) == True:
                coord = np.concatenate((coord, CC), 0)
                self.coord = coord

            Mesh.meshtri(self)
            PoorQ = Mesh.poorqual(self)
            coord = self.coord

            if np.shape(new_coord)[0] == np.shape(coord)[0]:
                print('Minimum Angle Unreachable')
                break

            new_coord = copy.deepcopy(coord)

            if ani == 1:
                plt.clf()
                plt.triplot(self.coord[:, 0], self.coord[:, 1], (self.Tri - 1).astype(int), color = 'k', linewidth = 0.4)
                plt.axis('equal')
                fig.canvas.draw()
                plt.pause(0.0001)

            iter = iter + 1
            #print(iter)

        n_nodes = np.shape(coord)[0]
        coord = np.concatenate((np.transpose([np.arange(1, n_nodes + 1)]), coord), 1)
        self.n_nodes = n_nodes
        self.coord = coord


    # COMPUTING THE COORDINATES OF THE NODES LOCATED ON THE EDGES ---------------------------------------------------- #
    def __polcontour(self):

        Pol = self.Pol
        e = self.e

        n_edges = len(Pol)
        Pol = np.concatenate((Pol, np.zeros([n_edges, 1])), 1)

        for i in range(0, n_edges):
            if i == n_edges - 1:
                Pol[i, 2] = np.linalg.norm([Pol[n_edges - 1, 0] - Pol[0, 0], Pol[n_edges - 1, 1] - Pol[0, 1]])
            else:
                Pol[i, 2] = np.linalg.norm([Pol[i + 1, 0] - Pol[i, 0], Pol[i + 1, 1] - Pol[i, 1]])

        xPol = [[Pol[0, 0]]]
        yPol = [[Pol[0, 1]]]

        if np.shape(e) == ():
            for i in range(0, n_edges):
                edges_n_nodes = round(Pol[i, 2] / e)

                if i == n_edges - 1:
                    x_edge = np.linspace(Pol[n_edges - 1, 0], Pol[0, 0], edges_n_nodes+1)
                    x_edge = np.delete(x_edge, 0, 0)

                    if Pol[0, 0] == Pol[n_edges - 1, 0]:
                        y_edge = np.linspace(Pol[n_edges - 1, 1], Pol[0, 1], edges_n_nodes + 1)
                        y_edge = np.delete(y_edge, 0, 0)
                    else:
                        var = (Pol[0, 1] - Pol[n_edges - 1, 1]) / (Pol[0, 0] - Pol[n_edges - 1, 0])
                        y_edge = var * (x_edge - Pol[0, 0]) + Pol[0, 1]

                else:
                    x_edge = np.linspace(Pol[i, 0], Pol[i + 1, 0], edges_n_nodes + 1)
                    x_edge = np.delete(x_edge, 0, 0)

                    if Pol[i + 1, 0] == Pol[i, 0]:
                        y_edge = np.linspace(Pol[i, 1], Pol[i + 1, 1], edges_n_nodes + 1)
                        y_edge = np.delete(y_edge, 0, 0)
                    else:
                        var = (Pol[i + 1, 1] - Pol[i, 1]) / (Pol[i + 1, 0] - Pol[i, 0])
                        y_edge = var * (x_edge - Pol[i + 1, 0]) + Pol[i + 1, 1]

                xPol = np.concatenate((xPol, [x_edge]), 1)
                yPol = np.concatenate((yPol, [y_edge]), 1)

        else:
            for i in range(0, n_edges):
                edges_n_nodes = round(Pol[i, 2] / e[i])

                if i == n_edges - 1:
                    x_edge = np.linspace(Pol[n_edges - 1, 0], Pol[0, 0], edges_n_nodes + 1)
                    x_edge = np.delete(x_edge, 0, 0)

                    if Pol[0, 0] == Pol[n_edges - 1, 0]:
                        y_edge = np.linspace(Pol[n_edges - 1, 1], Pol[0, 1], edges_n_nodes + 1)
                        y_edge = np.delete(y_edge, 0, 0)
                    else:
                        var = (Pol[0, 1] - Pol[n_edges - 1, 1]) / (Pol[0, 0] - Pol[n_edges - 1, 0])
                        y_edge = var * (x_edge - Pol[0, 0]) + Pol[0, 1]

                else:
                    x_edge = np.linspace(Pol[i, 0], Pol[i + 1, 0], edges_n_nodes + 1)
                    x_edge = np.delete(x_edge, 0, 0)

                    if Pol[i + 1, 0] == Pol[i, 0]:
                        y_edge = np.linspace(Pol[i, 1], Pol[i + 1, 1], edges_n_nodes + 1)
                        y_edge = np.delete(y_edge, 0, 0)
                    else:
                        var = (Pol[i + 1, 1] - Pol[i, 1]) / (Pol[i + 1, 0] - Pol[i, 0])
                        y_edge = var * (x_edge - Pol[i + 1, 0]) + Pol[i + 1, 1]

                xPol = np.concatenate((xPol, [x_edge]), 1)
                yPol = np.concatenate((yPol, [y_edge]), 1)


        n_nodes_edges = np.shape(xPol)[1] - 1
        coord = np.concatenate((np.transpose(xPol), np.transpose(yPol)), 1)
        coord = np.delete(coord, -1, 0)

        self.xPol = xPol
        self.yPol = yPol
        self.n_edges = n_edges
        self.n_nodes_edges = n_nodes_edges
        self.coord = coord


    # COMPUTING DELAUNAY TRIANGULATION METHOD ------------------------------------------------------------------------ #
    def meshtri(self):
        coord = self.coord
        Pol = self.Pol
        n_nodes_edges = self.n_nodes_edges

        Tri = sp.Delaunay(coord).simplices + 1
        n_el = np.shape(Tri)[0]

        for i in range(n_el - 1, -1, -1):
            v1 = coord[Tri[i, 1] - 1, :] - coord[Tri[i, 0] - 1, :]
            v2 = coord[Tri[i, 2] - 1, :] - coord[Tri[i, 0] - 1, :]

            det12 = np.linalg.det(np.concatenate((v1.reshape(2, 1), v2.reshape(2, 1)), 1))
            if det12 < 10E-5:
                Tri = np.delete(Tri, i, 0)

        n_el = np.shape(Tri)[0]

        for i in range(n_el - 1, -1, -1):
            n1 = (coord[Tri[i, 0] - 1, :] + coord[Tri[i, 1] - 1, :]) / 2
            n2 = (coord[Tri[i, 0] - 1, :] + coord[Tri[i, 2] - 1, :]) / 2
            n3 = (coord[Tri[i, 2] - 1, :] + coord[Tri[i, 1] - 1, :]) / 2

            if gm.Point(n1).intersects(gm.Polygon(Pol)) == False and \
                    gm.Point(n2).intersects(gm.Polygon(Pol)) == False:
                Tri = np.delete(Tri, i, 0)
                continue

            if gm.Point(n1).intersects(gm.Polygon(Pol)) == False and \
                    gm.Point(n3).intersects(gm.Polygon(Pol)) == False:
                Tri = np.delete(Tri, i, 0)
                continue

            if gm.Point(n3).intersects(gm.Polygon(Pol)) == False and \
                    gm.Point(n2).intersects(gm.Polygon(Pol)) == False:
                Tri = np.delete(Tri, i, 0)
                continue

            if gm.Point(n1).intersects(gm.Polygon(Pol)) == False and \
                    gm.Point(n2).intersects(gm.Polygon(Pol)) == False:
                Tri = np.delete(Tri, i, 0)
                continue

            if Tri[i, 0] <= n_nodes_edges and Tri[i, 1] <= n_nodes_edges and \
                    Tri[i, 2] <= n_nodes_edges:
                if gm.Point(n1).intersects(gm.Polygon(Pol)) == False or \
                        gm.Point(n2).intersects(gm.Polygon(Pol)) == False or \
                        gm.Point(n3).intersects(gm.Polygon(Pol)) == False:
                    Tri = np.delete(Tri, i, 0)
                    continue

        n_el = np.shape(Tri)[0]

        self.Tri = Tri
        self.n_el = n_el


    # FINDING THE POOR QUALITY TRIANGLES METHOD ---------------------------------------------------------------------- #
    def poorqual(self):
        n_el = self.n_el
        Tri = self.Tri
        coord = self.coord
        minAng = self.minAng

        PoorQ = np.zeros([1, 4])

        try:
            for i in range(0, n_el):
                v1 = np.array([[coord[Tri[i, 1] - 1, 0] - coord[Tri[i, 0] - 1, 0]],
                               [coord[Tri[i, 1] - 1, 1] - coord[Tri[i, 0] - 1, 1]]])
                v2 = np.array([[coord[Tri[i, 2] - 1, 0] - coord[Tri[i, 0] - 1, 0]],
                               [coord[Tri[i, 2] - 1, 1] - coord[Tri[i, 0] - 1, 1]]])
                v3 = np.array([[coord[Tri[i, 2] - 1, 0] - coord[Tri[i, 1] - 1, 0]],
                               [coord[Tri[i, 2] - 1, 1] - coord[Tri[i, 1] - 1, 1]]])

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                n3 = np.linalg.norm(v3)

                a1 = np.arccos(sum(v1 * v2) / n1 / n2)
                a2 = np.arccos(-sum(v1 * v3) / n1 / n3)
                a3 = np.arccos(sum(v3 * v2) / n3 / n2)

                if a1 < minAng or a2 < minAng or a3 < minAng:
                    PoorQ_i = np.concatenate(([Tri[i, :]], [min([a1, a2, a3])]), 1)
                    PoorQ = np.concatenate((PoorQ, PoorQ_i), 0)

            PoorQ = np.delete(PoorQ, 0, 0)

        except:
            PoorQ = list()

        return PoorQ


    # FINDING TRIANGLES CIRCUMCENTER METHOD -------------------------------------------------------------------------- #
    def __circumcenter(tri_coord):

        x1 = tri_coord[0, 0]
        y1 = tri_coord[0, 1]
        x2 = tri_coord[1, 0]
        y2 = tri_coord[1, 1]
        x3 = tri_coord[2, 0]
        y3 = tri_coord[2, 1]

        if y3 == y2 or y2 == y1:

            x2 = tri_coord[0, 0]
            y2 = tri_coord[0, 1]
            x3 = tri_coord[1, 0]
            y3 = tri_coord[1, 1]
            x1 = tri_coord[2, 0]
            y1 = tri_coord[2, 1]

            if y3 == y2 or y2 == y1:
                x3 = tri_coord[0, 0]
                y3 = tri_coord[0, 1]
                x1 = tri_coord[1, 0]
                y1 = tri_coord[1, 1]
                x2 = tri_coord[2, 0]
                y2 = tri_coord[2, 1]

        xc = -((x3 ** 2 - x2 ** 2 + y3 ** 2 - y2 ** 2) / 2 / (y3 - y2) - (x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2) / 2
               / (y2 - y1)) / ((x2 - x1) / (y2 - y1) - (x3 - x2) / (y3 - y2))
        yc = -(x2 - x1) * xc / (y2 - y1) + (x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2) / 2 / (y2 - y1)

        CC = np.array([[xc, yc]])

        return CC


    # PLOTING POLYGON CONTOUR ---------------------------------------------------------------------------------------- #
    def showcontour(self):

        Mesh.__polcontour(self)
        xPol = self.xPol
        yPol = self.yPol
        coord_contour = np.concatenate((np.transpose(xPol), np.transpose(yPol)), 1)
        coord_contour = np.delete(coord_contour, -1, 0)

        codes = [Path.MOVETO] + [Path.LINETO] * (self.n_nodes_edges - 2) + [Path.CLOSEPOLY]
        vertices = coord_contour

        path = Path(vertices, codes)
        pathpatch = PathPatch(path, facecolor = 'none', edgecolor = 'black')

        fig, ax = plt.subplots()
        ax.add_patch(pathpatch)
        plt.scatter(coord_contour[:, 0], coord_contour[:, 1], c = 'k', marker = '+', s = 30, linewidth = 0.5)
        ax.autoscale_view()
        plt.axis('equal')
        plt.show()


    # PLOTING FINAL MESH --------------------------------------------------------------------------------------------- #
    def showmesh(self):

        coord = self.coord
        tri = self.Tri
        n_nodes_edges = self.n_nodes_edges
        n_nodes = self.n_nodes
        n_el = self.n_el

        plt.close()
        plt.figure()
        plt.triplot(coord[:, 1], coord[:, 2], (tri - 1).astype(int), color = 'k', linewidth = 0.4)
        plt.title('Edge nodes numbers')
        plt.axis('equal')
        for i in range(0, n_nodes_edges):
            plt.annotate(str(coord[i, 0].astype(int)), coord[i, 1:3], size = 6)

        plt.figure()
        plt.triplot(coord[:, 1], coord[:, 2], (tri - 1).astype(int), color = 'k', linewidth = 0.4)
        plt.title('Nodes numbers')
        plt.axis('equal')
        for i in range(0, n_nodes):
            plt.annotate(str(coord[i, 0].astype(int)), coord[i, 1:3], size = 6)

        plt.figure()
        plt.triplot(coord[:, 1], coord[:, 2], (tri - 1).astype(int), color = 'k', linewidth = 0.4)
        plt.title('Elements numbers')
        plt.axis('equal')
        txtcoord = np.zeros([n_el, 2])
        for i in range(0, n_el):
            txtcoord[i, 0] = sum(coord[(tri[i, :] - 1).astype(int), 1]) / 3
            txtcoord[i, 1] = sum(coord[(tri[i, :] - 1).astype(int), 2]) / 3
        for i in range(0, n_el):
            plt.annotate(str(i), txtcoord[i, :], size = 4, color = 'r')

        plt.figure()
        plt.triplot(coord[:, 1], coord[:, 2], (tri - 1).astype(int), color = 'k', linewidth = 0.4)
        plt.title('Mesh')
        plt.axis('equal')


    def findvertindex(self):

        n_nodes = self.n_nodes
        Pol = self.Pol
        coord = self.coord
        n_vertices = Pol.shape[0]

        vertices_index = []

        for i in range(n_nodes):
            for j in range(n_vertices):
                if np.sum((coord[i, 1:3] == Pol[j, :]) * 1) == 2:
                    vertices_index.append(i + 1)

        self.vertices_index = vertices_index