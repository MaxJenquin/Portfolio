from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import gpytorch
from gpytorch.kernels import Kernel
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import numpy as np
gpytorch.settings.debug._state = False

# ExactGPModel Class, based on GPyTorch tutorial ----------------------------------------------------------------------#

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class OUKernel(Kernel):
    def __init__(self, ard_num_dims = None, log_lengthscale_prior = None, eps = 1e-6, active_dims = None, batch_size = 1):
        super(OUKernel, self).__init__(
            has_lengthscale = True,
            ard_num_dims = ard_num_dims,
            batch_size = batch_size,
            active_dims = active_dims,
            log_lengthscale_prior = log_lengthscale_prior,
            eps = eps,
        )

    def forward(self, x1, x2, **params):
        x1 = x1.div(self.lengthscale)
        x2 = x2.div(self.lengthscale)
        x1, x2 = self._create_input_grid(x1, x2, **params)

        diff = (x1 - x2).norm(2, dim = -1)

        return diff.div(-1).exp()

#----------------------------------------------------------------------------------------------------------------------#

Pol = np.array([[25, 25],
                [75, 30],
                [80, 60],
                [20.01, 75]])

Q = 2
T_edge = 20

x = np.array([[Pol[0, 0], Pol[0, 1], Pol[1, 0], Pol[1, 1], Pol[2, 0], Pol[2, 1], Pol[3, 0], Pol[3, 1], Q, T_edge]])

norm_const = np.load('/Users/christophebonneville/Google Drive/Shared Python Project/GP Fitting/Normalization.npy').item()
mean_train_x = norm_const['mean_train_x']
mean_train_y = norm_const['mean_train_y']
std_train_x = norm_const['std_train_x']

x = (x - mean_train_x) / std_train_x
x = torch.tensor(x).type(torch.FloatTensor)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
kernelOU = OUKernel(ard_num_dims = 10)
kernel = gpytorch.kernels.ScaleKernel(kernelOU * kernelOU)

ntrain = 500
D = np.load('/Users/christophebonneville/Google Drive/Shared Python Project/HEAT DATABASE/Data_heat_Quadrilateral_Full_Solution.npy').item()
xtrain = D['X']
xtrain = xtrain[0:ntrain, :]
ytrain = D['y']
ytrain = ytrain[0:ntrain, :]

xtrain = (xtrain - mean_train_x) / std_train_x
xtrain = torch.tensor(xtrain).type(torch.FloatTensor)

ytrain = ytrain - mean_train_y
ytrain = torch.tensor(ytrain).type(torch.FloatTensor)


tic = time.time()
U = np.zeros([25, 1])
U[0:16, 0] = T_edge
i = 16

for node in range(1, 10):
    model = ExactGPModel(xtrain, ytrain[:, node - 1], likelihood, kernel)
    model.load_state_dict(torch.load('/Users/christophebonneville/Google Drive/Shared Python Project/GP Fitting/Full_Solution_GP_Model_Ref' + str(node) + '.pt'))
    model.eval()
    likelihood.eval()

    T_node = likelihood(model(x)).mean.detach().numpy() + mean_train_y[node - 1]
    U[i, 0] = T_node
    i += 1

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

x = coord[:, 0]
y = coord[:, 1]

Tri = np.array([[22, 11, 12],
               [23, 22, 20],
               [23, 24, 10],
               [11, 23, 10],
               [23, 11, 22],
               [13, 22, 12],
               [ 5, 18,  4],
               [18,  3,  4],
               [17, 18, 20],
               [17, 16,  2],
               [ 3, 17,  2],
               [17,  3, 18],
               [24, 21,  6],
               [21,  5,  6],
               [ 5, 21, 18],
               [18, 21, 20],
               [21, 23, 20],
               [23, 21, 24],
               [24,  9, 10],
               [ 7, 24,  6],
               [ 9,  24,  8],
               [ 7,  8, 24],
               [16, 19, 14],
               [19, 13, 14],
               [13, 19, 22],
               [22, 19, 20],
               [19, 17, 20],
               [17, 19, 16],
               [15, 16, 14],
               [16,  1,  2],
               [ 1, 16,  0],
               [15,  0, 16]])

fig = plt.figure(1, figsize = (8, 6))
plt.tricontourf(tri.Triangulation(x, y, Tri), U[:, 0], 500, cmap = plt.cm.jet)
plt.colorbar()
plt.axis('equal')
toc = time.time() - tic
print(toc)
plt.show()