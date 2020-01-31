from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import gpytorch
import caffeine
from gpytorch.kernels import Kernel
import time
import numpy as np

gpytorch.settings.debug._state = False
caffeine.on(display = True)

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


D = np.load('/Users/christophebonneville/Google Drive/Shared Python Project/HEAT DATABASE/Data_heat_FE.npy').item()

ntrain = 500
ntest = 500

x = D['X']
n = x.shape[0]
xtrain = np.copy(x[0:ntrain, :])
mean_train_x = np.mean(xtrain, axis = 0)
std_train_x = np.std(xtrain, axis = 0)
xtrain = (xtrain - mean_train_x) / std_train_x
xtrain = torch.tensor(xtrain).type(torch.FloatTensor)
xstar = np.copy(x[n - ntest:n, :])
xstar = (xstar - mean_train_x) / std_train_x
xstar = torch.tensor(xstar).type(torch.FloatTensor)

y = D['y']
ytrain = np.copy(y[0:ntrain, :])
ytrain_mean = np.mean(ytrain)
ytrain = ytrain - ytrain_mean
ytrain = torch.tensor(ytrain).type(torch.FloatTensor)
ytrain = ytrain[:, 0]
ystar = np.copy(y[n - ntest:n, :])

likelihood = gpytorch.likelihoods.GaussianLikelihood()
kernelOU = OUKernel(ard_num_dims = 4)
kernel = gpytorch.kernels.ScaleKernel(kernelOU * kernelOU)
model = ExactGPModel(xtrain, ytrain, likelihood, kernel)

model.train()
likelihood.train()

optimizer = torch.optim.Adam([{'params': model.parameters()}], lr = 0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

tic = time.time()
training_iter = 500
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(xtrain)
    loss = -mll(output, ytrain)
    loss.backward()

    print(
        'Iter %d/%d - Loss: %.3f  log_noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.likelihood.log_noise.item()
        ))
    optimizer.step()

toc = time.time() - tic
print(toc)

model.eval()
likelihood.eval()

f_pred = model(xtrain)
y_pred = likelihood(model(xtrain))
y_pred_mean = y_pred.mean
pred_train = y_pred_mean.detach().numpy() + ytrain_mean
pred_train = pred_train.reshape((ntrain, 1))

f_pred = model(xstar)
y_pred = likelihood(model(xstar))
y_pred_mean = y_pred.mean
pred_star = y_pred_mean.detach().numpy() + ytrain_mean
pred_star = pred_star.reshape((ntest, 1))

error_train = np.abs((y[0:ntrain, :] - pred_train) / y[0:ntrain, :]) * 100
error_star = np.abs((ystar - pred_star) / ystar) * 100

#lower_star, upper_star = y_pred.confidence_region()
#lower_star = lower_star.detach().numpy() + ytrain_mean
#upper_star = upper_star.detach().numpy() + ytrain_mean