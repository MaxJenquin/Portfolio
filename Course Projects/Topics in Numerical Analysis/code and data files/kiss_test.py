import torch
import gpytorch
import numpy as np
import time
from io import StringIO
from matplotlib import pyplot as plt


class KISS_ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(KISS_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        gridsize = gpytorch.utils.grid.choose_grid_size(train_x, 1.0)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(kernel, grid_size=gridsize, num_dims=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


csv_path = 'data2_grid.csv'

# input data, put it in a 1-dimensional torch tensor
with open(csv_path, 'r') as file:
    inputs = file.read()
data = np.genfromtxt(StringIO(inputs), delimiter=',', usecols=(0, 1), dtype=(float, float), skip_header=0,
                     autostrip=True, names=["t", "y"])
times = torch.tensor([x[0] for x in data]).reshape(-1, 1).type(torch.FloatTensor)[:, 0]
vals = torch.tensor([x[1] for x in data]).reshape(-1, 1).type(torch.FloatTensor)[:, 0]

kernelRBF = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

xdata = times[:1000]
ydata = vals[:1000]

evalpoints = torch.linspace(xdata[0].item(), (1.1*xdata[-1]-0.1*xdata[0]).item(), 1000)
# initialize model, optimizer, and marginal log likelihood for training
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = KISS_ExactGPModel(xdata, ydata, likelihood, kernelRBF)
print(model.covar_module.grid_size)
likelihood.train()
model.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    ], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# train
start = time.time()
for j in range(100):
    optimizer.zero_grad()
    output = model(xdata)
    loss = -mll(output, ydata)
    loss.backward()
    optimizer.step()

model.eval()
likelihood.eval()
preds = likelihood(model(evalpoints))
# TODO: end timer here instead? see if it makes a difference at least.
mean = preds.mean
lower, upper = preds.confidence_region()
end = time.time()

print(model.covar_module.base_kernel.base_kernel.lengthscale.item())
print(model.likelihood.noise.item())

print(end-start)

f, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(evalpoints.numpy(), mean.detach().numpy(), 'b')
ax.plot(xdata.numpy(), ydata.numpy(), 'k.')
ax.fill_between(evalpoints.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
plt.show()

