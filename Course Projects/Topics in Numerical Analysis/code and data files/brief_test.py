import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# here we just find the length scale of sin(2pix) on [0,1]

# why not a linspace? good question - no particular reason except we ripped this sampling scheme from the project
expdist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
n_points = 100
xdata = expdist.sample(sample_shape=torch.Size([n_points]))
for j in range(1, n_points):
    xdata[j] += xdata[j-1]
xdata = xdata - xdata[0]
xdata = xdata/xdata[-1]

ydata = torch.sin(2*np.pi*xdata)

xdata = xdata[:, 0]
ydata = ydata[:, 0]

kernelRBF = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(xdata, ydata, likelihood, kernelRBF)

likelihood.train()
model.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for j in range(1000):
    optimizer.zero_grad()
    output = model(xdata)
    loss = -mll(output, ydata)
    loss.backward()
    optimizer.step()
    print(model.covar_module.base_kernel.lengthscale.item())

model.eval()
likelihood.eval()
print(model.covar_module.base_kernel.lengthscale.item())

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(xdata.numpy(), ydata.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

plt.show()
