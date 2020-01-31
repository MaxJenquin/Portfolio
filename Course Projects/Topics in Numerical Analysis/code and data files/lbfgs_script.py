import torch
import gpytorch
import numpy as np
import time
from io import StringIO
from matplotlib import pyplot as plt
import gpytorch_LBFGS as util


# sample function 1 - complicated quasi-periodic function with single timescale of variation
def data_function_1(inputs):
    # use t0 >= 2, at t=1 there's a fast variation singularity
    term1 = torch.mul((inputs/25.0).cos(), (inputs/64).cos())
    term2 = torch.mul(torch.pow(inputs, 0.5), torch.sin(torch.mul(torch.pow(inputs, 0.5), torch.pow(2*inputs.log(), -1.0))))
    return 50*term1 + term2


# sample function 2 - quasi-periodic function with two timescales of variation
def data_function_2(inputs):
    # use t0>=0
    term1 = torch.mul(torch.pow(inputs, 0.25), (torch.pow(inputs, 1.1)/75.0).cos())
    term2 = torch.mul(torch.pow(inputs, 0.4), (inputs/4.0).sin())
    return 10*term1 + 2*term2


# sample function 3 - simple periodic function with significant added noise
def data_function_3(inputs):
    analytic = 70.0*(inputs/100.0).sin()
    m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([10.0]))
    noise = m.sample(inputs.shape).reshape(-1, 1)
    return analytic+noise


def clean_data_function_3(inputs):
    return 70.0*(inputs/100.0).sin()


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


def train_bfgs(xdata, ydata, kernel, training_iter=100):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = KISS_ExactGPModel(xdata, ydata, likelihood, kernel)
    model.train()
    likelihood.train()

    optimizer = util.FullBatchLBFGS(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def closure():
        optimizer.zero_grad()
        output = model(xdata)
        loss = -mll(output, ydata)
        return loss

    start_time = time.time()
    loss = closure()
    loss.backward()

    for i in range(training_iter):
        options = {'closure': closure, 'current loss': loss, 'max ls': 10}
        loss, _, _, _, _, _, _, fail = optimizer.step(options)
        if fail:
            print('Converged after %i iterations' % (i+1))
            break

    return model, likelihood, start_time


csv_path = 'data2_grid.csv'

# input data, put it in a 1-dimensional torch tensor
with open(csv_path, 'r') as file:
    inputs = file.read()
data = np.genfromtxt(StringIO(inputs), delimiter=',', usecols=(0, 1), dtype=(float, float), skip_header=0,
                     autostrip=True, names=["t", "y"])
times = torch.tensor([x[0] for x in data]).reshape(-1, 1).type(torch.FloatTensor)[:, 0]
vals = torch.tensor([x[1] for x in data]).reshape(-1, 1).type(torch.FloatTensor)[:, 0]


# construct simple GP model
kernelRBF = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
kernelPER = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

startn = 100
nstep = 100
outputs = np.zeros(((5000-startn-1)//nstep + 1, 3))
fig = plt.figure(figsize=(8, 10))
for n in range(startn, 5000, nstep):
    index = (n-startn)//nstep
    xdata = times[:n]
    ydata = vals[:n]
    # evaluate model on grid-spaced points that extend ten percent beyond training interval in time
    # TODO: adjust test set as training set increases? Windowed predictions?
    # evalpoints = torch.linspace(xdata[0].item(), (1.1*xdata[-1]-0.1*xdata[0]).item(), 1000)
    # alternate option - just over data region
    alt_evalpoints = torch.linspace(xdata[0].item(), xdata[-1].item(), 500)
    evalpoints = alt_evalpoints

    # initialize and train, timer built in to function
    model, likelihood, start = train_bfgs(xdata, ydata, kernelRBF*kernelPER)

    model.eval()
    likelihood.eval()
    preds = likelihood(model(evalpoints))
    # TODO: end timer here instead? see if it makes a difference at least.
    mean = preds.mean
    lower, upper = preds.confidence_region()
    end = time.time()
    # TODO: data function is here
    true_vals = data_function_2(evalpoints)
    rel_mean_error = torch.div((true_vals - mean), true_vals).abs().mean().item()
    outputs[index, 0] = n
    outputs[index, 1] = end-start
    outputs[index, 2] = rel_mean_error
    print('iteration with %i points took %f seconds' % (n, end-start))
    print('relative mean prediction error: %f\n' % rel_mean_error)

    # plot fit results every thousand data points
    if n % 1000 == 0:
        ax = plt.subplot(5, 1, n//1000)
        ax.plot(xdata.numpy(), ydata.numpy(), 'k.')
        ax.plot(evalpoints.numpy(), mean.detach().numpy(), 'b')
        ax.fill_between(evalpoints.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        ax.legend(['Data', 'Mean', 'Confidence'])

plt.show()
savefile = open('lbfgs_data2_grid.csv', 'w')
np.savetxt(savefile, outputs, delimiter=',', fmt='%f')
savefile.close()
