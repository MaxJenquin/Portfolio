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


print('started: ' + str(time.ctime(int(time.time()))))
for master in range(2):
    # preliminaries
    kernelRBF = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernelPER = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
    num_trials = 30
    if master == 0:
        data_function = data_function_1
    elif master == 1:
        data_function = data_function_2

    expdist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
    normdist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([3.0]))
    outputs = np.zeros(((30000-1000-1) // 1000 + 1, 6))
    for n in range(1000, 30000, 1000):
        index = (n-1000) // 1000
        rel_mean_errors = np.empty(num_trials)
        times = np.empty(num_trials)
        for i in range(num_trials):
            # generate random inter-arrival times for data points, then data with noise
            xdata = expdist.sample(sample_shape=torch.Size([n]))
            xdata[0] = 2
            for j in range(1, n):
                xdata[j] += xdata[j-1]
            ydata = data_function(xdata) + normdist.sample(sample_shape=torch.Size([n]))

            # flatten xdata, ydata into 1d arrays
            xdata = xdata[:, 0]
            ydata = ydata[:, 0]

            # generate test points
            evalpoints = torch.linspace(xdata[0].item(), xdata[-1].item(), 500)

            # initialize and train, timer built in
            model, likelihood, start = train_bfgs(xdata, ydata, kernelRBF*kernelPER)
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():  # activates fast predictions via Lanczos estimates (LOVE)
                likelihood.eval()
                preds = likelihood(model(evalpoints))
                mean = preds.mean
                lower, upper = preds.confidence_region()
            end = time.time()

            # check error
            true_vals = data_function(evalpoints)
            rel_mean_errors[i] = torch.div((true_vals-mean), true_vals).abs().mean().item()
            times[i] = end-start
            print('(%f, %f)' % (times[i], rel_mean_errors[i]))

        print('On average, %i points modeled in %f seconds with relative mean error %f\n' % (n, np.mean(times), np.mean(rel_mean_errors)))
        outputs[index, 0] = n                                                   # number of data points
        outputs[index, 1] = np.mean(times)                                      # mean modeling time
        outputs[index, 2] = np.mean(rel_mean_errors)                            # mean error over test points
        outputs[index, 3] = np.max(times)                                       # longest modeling time
        outputs[index, 4] = np.min(times)                                       # shortest modeling time
        outputs[index, 5] = np.size(rel_mean_errors[rel_mean_errors > 1])       # number of sub-par fits out of 30


    savefile = open('data'+str(master+1)+'_window_size_test.csv', 'w')
    np.savetxt(savefile, outputs, delimiter=',', fmt='%f')
    savefile.close()

print('ended: ' + str(time.ctime(int(time.time()))))
