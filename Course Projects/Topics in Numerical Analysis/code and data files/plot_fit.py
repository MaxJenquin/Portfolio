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


n = 7500

kernelRBF = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
kernelPER = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

expdist = torch.distributions.exponential.Exponential(torch.tensor([1.0]))
normdist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([3.0]))

xdata = expdist.sample(sample_shape=torch.Size([n]))
xdata[0] = 2
for j in range(1, n):
    xdata[j] += xdata[j-1]

ydata_1 = data_function_1(xdata)+normdist.sample(sample_shape=torch.Size([n]))
ydata_2 = data_function_2(xdata)+normdist.sample(sample_shape=torch.Size([n]))

xdata = xdata[:, 0]
ydata_1 = ydata_1[:, 0]
ydata_2 = ydata_2[:, 0]

evalpoints = torch.linspace(xdata[0].item(), xdata[-1].item(), 1000)

# model ydata_1
model_1, likelihood_1, start_1 = train_bfgs(xdata, ydata_1, kernelRBF*kernelPER)
model_1.eval()
likelihood_1.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds_1 = likelihood_1(model_1(evalpoints))
    mean_1 = preds_1.mean
    lower_1, upper_1 = preds_1.confidence_region()
end_1 = time.time()
true_vals_1 = data_function_1(evalpoints)
rel_mean_error_1 = torch.div((true_vals_1-mean_1), true_vals_1).abs().mean().item()

# model ydata_2
model_2, likelihood_2, start_2 = train_bfgs(xdata, ydata_2, kernelRBF*kernelPER)
model_2.eval()
likelihood_2.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds_2 = likelihood_2(model_2(evalpoints))
    mean_2 = preds_2.mean
    lower_2, upper_2 = preds_2.confidence_region()
end_2 = time.time()
true_vals_2 = data_function_2(evalpoints)
rel_mean_error_2 = torch.div((true_vals_2-mean_2), true_vals_2).abs().mean().item()

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(211)
ax1.plot(xdata.numpy(), ydata_1.numpy(), 'k.', label='Data')
ax1.plot(evalpoints.numpy(), mean_1.numpy(), 'b', label="Mean")
ax1.fill_between(evalpoints.numpy(), lower_1.numpy(), upper_1.numpy(), alpha=0.5, label="Confidence")
ax1.set_ylabel('$g_1$', rotation=0)
plt.legend(bbox_to_anchor=(0., -0.19, 1., 1.), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_xticks([])
plt.suptitle('Example Fits of KISS-GP Model with L-BFGS')

ax2 = plt.subplot(212)
l1 = ax2.plot(xdata.numpy(), ydata_2.numpy(), 'k.')
l2 = ax2.plot(evalpoints.numpy(), mean_2.numpy(), 'b')
l3 = ax2.fill_between(evalpoints.numpy(), lower_2.numpy(), upper_2.numpy(), alpha=0.5)
ax2.set_ylabel('$g_2$', rotation=0)
#fig.legend((l1, l2, l3), ('Data', 'Mean', 'Confidence'))

print('Data Set 1: modeling %i points took %f seconds' % (n, end_1-start_1))
print('relative mean prediction error: %f\n' % rel_mean_error_1)

print('Data Set 2: modeling %i points took %f seconds' % (n, end_2-start_2))
print('relative mean prediction error: %f\n' % rel_mean_error_2)

plt.xlabel('Time')
plt.tight_layout
plt.show()

