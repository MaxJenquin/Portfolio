import torch
from matplotlib import pyplot as plt
import numpy as np


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


# generate grid-spaced data for each of the three functions

grid_times = torch.linspace(2, 5001, steps=5000).reshape(-1, 1)
grid_vals_1 = data_function_1(grid_times)
grid_vals_2 = data_function_2(grid_times)
grid_vals_3 = data_function_3(grid_times)
grid_data_1 = torch.cat((grid_times, grid_vals_1), dim=1)
grid_data_2 = torch.cat((grid_times, grid_vals_2), dim=1)
grid_data_3 = torch.cat((grid_times, grid_vals_3), dim=1)

savefile = open('data1_grid.csv', 'w')
np.savetxt(savefile, grid_data_1.numpy(), delimiter=',', fmt='%f')
savefile.close()
savefile = open('data2_grid.csv', 'w')
np.savetxt(savefile, grid_data_2.numpy(), delimiter=',', fmt='%f')
savefile.close()
savefile = open('data3_grid.csv', 'w')
np.savetxt(savefile, grid_data_3.numpy(), delimiter=',', fmt='%f')
savefile.close()


# generate 5 trials of exponential inter-arrival time data for each of the three functions

m = torch.distributions.exponential.Exponential(torch.tensor([1.0]))

num_times = 5000
t0 = 2

for i in range(5):
    times = m.sample(sample_shape=torch.Size([num_times]))
    times[0] = t0
    for j in range(1, num_times):
        times[j] += times[j-1]

    vals_1 = data_function_1(times)
    vals_2 = data_function_2(times)
    vals_3 = data_function_3(times)
    data_1 = torch.cat((times, vals_1), dim=1)
    data_2 = torch.cat((times, vals_2), dim=1)
    data_3 = torch.cat((times, vals_3), dim=1)

    savefile = open('data1_exptrial_'+str(i+1)+'.csv', 'w')
    np.savetxt(savefile, data_1.numpy(), delimiter=',', fmt='%f')
    savefile.close()
    savefile = open('data2_exptrial_' + str(i + 1) + '.csv', 'w')
    np.savetxt(savefile, data_2.numpy(), delimiter=',', fmt='%f')
    savefile.close()
    savefile = open('data3_exptrial_' + str(i + 1) + '.csv', 'w')
    np.savetxt(savefile, data_3.numpy(), delimiter=',', fmt='%f')
    savefile.close()


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(grid_times.numpy(), grid_vals_1.numpy())
plt.subplot(3, 1, 2)
plt.plot(grid_times.numpy(), grid_vals_2.numpy())
plt.subplot(3, 1, 3)
plt.plot(grid_times.numpy(), grid_vals_3.numpy())
plt.show()
