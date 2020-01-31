import torch
import gpytorch


#temporary testing ground
# sample function 3 - simple periodic function with significant added noise
def data_function_3(inputs):
    analytic = 70.0*(inputs/100.0).sin()
    print(analytic.shape)
    print(inputs.shape)
    m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([5.0]))
    noise = m.sample(inputs.shape).reshape(-1, 1)
    print(noise.shape)
    return analytic+noise


print((1000-100-1)//(100+1))
print((1000-100-1)//100+1)



test = torch.linspace(0, 100, 1000).reshape(-1, 1)
data = data_function_3(test)

print((data-test).abs().mean().item())

print(test.shape)
print(data.shape)
