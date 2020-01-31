import numpy as np


# list csv names
datanum = '1'
file_list = ['data'+datanum+'_grid.csv']
for j in range(5):
    file_list.append('data'+datanum+'_exptrial_'+str(j+1)+'.csv')

n_points, times_grid, rel_errors_grid = np.genfromtxt(file_list[0], delimiter=',', usecols=(1, 2, 3), unpack=True, dtype=None)
rel_errors = np.empty((n_points.size(), 5))
times = np.empty((n_points.size(), 5))
foo = np.empty((n_points.size(), 5))
for file in file_list[1:]:
    foo[:, j], times[:, j], rel_errors[:, j] = np.genfromtxt(file, delimiter=',', usecols=(1, 2, 3), unpack=True, dtype=None)

times_exp = np.mean(times, axis=1)
rel_errors_exp = np.mean(rel_errors, axis=1)

print(rel_errors_exp.shape)
print(times_exp.shape)
print(times_grid.shape)
print(rel_errors_grid.shape)
