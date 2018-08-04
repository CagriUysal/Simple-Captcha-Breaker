import numpy as np
with open('shapes') as f:
	shapes = [[int(j) for j in i.split('x')] for i in f.read().splitlines()]
print np.array(shapes).mean(axis=0)
