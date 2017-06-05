import sys

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []
Z = []
Tri = []
with open(sys.argv[1]) as f:
    lines = f.read().splitlines()
    idx = lines.index('end_header') + 1
    while len(lines[idx].split()) == 6 or len(lines[idx].split()) == 3:
        line = lines[idx].split()
        X.append(float(line[0]))
        Y.append(float(line[1]))
        Z.append(float(line[2]))
        idx += 1
    while idx < len(lines):
        _,a,b,c = map(int,lines[idx].split())
        Tri.append((a,b,c))
        idx += 1

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(X,Y,Z,triangles=Tri)

plt.axis('off')
plt.show()
#fig.savefig('test.pdf')
