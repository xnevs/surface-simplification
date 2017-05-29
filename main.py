import sys
from collections import defaultdict
import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = []
triangles = []
with open(sys.argv[1]) as f:
    line = next(f).split()
    while line[0] != 'element':
        line = next(f).split()
    num_points = int(line[2])
    line = next(f).split()
    while line[0] != 'element':
        line = next(f).split()
    num_triangles = int(line[2])
    line = next(f).split()
    while line[0] != 'end_header':
        line = next(f).split()
    for i in range(num_points):
        line = map(float,next(f).split())
        x = next(line)
        y = next(line)
        z = next(line)
        points.append( np.array((x,y,z)) )
    for i in range(num_triangles):
        _,a,b,c = map(int,next(f).split())
        triangles.append((a,b,c))


graph = [set() for _ in points]
for (a,b,c) in triangles:
    graph[a].add(b)
    graph[a].add(c)
    graph[b].add(a)
    graph[b].add(c)
    graph[c].add(a)
    graph[c].add(b)

def plane(a,b,c):
    n = np.cross(b-a,c-a)
    n = n / np.linalg.norm(n)
    d = -n.dot(a)
    return np.append(n,d)

def triangle_quadric(a,b,c):
    u = plane(a,b,c)
    return np.outer(u,u)

def point_quadric(points,graph,i):
    Q = np.zeros((4,4))
    for j,k in itertools.combinations(graph[i],2):
        if k in graph[j]:
            Q += triangle_quadric(points[i],points[j],points[k])
    return Q

Qs = [point_quadric(points,graph,i) for i in range(len(points))]

def edge_point(points,Qs,i,j):
    Q = Qs[i] + Qs[j]
    try:
        c = np.linalg.solve(Q[:-1,:-1],-Q[:-1,-1])
    except np.linalg.LinAlgError:  # TODO
        c = (points[i] + points[j]) / 2
    c = np.append(c,1)
    error = np.dot(np.dot(c,Q),c)
    return (-error,(i,j),c[:-1])

pq = [edge_point(points,Qs,i,j) for i in range(len(points)) for j in graph[i] if i<j]
pq.sort()


def is_safe(graph,a,b):
    return len(graph[a] & graph[b]) == 2

def contract(points,graph,Qs,pq,i,j,c):
    graph[i].remove(j)
    graph[i].update(graph[j] - {i})
    for k in graph[j]:
        if k != i:
            graph[k].remove(j)
            graph[k].add(i)
    graph[j] = None

    points[i] = c
    Qs[i] = point_quadric(points,graph,i)
    for k in graph[i]:
        Qs[k] = point_quadric(points,graph,k)

    new_pq = []
    for error,e,cc in pq:
        if not (i in e or j in e):
            new_pq.append((error,e,cc))
    for k in graph[i]:
        new_pq.append(edge_point(points,Qs,i,k))
    new_pq.sort()
    return new_pq

count = int(sys.argv[2]);
while pq and count > 0:
    _,(i,j),c = pq.pop()
    if is_safe(graph,i,j):
        pq = contract(points,graph,Qs,pq,i,j,c)
        count -= 1

Tri = [(i,j,k) for i in range(len(graph)) if graph[i] for j in graph[i] if graph[j] and i<j for k in (graph[i] & graph[j]) if graph[k] and j<k]

P = np.array(points)
for i in range(len(graph)):
    if graph[i] is None:
        P[i,:] = np.array([0,0,0])

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(P[:,0],P[:,1],P[:,2],triangles=Tri)
plt.show()
