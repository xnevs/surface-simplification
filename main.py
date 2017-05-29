import sys
from collections import defaultdict
import itertools
from sortedcontainers import SortedDict

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class EdgePriorityQueue:
    def __init__(self):
        self.tree = SortedDict()
        self.errors = dict()

    def push(self,error,e,c):
        if e[1] < e[0]:
            e = (e[1],e[0])
        self.errors[e] = error
        self.tree[(error,e)] = c

    def pop(self):
        (error,e),c = self.tree.popitem()
        self.errors.pop(e)
        return (e,c)

    def discard(self,e):
        if e[1] < e[0]:
            e = (e[1],e[0])
        try:
            self.tree.pop((self.errors[e],e))
            self.errors.pop(e)
        except KeyError:
            pass


try:
    filename = sys.argv[1]
    count = int(sys.argv[2])
except:
    print("Sample input for file name:")
    print("examples/cow.ply")
    fileName = input("Input file name: ").strip()
    count = int(input("Input contract length: ").strip())

points = []
triangles = []
with open(filename) as f:
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

print('read file')

graph = [set() for _ in points]
for (a,b,c) in triangles:
    graph[a].add(b)
    graph[a].add(c)
    graph[b].add(a)
    graph[b].add(c)
    graph[c].add(a)
    graph[c].add(b)

print('built graph')

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
    for j in graph[i]:
        for k in (graph[i] & graph[j]):
            Q += triangle_quadric(points[i],points[j],points[k])
    return Q

Qs = [point_quadric(points,graph,i) for i in range(len(points))]
print('after Qs')

def edge_point(points,Qs,i,j):
    Q = Qs[i] + Qs[j]
    try:
        c = np.linalg.solve(Q[:-1,:-1],-Q[:-1,-1])
        ##print("OK")
    except np.linalg.LinAlgError:  # TODO
        print("NOT OK")
        c = (points[i] + points[j]) / 2
    c = np.append(c,1)
    error = np.dot(np.dot(c,Q),c)
    return (-error,c[:-1])


pq = EdgePriorityQueue()
for i in range(len(points)):
    for j in graph[i]:
        if i < j:
            error,c = edge_point(points,Qs,i,j)
            pq.push(error,(i,j),c)
print('after pq initialization')

def is_safe(graph,a,b):
    return len(graph[a] & graph[b]) == 2

def contract(points,graph,Qs,pq,i,j,c):
    for k in graph[i]:
        pq.discard((i,k))
    for k in graph[j]:
        pq.discard((j,k))
    
    graph[i].remove(j)
    graph[i].update(graph[j])
    for k in graph[j]:
        graph[k].discard(j)
        graph[k].add(i)
    graph[i].remove(i) # i was added once in 123 and once in 126
    graph[j] = None

    points[i] = c
    Qs[i] = point_quadric(points,graph,i)
    for k in graph[i]:
        Qs[k] = point_quadric(points,graph,k)

    for k in graph[i]:
        error,c = edge_point(points,Qs,i,k)
        pq.push(error,(i,k),c)
    

while pq and count > 0:
    (i,j),c = pq.pop()
    if is_safe(graph,i,j):
        contract(points,graph,Qs,pq,i,j,c)
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
