import sys
from collections import defaultdict
import itertools

import numpy as np

points = []
triangles = []
with open(sys.argv[1]) as f:
    num_points = int(next(itertools.islice(f,3,4)).split()[2])
    num_triangles = int(next(itertools.islice(f,6,7)).split()[2])
    next(itertools.islice(f,2,2),None)
    for i in range(num_points):
        x,y,z,_,_,_ = map(float,next(f).split())
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

def plane(points,a,b,c):
    """ (a,b,c) je trikotnik
        vrne njegovo ravnino
    """
    u = points[a]
    v = points[b]
    w = points[c]
    n = np.cross(v-u,w-u)
    n = n / np.linalg.norm(n)
    d = -n.dot(u)
    return np.append(n,d)

def triangle_quadric(points,a,b,c):
    u = plane(points,a,b,c)
    return np.outer(u,u)

def point_quadric(points,graph,i):
    Q = np.zeros((4,4))
    for j,k in itertools.combinations(graph[i],2):
        if k in graph[j]:
            Q += triangle_quadric(points,i,j,k)
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
    pq = new_pq

while pq:
    _,(i,j),c = pq.pop()
    if is_safe(graph,i,j):
        contract(points,graph,Qs,pq,i,j,c)

print(graph)
