import sys
from collections import defaultdict
import itertools
from sortedcontainers import SortedDict

import numpy as np

class EdgePriorityQueue:
    def __init__(self):
        self.tree = SortedDict()
        self.errors = dict()

    def __bool__(self):
        return bool(self.tree)

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

def plane(a,b,c):
    n = np.cross(b-a,c-a)
    n = n / np.linalg.norm(n)
    d = -n.dot(a)
    return np.append(n,d)

def triangle_quadric(a,b,c):
    u = plane(a,b,c)
    return np.outer(u,u)

class Surface:
    def from_ply(self,f):
        line = next(f).split()
        while line[0] != 'element':
            line = next(f).split()
        num_points = int(line[2])
        self.n = num_points
        
        line = next(f).split()
        while line[0] != 'element':
            line = next(f).split()
        num_triangles = int(line[2])

        line = next(f).split()
        while line[0] != 'end_header':
            line = next(f).split()

        self.points = []
        for _ in range(num_points):
            line = map(float,next(f).split())
            x = next(line)
            y = next(line)
            z = next(line)
            self.points.append( np.array((x,y,z)) )

        self.graph = [set() for _ in range(num_points)]
        self.Qs = [np.zeros((4,4)) for _ in range(num_points)]
        for _ in range(num_triangles):
            i,j,k = sorted(map(int,next(f).split()[1:]))
            self.graph[i].update([j,k])
            self.graph[j].update([i,k])
            self.graph[k].update([i,j])
            Q = triangle_quadric(self.points[i],self.points[j],self.points[k])
            self.Qs[i] += Q
            self.Qs[j] += Q
            self.Qs[k] += Q
     
    def ply(self):
        points,triangles = self.points_and_triangles()
        lines = []
        lines.append('ply')
        lines.append('format ascii 1.0')
        lines.append('element vertex {}'.format(len(points)))
        lines.append('property float x')
        lines.append('property float y')
        lines.append('property float z')
        lines.append('element face {}'.format(len(triangles)))
        lines.append('property list uchar uint vertex_indices')
        lines.append('end_header')
        for p in points:
            lines.append(' '.join(map(str,p)))
        for t in triangles:
            lines.append('3 ' + ' '.join(map(str,t)))
        return '\n'.join(lines)

    def from_points_and_triangles(self,points,triangles):
        self.points = []
        for point in points:
            self.points.append(point)
        self.n = len(self.points)

        self.graph = [set() for _ in points]
        self.Qs = [np.zeros((4,4)) for _ in points]
        for (i,j,k) in triangles:
            self.graph[i].update([j,k])
            self.graph[j].update([i,k])
            self.graph[k].update([i,j])
            Q = triangle_quadric(self.points[i],self.points[j],self.points[k])
            self.Qs[i] += Q
            self.Qs[j] += Q
            self.Qs[k] += Q

    def points_and_triangles(self):
        points = []
        idxs = dict()
        count = 0
        for i in range(self.n):
            if self.points[i] is not None:
                points.append(self.points[i])
                idxs[i] = count
                count += 1
        triangles = [(idxs[i],idxs[j],idxs[k])
                        for i in idxs.keys()
                            if self.graph[i]
                                for j in self.graph[i]
                                    if i < j
                                        for k in (self.graph[i] & self.graph[j])
                                            if j < k]
        return points,triangles

    def point_quadric(self,i):
        Q = np.zeros((4,4))
        for j in self.graph[i]:
            for k in (self.graph[i] & self.graph[j]):
                Q += triangle_quadric(self.points[i],self.points[j],self.points[k])
        return Q

    def edge_point(self,i,j):
        Q = self.Qs[i] + self.Qs[j]
        try:
            c = np.linalg.solve(Q[:-1,:-1],-Q[:-1,-1])
        except np.linalg.LinAlgError:  # TODO
            print("NOT OK",file=sys.stderr)
            c = (self.points[i] + self.points[j]) / 2
        c = np.append(c,1)
        error = np.dot(np.dot(c,Q),c)
        return (-error,c[:-1])

    def is_safe(self,i,j):
        return len(self.graph[i] & self.graph[j]) == 2

    def contract(self,i,j,c):
        self.points[i] = c
        self.points[j] = None
        
        self.graph[i].remove(j)
        self.graph[i].update(self.graph[j])
        for k in self.graph[j]:
            self.graph[k].discard(j)
            self.graph[k].add(i)
        self.graph[i].remove(i) # i was added once before the for loop and once in it
        self.graph[j] = None

        if msmQ:
            self.Qs[i] = self.point_quadric(i)
            for k in self.graph[i]:
                self.Qs[k] = self.point_quadric(k)
        else:
            self.Qs[i] = self.Qs[i] + self.Qs[j]
        self.Qs[j] = None

def contract(surface,pq,i,j,c):
    for k in surface.graph[i]:
        pq.discard((i,k))
    for k in surface.graph[j]:
        pq.discard((j,k))
    surface.contract(i,j,c)
    for k in surface.graph[i]:
        error,c = surface.edge_point(i,k)
        pq.push(error,(i,k),c)
        

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
        count = int(sys.argv[2])
        msmQ = bool(int(sys.argv[3]))
    except:
        print("Sample input for file name:")
        print("examples/cow.ply")
        filename = input("Input file name: ").strip()
        count = int(input("Input contract length: ").strip())

    surface = Surface()
    with open(filename) as f:
        surface.from_ply(f)

    pq = EdgePriorityQueue()
    for i in range(surface.n):
        for j in surface.graph[i]:
            if i < j:
                error,c = surface.edge_point(i,j)
                pq.push(error,(i,j),c)
        
    while pq and count > 0:
        (i,j),c = pq.pop()
        if surface.is_safe(i,j):
            contract(surface,pq,i,j,c)
            count -= 1

    print(surface.ply())
