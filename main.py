import sys
from collections import defaultdict
import itertools
from sortedcontainers import SortedDict

import numpy as np

class EdgePriorityQueue:
    """A priority queue for edge contractions.
    Each element contains:
        * the edge to be contracted ,
        * the error measure for the contraction ,
        * the point the edge will contract to .
    """

    def __init__(self):
        """Initialize the priority queue."""
        # self.tree is used to store the edges in sorted
        # order according to the error.
        self.tree = SortedDict()
        # self.errors represents a mapping (edge --> error)
        # that enables us to find a given edge in self.tree.
        # (required for discard)
        self.errors = dict()

    def __bool__(self):
        """True if it contains some elements."""
        return bool(self.tree)

    def __len__(self):
        """Number of elements in the queue."""
        return len(self.tree)

    def push(self,error,e,c):
        """Add an edge with the given error and new point (c)
        to the priority queue.
        """
        # always store the vertices of an edge in sorted order
        # to avoid storing the same edge twice (as (i,j) and (j,i))
        if e[1] < e[0]:
            e = (e[1],e[0])
        self.errors[e] = error
        self.tree[(error,e)] = c

    def pop(self):
        """Remove and return the edge with the minimal error."""
        (error,e),c = self.tree.popitem()
        self.errors.pop(e)
        return (e,c)

    def discard(self,e):
        """Discard a given edge from the queue (if present)."""
        if e[1] < e[0]:
            e = (e[1],e[0])
        try:
            self.tree.pop((self.errors[e],e))
            self.errors.pop(e)
        except KeyError:
            pass

def plane(a,b,c):
    """Return the unit normal to the plane
    spanned by the points a, b and c."""
    n = np.cross(b-a,c-a)
    n = n / np.linalg.norm(n)
    d = -n.dot(a)
    return np.append(n,d)

def triangle_quadric(a,b,c):
    """Return the quadric for the triangle <a,b,c>."""
    u = plane(a,b,c)
    return np.outer(u,u)

class Surface:
    """A class that represents a triangulated surface
    (a 2-manifold without boundary).
    """
    def __init__(self):
        # self.transform required to recall the original triangles
        self.transform = dict()
        
    def from_ply(self,f):
        """Initialize the surface from a .ply file."""
        line = next(f).split()
        while line[0] != 'element':
            line = next(f).split()
        num_points = int(line[2])
        self.n = num_points
        
        line = next(f).split()
        while line[0] != 'element':
            line = next(f).split()
        self.num_triangles = int(line[2])

        line = next(f).split()
        while line[0] != 'end_header':
            line = next(f).split()

        points = (np.array(list(map(float,line.split()[:3]))) for line in itertools.islice(f,num_points))
        self.init_points(points)

        triangles = (tuple(map(int,line.split()[1:4])) for line in itertools.islice(f,self.num_triangles))
        self.init_triangles(triangles)
     
    def ply(self):
        """Return the surface as a string in ply format."""
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

    def init_points(self,points):
        """Initialize the vertices of the surface."""
        self.points = []
        for point in points:
            self.points.append(point)
        self.n = len(self.points)

        self.graph = [defaultdict(lambda: np.zeros((4,4))) for _ in range(self.n)]
        self.Qs = [np.zeros((4,4)) for _ in range(self.n)]
        self.transform = dict([(i,i) for i in range(len(self.points))])

    def init_triangles(self,triangles):
        """Initialize the trinagles."""
        triangles = list(triangles)
        self.begin_triangles = triangles
        for (i,j,k) in triangles:
            Q = triangle_quadric(self.points[i],self.points[j],self.points[k])
            self.Qs[i] += Q
            self.Qs[j] += Q
            self.Qs[k] += Q
            self.graph[i][j] += Q
            self.graph[i][k] += Q
            self.graph[j][i] += Q
            self.graph[j][k] += Q
            self.graph[k][i] += Q
            self.graph[k][j] += Q

    def from_points_and_triangles(self,points,triangles):
        """Initialize the surface from a list of points
        and a list of triangles."""
        self.init_points(points)
        self.init_triangles(triangles)

    def points_and_triangles(self):
        """Return the lists of points and triangles
        of the surface."""
        points = []
        idxs = dict()
        count = 0
        for i in range(self.n):
            if self.points[i] is not None:
                points.append(self.points[i])
                idxs[i] = count
                count += 1
                
        trianglesB = [tuple(sorted((i,j,k)))
                        for i in idxs.keys()
                            if self.graph[i]
                                for j in self.graph[i].keys()
                                    if i < j
                                        for k in (self.graph[i].keys() & self.graph[j].keys())
                                            if j < k]


        for i in range(len(self.points)):
            while self.transform[self.transform[i]] != self.transform[i]:
                self.transform[i] = self.transform[self.transform[i]]
        
        trianglesA = list()
        for tri in self.begin_triangles:
            trianglesA.append(tuple(sorted(map(lambda x: self.transform[self.transform[x]],tri))))

        triangles = set(trianglesB).intersection(set(trianglesA))
        triangles = sorted([tuple(map(lambda x: idxs[x],tri)) for tri in triangles])
        
        return points,triangles

    def point_quadric(self,i):
        """Calculate the quadric for the point at index i."""
        Q = np.zeros((4,4))
        # iterate over all triangles that contain the point i
        for j in self.graph[i].keys():
            for k in (self.graph[i].keys() & self.graph[j].keys()):
                Q += triangle_quadric(self.points[i],self.points[j],self.points[k])
        return Q

    def edge_point(self,i,j):
        """Calculate the best point to contract an edge to
        and the error for the contraction."""
        Q = self.Qs[i] + self.Qs[j]
        try:
            c = np.linalg.solve(Q[:-1,:-1],-Q[:-1,-1])
        except np.linalg.LinAlgError:
            c = (self.points[i] + self.points[j]) / 2
        c = np.append(c,1)
        error = np.dot(np.dot(c,Q),c)
        # return -error because the edges are
        # to be stored in a max priority queue
        return (-error,c[:-1])

    def is_safe(self,i,j):
        """Return True if the link condition holds for (i,j)."""
        return len(self.graph[i].keys() & self.graph[j].keys()) <= 2

    def contract(self,i,j,c):
        """Contract the edge (i,j) to a new point c."""
        self.transform[j] = i
        a = self.points[i]
        b = self.points[j]

        # set i to the new point c
        self.points[i] = c
        # invalidate j
        self.points[j] = None

        # store the quadric of the contracted edge
        # needed for: Q_c = Q_a + Q_b - Q_{ab} below
        Qab = self.graph[i][j]

        # subtract the quadrics of the triangles incident
        # to the edge (i,j)
        count = 0
        for k in self.graph[i].keys() & self.graph[j].keys():
            self.graph[i][k] -= triangle_quadric(a,b,self.points[k])
            self.graph[k][i] -= triangle_quadric(a,b,self.points[k])
            count += 1

        # move the edges from j to i
        for k in self.graph[j]:
            Qe = self.graph[k].pop(j)
            self.graph[i][k] += Qe # + only for x and y (because of the link condition others are zero)
            self.graph[k][i] += Qe # + only for x and y (because of the link condition others are zero)
        self.graph[i].pop(i) # i was added once before the for loop and once in it
        # invalidate j
        self.graph[j] = None

        # calculate the new quadric
        if msmQ: # use method 3 for the error measure
            self.Qs[i] = self.point_quadric(i)
            for k in self.graph[i]:
                self.Qs[k] = self.point_quadric(k)
        else: # use method 4 for the error measure
            self.Qs[i] = self.Qs[i] + self.Qs[j] - Qab
        # invalidate j
        self.Qs[j] = None
        # return the number of removed triangles (either 1 or 2)
        return count

def contract(surface,pq,i,j,c):
    """Contract the edge (i,j) to the point c.
    Incident edges are removed from the priority queue,
    the edge is contracted and then the new edges are reinserted."""
    for k in surface.graph[i]:
        pq.discard((i,k))
    for k in surface.graph[j]:
        pq.discard((j,k))
    count = surface.contract(i,j,c)
    for k in surface.graph[i]:
        error,c = surface.edge_point(i,k)
        pq.push(error,(i,k),c)
    return count
        

if __name__ == '__main__':
    try:
        filename = sys.argv[1] # a ply file
        count = int(sys.argv[2]) # the desired number of triangles
        if len(sys.argv) > 3:
            # Should we use method 3 for the error measure?
            msmQ = bool(int(sys.argv[3]))
        else:
            msmQ = False
    except:
        print("Sample input for file name:")
        print("examples/cow.ply")
        filename = input("Input file name: ").strip()
        count = int(input("Input contract length: ").strip())
        msmQ = False

    # create a new surface
    surface = Surface()
    # read it from a ply file
    with open(filename) as f:
        surface.from_ply(f)

    # create an empty priority queue for edge contractions
    pq = EdgePriorityQueue()
    # fill it with the edges
    for i in range(surface.n):
        for j in surface.graph[i]:
            if i < j:
                error,c = surface.edge_point(i,j)
                pq.push(error,(i,j),c)

    # calculate the number of triangles we must eliminate
    count = surface.num_triangles - count

    # keep contracting the edges until the desired
    # number of triangles is reached or no further
    # contractions are possible
    while count>0 and pq:
        # pick the edge with the minimal error
        (i,j),c = pq.pop()
        # check if it is safe to contract
        if surface.is_safe(i,j):
            # and contract it
            count -= contract(surface,pq,i,j,c)

    # output the result in ply format to stdout
    print(surface.ply())
