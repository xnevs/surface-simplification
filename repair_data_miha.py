import sys
from collections import defaultdict
import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = []
triangles = []
file = sys.argv[1]
with open(file) as f:
    num_points = int(next(itertools.islice(f,3,4)).split()[2])
    num_triangles = int(next(itertools.islice(f,6,7)).split()[2])
    next(itertools.islice(f,2,2),None)
    for i in range(num_points):
        line = map(lambda x : x[:5],next(f).split())
        x = next(line)
        y = next(line)
        z = next(line)
        points.append( (x,y,z) )
    for i in range(num_triangles):
        _,a,b,c = map(int,next(f).split())
        triangles.append((a,b,c))


unique_points = dict()

for i in range(len(points)):
    p = points[i]
    if p in unique_points:
        unique_points[p].append(i)
    else:
        unique_points[p] = [i]

unique_index = dict()
unique_points_rep = dict()

for p in unique_points:
    sameInd = unique_points[p]
    for z in sameInd:
        unique_index[z] = sameInd[0]
        
    unique_points_rep[p] = sameInd[0]

unique_points_rep_rev = dict()
for p in unique_points_rep:
    unique_points_rep_rev[unique_points_rep[p]] = p


points_index = [unique_points_rep[p] for p in points]
old_index = sorted(set(points_index))
new_index = [z for z in range(len(old_index))]
change_index = dict([(old_index[i],i) for i in new_index])

print(len(old_index))
file_lines = list()

newTriangles = [tuple(map(lambda x: change_index[unique_index[x]], tri)) for tri in triangles]

s = """ply
format ascii 1.0
element vertex """+ str(len(old_index))+""" 
property float32 x
property float32 y
property float32 z
element face """ + str(len(newTriangles)) + """ 
property list uint8 int32 vertex_indices
end_header"""
file_lines.append(s)



for z in old_index:
    s = " ".join(list(unique_points_rep_rev[z]))
    file_lines.append(s)



for tri in newTriangles:
    s = "3 "+ str(tri[0]) + " " + str(tri[1]) + " " +str(tri[2])
    file_lines.append(s)
print(len(newTriangles))
with open(file+"_changed.ply",'w') as f:
    f.write("\n".join(file_lines))
    
