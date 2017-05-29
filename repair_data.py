import sys
from collections import defaultdict
import itertools

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

try:
    fileName = sys.argv[1]
except:
    print("Sample input for file name:")
    print("examples/cow.ply")
    fileName = input("Input file name: ").strip()

points = []
triangles = []
with open(fileName) as f:
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
        line = map(lambda x : x[:5],next(f).split())
        x = next(line)
        y = next(line)
        z = next(line)
        points.append((x,y,z))
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

newTriangles = [tuple(map(lambda x: change_index[unique_index[x]], tri))
                for tri in triangles]

print("Trikotniki")
print(len(triangles))
print(len(newTriangles))
print("Tocke")
print(len(points))
print(len(old_index))
file_lines = list()
for z in old_index:
    s = " ".join(list(unique_points_rep_rev[z]))
    file_lines.append(s)

for tri in newTriangles:
    s = "3 " + " ".join(map(str,tri))
    file_lines.append(s)

[nameS,nameE] = fileName.split(".")
newFileName = "".join([nameS,"_changed.",nameE])
file_lines_head = ['element vertex '+str(len(old_index)),
                   'element face '+str(len(newTriangles)),
                   'end_header \n']

with open(newFileName,"w") as f:
    f.write("\n".join(file_lines_head))
    f.write("\n".join(file_lines))
    f.write("\n")
    
