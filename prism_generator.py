def prism_generator(polyXY, polyT, h, hsize):
    points = []
    for i in range(h + 1):
        for (x,y) in polyXY:
            points.append((x,y,i*hsize))

    triangles = []
    c = h * len(polyXY)
    for (i1,i2,i3) in polyT:
        ##spodnja osnovnica
        triangles.append((i1,i2,i3))
    for (i1,i2,i3) in polyT:
        ##zgornja osnovnica
        triangles.append((i1+c,i2+c,i3+c))

    lp = len(polyXY)
    for i in range(h):
        ic0 = i * lp
        ic1 = ic0 + lp
        for j in range(len(polyXY)):
            i00, i01 = j, (j+1)%lp
            (i00,i01,i10,i11) = (i00 + ic0, i01 + ic0, i00 + ic1, i01 + ic1)
            triangle1 = (i00, i01, i10)
            triangle2 = (i01, i10, i11)
            triangles.extend([triangle1, triangle2])

    return points, triangles

def write_to_file(fileName, points, triangles):
    file_lines = list()
    for p in points:
        s = " ".join(map(str,p))
        file_lines.append(s)

    for tri in triangles:
        s = "3 " + " ".join(map(str,tri))
        file_lines.append(s)

    file_lines_head = ['element vertex '+str(len(points)),
                       'element face '+str(len(triangles)),
                       'end_header \n']

    with open(fileName,"w") as f:
        f.write("\n".join(file_lines_head))
        f.write("\n".join(file_lines))
        f.write("\n")


def prism_to_file(file, polyXY, polyT, h, hsize):
    if h < 1:
        print("slab h")
        return
    ##print((polyXY, polyT, h))
    points, triangles = prism_generator(polyXY, polyT, h, hsize)
    write_to_file(file, points, triangles)


def main():
    file = "prism.ply"

    polyXY = [(0,0),(100,0),(200,120),(300,0),(400,0),
              (400,100),(290,100),(200,130),(110,100),(0,100)]

    polyT = [(0,1,9),(1,9,8),
             (1,2,8),(2,8,7),
             (2,6,7),(2,3,6),
             (3,5,6),(3,4,5)]
    h = 1

    hsize = 550
    
    prism_to_file(file, polyXY, polyT, h, hsize)

if __name__ == "__main__":
    main()
