# surface-simplification
A project for a class on Computational Topology

The goal of this project is to implement a surface simplification method as presented in [1].

A surface is defined by a mesh of triangles.
The simplification progresses by performing edge contractions (each eliminating either one or two triangles) until the desired number of remaining triangles is reached.
The edge to be contracted is chosen based on an error measure that quantifies the damage its contration will cause to the overall shape. (see page 52 of [1])

[1] Edelsbrunner, Herbert, and John Harer. Computational topology: an introduction. American Mathematical Soc., 2010.

## Dependencies

NumPy and SortedContainers Python packages are required for `main.py`. Additionally, Matplotlib is required for `plot.py`.

## Running

The main simplification algorithm is implemented in `main.py'.
It works on a triangulated surface given in [PLY](https://en.wikipedia.org/wiki/PLY_(file_format)) format.

The first command line argument is a `.ply` file, the second is the desired number of triangles after simplification, e.g.,

`python3 main.py examples/cow.ply 200 > out.ply`.

The program outputs a new mesh in ply format to standard output.

`plot.py` can be used to visualize the surfaces, e.g.,

`python3 plot.py out.ply`.
