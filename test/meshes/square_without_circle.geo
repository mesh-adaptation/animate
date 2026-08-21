// Copied from firedrake/tests/meshes
Point(1) = {-2, 2, 0, .5};
Point(2) = {-2, -2, 0, .5};
Point(3) = {2, -2, 0, .5};
Point(4) = {2, 2, 0, .5};

Point(5) = {0, 0, 0, .5};
Point(6) = {1, 0, 0, .5};
Point(7) = {-1, 0, 0, .5};
Point(8) = {0, 1, 0, .5};
Point(9) = {0, -1, 0, .5};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {8, 5, 6, 7};
Plane Surface(11) = {9, 10};
Plane Surface(12) = {10};
Physical Line(1) = {1, 2, 3, 4};
Physical Line(2) = {8};
Physical Line(3) = {5, 6, 7};
Physical Surface("SquareWithoutCircleSurface") = {11};
