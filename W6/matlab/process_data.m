load('points.mat')
Tr = DelaunayTri(X', Y');
T = Tr.Triangulation;
[TR CVs Bmask] = create_control_volumes(Tr, X', Y');
save('control_volumes', 'CVs', 'X', 'Y', 'T');