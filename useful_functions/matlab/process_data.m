function process_data(cwd)
load(fullfile(cwd, 'points.mat'))
[N, M] = size(X);
if N == 1
    X = X';
    Y = Y';
end
Tr = delaunayTriangulation(X, Y);
T = Tr.ConnectivityList;
[TR CVs Bmask] = create_control_volumes(Tr, X, Y);
save(fullfile(cwd, 'control_volumes'), 'CVs', 'X', 'Y', 'T');