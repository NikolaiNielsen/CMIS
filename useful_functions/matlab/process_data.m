function process_data(cwd)

% Load the "points.mat" file from the python cwd
load(fullfile(cwd, 'points.mat'))

% transpose X and Y if needed - Delaunay wants column format
[N, M] = size(X);
if N == 1
    X = X';
    Y = Y';
end

% Perform delaunayTriangulation so we can get connectivity matrix to create_control_volumes
Tr = delaunayTriangulation(X, Y);
T = Tr.ConnectivityList;

[TR CVs Bmask] = create_control_volumes(Tr, X, Y);

% Save the control volumes in python cwd
save(fullfile(cwd, 'control_volumes'), 'CVs', 'X', 'Y', 'T');