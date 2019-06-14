function process_data(cwd, outfile)

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

% Update X and Y if needed (delaunay may have deleted a point or two)
X = Tr.Points(:,1);
Y = Tr.Points(:,2);
[TR CVs Bmask] = create_control_volumes(Tr, X, Y);

% Save the control volumes in python cwd
save(fullfile(cwd, outfile), 'CVs', 'X', 'Y', 'T');