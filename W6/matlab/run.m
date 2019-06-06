clear all;
close all;
clc;

load('data.mat');  % Load mesh

[TR CVs Bmask] = create_control_volumes(T, X, Y);

figure(1);
clf;
draw_control_volumes(TR, CVs, 0.05);
axis equal;
title('Computational mesh')

[A, b] = matrix_assembly(T, X, Y, CVs);

figure(2)
clf;
spy(A);
title('A fill patteren');

figure(3)
clf;
plot(sort(eig(A)),'-r','LineWidth',2);
title('Eigenspectrum of A');

phi =  A\b;

figure(4)
clf;
draw_field(T,X,Y,phi);
axis equal;
title('Solution');

figure(5)
clf;
draw_gradients(X,Y,phi);
axis equal;
title('Gradients');

figure(6)
clf;
draw_stream_lines(X,Y,phi);
axis equal;
title('Stream Lines');

figure(7)
clf;
draw_surface(X,Y,phi);
axis equal;
title('Surface');
