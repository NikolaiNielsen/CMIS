function draw_gradients( X, Y, phi )
% Copyright 2012, Kenny Erleben, DIKU.
min_x = min(X(:));
max_x = max(X(:));
min_y = min(Y(:));
max_y = max(Y(:));
span_x = max_x - min_x;
span_y = max_y - min_y;
N = 100;
[GX, GY] = meshgrid(...
  min_x:span_x/(N-1):max_x,...
  min_y:span_y/(N-1):max_y...
  );
DT = DelaunayTri(X,Y);
[SI BC] = pointLocation(DT, GX(:), GY(:));
T = DT(:,:);
phi_i = phi(  T(SI,1) );
phi_j = phi(  T(SI,2) );
phi_k = phi(  T(SI,3) );
C =  phi_i .* BC(:,1) +...
  phi_j .* BC(:,2) +...
  phi_k .* BC(:,3);
C = reshape(C,N,N);
[DX,DY] = gradient(C);
contour(GX,GY,C,20);
hold on
quiver(GX,GY,DX,DY)
colormap hsv
hold off
end