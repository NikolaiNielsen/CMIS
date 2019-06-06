function draw_field( T, X, Y, phi )
% Copyright 2012, Kenny Erleben, DIKU.
pX = [X(T(:,1)) X(T(:,2)) X(T(:,3)) ]';
pY = [Y(T(:,1)) Y(T(:,2)) Y(T(:,3)) ]';
pC = [phi(T(:,1)) phi(T(:,2)) phi(T(:,3))]';

patch( pX, pY, pC );
colormap(gray) %((jet+white)/2);
colorbar
shading interp;
end