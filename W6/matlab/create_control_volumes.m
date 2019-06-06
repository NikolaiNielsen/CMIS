function [TR, CVs, Bmask] = create_control_volumes( T, X, Y )
% Copyright 2012, Kenny Erleben, DIKU
TR = TriRep(T, X, Y); % Gets all triangle vertex coordinates
IC = incenters(TR); % incenters. Use barycentric coords and side lengths
CX = IC(:,1);
CY = IC(:,2);

FB = freeBoundary(TR); %  Find the convex hull - use scipy.spatial.ConvexHull
Bmask = zeros(size(X));
Bmask(unique( FB(:) )) = 1;

N = length(X);
CVs = cell(N,1);
for i=1:N
  
  temp = vertexAttachments(TR,i); % Get indices of triangles attached to vertex
  indices = temp{:};
  
  I  = [];
  N  = [];
  OX = [];
  OY = [];
  DX = [];
  DY = [];
  L  = [];
  EX = [];
  EY = [];
  NX = [];
  NY = [];
  MX = [];
  MY = [];
  code = [];
  
  K = length(indices);
  
  if Bmask(i) == 1
    
    a = T(indices(1),1);
    b = T(indices(1),2);
    c = T(indices(1),3);
    [ii, jj, kk] = find_vertex_order( i, a, b, c);
    
    %-----------------------------------------------------------
    ox = X(i);
    oy = Y(i);
    [dx, dy] = project_to_edge(X(i), Y(i), X(jj), Y(jj), CX(indices(1)), CY(indices(1)));
    if i == 1
       fprintf('ox: %4.2f, oy: %4.2f\n', ox, oy)
       fprintf('dx: %4.2f, dy: %4.2f\n', dx, dy)
    end
    l = sqrt( (dx-ox).^2 + (dy-oy).^2 );  % Edge length
    ex = (dx - ox)./l;  % Unit edge direction vector
    ey = (dy - oy)./l;
    nx = -ey;            % Inward unit normal
    ny =  ex;
    mx = (dx+ox)./2.0;   % Edge min point
    my = (dy+oy)./2.0;
    
    I  = [I;  i];
    N  = [N; -1];
    OX = [OX; ox];
    OY = [OY; oy];
    DX = [DX; dx];
    DY = [DY; dy];
    L  = [L; l];
    EX = [EX; ex];
    EY = [EY; ey];
    NX = [NX; -nx];  % Flip to get outward unit normal
    NY = [NY; -ny];
    MX = [MX; mx];
    MY = [MY; my];
    code = [code; 2];
    
    %-----------------------------------------------------------
    ox =  dx;
    oy =  dy;
    dx = CX(indices(1));  % Destination point
    dy = CY(indices(1));
    
    l = sqrt( (dx-ox).^2 + (dy-oy).^2 );  % Edge length    
    ex = (dx - ox)./l;  % Unit edge direction vector
    ey = (dy - oy)./l;
    nx = -ey;            % Inward unit normal
    ny =  ex;  
    mx = (dx+ox)./2.0;   % Edge min point
    my = (dy+oy)./2.0;

    I  = [I;  i];
    N  = [N;  jj];
    OX = [OX; ox];
    OY = [OY; oy];
    DX = [DX; dx];
    DY = [DY; dy];
    L  = [L; l];
    EX = [EX; ex];
    EY = [EY; ey];
    NX = [NX; -nx];  % Flip to get outward unit normal
    NY = [NY; -ny];
    MX = [MX; mx];
    MY = [MY; my];
    code = [code; 1];
    
  end
  
  lastK = K;
  if Bmask(i) == 1
    lastK = K - 1;  % Make sure the for loop does not close the CV when visting the last triangle in the CV.
  end
  
  for j=1:lastK
    
    a = T(indices(j),1);
    b = T(indices(j),2);
    c = T(indices(j),3);
    [ii, jj, kk] = find_vertex_order( i, a, b, c);
    
    o = indices(j);           % Origin vertex index
    d = indices(mod(j,K)+1);  % Destination vertex index
    ox = CX(o);  % Origin point
    oy = CY(o);
    dx = CX(d);  % Destination point
    dy = CY(d);
    
    l = sqrt( (dx-ox).^2 + (dy-oy).^2 );  % Edge length
    ex = (dx - ox)./l;  % Unit edge direction vector
    ey = (dy - oy)./l;
    nx = -ey;            % Inward unit normal
    ny =  ex;
    mx = (dx+ox)./2.0;   % Edge min point
    my = (dy+oy)./2.0;
        
    I  = [I;  i];    
    N  = [N;  kk];
    OX = [OX; ox];
    OY = [OY; oy];
    DX = [DX; dx];
    DY = [DY; dy];
    L  = [L; l];
    EX = [EX; ex];
    EY = [EY; ey];
    NX = [NX; -nx];  % Flip to get outward unit normal
    NY = [NY; -ny];
    MX = [MX; mx];
    MY = [MY; my];
    code = [code; 0];
    
  end
  
  
  if Bmask(i) == 1
    
    a = T(indices(K),1);
    b = T(indices(K),2);
    c = T(indices(K),3);
    [ii, jj, kk] = find_vertex_order( i, a, b, c);
        
    %-----------------------------------------------------------
    
    ox = CX(indices(K));  % Origin point
    oy = CY(indices(K));
    [dx, dy] = project_to_edge(X(kk), Y(kk), X(i), Y(i), ox, oy);
    
    l = sqrt( (dx-ox).^2 + (dy-oy).^2 );  % Edge length
    ex = (dx - ox)./l;  % Unit edge direction vector
    ey = (dy - oy)./l;
    nx = -ey;            % Inward unit normal
    ny =  ex;
    mx = (dx+ox)./2.0;   % Edge min point
    my = (dy+oy)./2.0;
    
    I  = [I; i];
    N  = [N; kk];
    OX = [OX; ox];
    OY = [OY; oy];
    DX = [DX; dx];
    DY = [DY; dy];
    L  = [L; l];
    EX = [EX; ex];
    EY = [EY; ey];
    NX = [NX; -nx];  % Flip to get outward unit normal
    NY = [NY; -ny];
    MX = [MX; mx];
    MY = [MY; my];
    code = [code; 1];
    
    %---------------------------------------------------------------------
    
    ox = dx;
    oy = dy;
    dx = X(i);
    dy = Y(i);
    
    l = sqrt( (dx-ox).^2 + (dy-oy).^2 );  % Edge length
    ex = (dx - ox)./l;  % Unit edge direction vector
    ey = (dy - oy)./l;
    nx = -ey;            % Inward unit normal
    ny =  ex;
    mx = (dx+ox)./2.0;   % Edge min point
    my = (dy+oy)./2.0;
    I  = [I;  i];
    N  = [N; -1];
    OX = [OX; ox];
    OY = [OY; oy];
    DX = [DX; dx];
    DY = [DY; dy];
    L  = [L; l];
    EX = [EX; ex];
    EY = [EY; ey];
    NX = [NX; -nx];  % Flip to get outward unit normal
    NY = [NY; -ny];
    MX = [MX; mx];
    MY = [MY; my];
    code = [code; 2];
    
  end
  
  CV = struct(...
    'I',I,...
    'N',N,...
    'ox',OX,...
    'oy',OY,...
    'dx',DX,...
    'dy',DY,...
    'l',L,...
    'ex',EX,...
    'ey',EY,...
    'nx',NX,...
    'ny',NY,...
    'mx',MX,...
    'my',MY,...
    'code', code...
    );
  
  CVs{i} = CV ;
end

end
