function [A, b] = matrix_assembly( T, X, Y, CVs )
% Copyright 2012, Kenny Erleben, DIKU

[N, ~] = size(CVs);
A = zeros(N,N);
b = zeros(N,1);

for i=1:N
  CV = CVs{i};
  
  
  E = length(CV.I); % Get the number of edge on the boundary of this CV
  for e=1:E
    
    %i = CV.I(e);  % CV index on inside of edge
    j  = CV.N(e);  % CV index on outside of edge
    l  = CV.l(e);  % Length of edge
    ox = CV.ox(e); % Origin point of edge
    oy = CV.oy(e);
    dx = CV.dx(e); % Destination point of edge
    dy = CV.dy(e);
    mx = CV.mx(e); % Midpoint on edge
    my = CV.my(e);
    nx = CV.nx(e); % Outward unit normal of edge
    ny = CV.ny(e);
    ex = CV.ex(e); % Unit edge direction vector
    ey = CV.ey(e);
    
    code = CV.code(e);
    
    
    Mx = 0;
    My = 0;
    % >>> Add your solution for computing M field <<<<<
    
    
    switch( code)
      case 0
        % edge is inside domain and everything is fine and we can use
        % 'straightforward' discretization
        
        % >>> Add your solution here <<<<<
        
      case 1
        % edge is coming from inside domain and ends on the physical
        % boundary -- we must apply some special discretization
        
        % >>> Add your solution here <<<<<
        
        
      case 2
        % edge is one the physical boundary -- we must apply some boundary
        % condition, here we got von Neumann condition saying
        %    nabla phi dot n = 0
        
        % >>> Add your solution here <<<<<
        
      otherwise
        % An error occured -- an unrecognized code was encountered?
    end
    
  end
  
end

% Add Dirichlet condition to some node n , phi_n = 0;

% >>> Add your solution here <<<<<

end