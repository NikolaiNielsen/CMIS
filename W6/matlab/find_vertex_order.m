function [i, j, k] = find_vertex_order(i, a, b, c)
% Copyright 2012, Kenny Erleben, DIKU
    if(i==a)
      j = b;
      k = c;
    end
    if(i==b)
      j = c;
      k = a;
    end
    if(i==c)
      j = a;
      k = b;
    end
end