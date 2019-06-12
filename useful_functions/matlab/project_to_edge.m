function [px, py] = project_to_edge(xi, yi, xj, yj, x, y)
  tmp_dx  = xj-xi;
  tmp_dy  = yj-yi;
  tmp_l   = sqrt((tmp_dx*tmp_dx) + (tmp_dy*tmp_dy));
  tmp_ex  = tmp_dx/tmp_l;
  tmp_ey  = tmp_dy/tmp_l;
  tmp_dot = (x-xi)*tmp_ex +(y-yi)*tmp_ey;
  px      = tmp_dot*tmp_ex + xi;
  py      = tmp_dot*tmp_ey + yi;
end