function draw_control_volumes( TR, CVs, scale)
% Copyright 2012, Kenny Erleben, DIKU
triplot(TR);
hold on;
[N, M] = size(CVs);
for i=1:N
  CV = CVs{i};
  plot([CV.ox CV.dx],[CV.oy CV.dy],'-r','LineWidth',2);
  plot(CV.ox, CV.oy,'*g','LineWidth',2);
  plot(CV.mx, CV.my,'*b','LineWidth',2);
  plot([CV.ox  scale*CV.ex+CV.ox]', [CV.oy  scale*CV.ey+CV.oy]','-g','LineWidth',2);
  plot([CV.mx  scale*CV.nx+CV.mx]', [CV.my  scale*CV.ny+CV.my]','-b','LineWidth',2);
end
hold off;
end