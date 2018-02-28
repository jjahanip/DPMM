% Copyright (C) 2007 Jacob Eisenstein: jacobe at mit dot edu
% distributable under GPL, see README.txt

[Y,z,mu,ss,p] = drawGmm(2000);
subplot(1,2,1);
title('generative clusters');
scatterMixture(Y,z);
params = vdpmm(Y,100);
subplot(1,2,2);
title('dpmm clustering');
scatterMixture(Y,params(end).classes);



figure,
subplot(1,2,1);
title('generative clusters');
scatterMixture(Y,z);
subplot(1,2,2);
title('dpmm clustering');
scatterMixture(Y,params(20).classes);

