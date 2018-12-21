

[Y,z,mu,ss,p] = drawGmm(2000);
subplot(1,2,1);
title('generative clusters');
scatterMixture(Y,z);
clusters = safe_dpmm(Y, 20, 100);
subplot(1,2,2);
title('dpmm clustering');
scatterMixture(Y,clusters);