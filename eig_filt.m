% Make an eigenvector filter. See if I can make commuting matrices by
% making small adjustments st two datasets have the same eigenvectors.

s0 = 8;
s1 = 5;

Z0 = randn(10,s0);
Z1 = randn(10,s1);

C0 = 1/(s0-1)*Z0*Z0';
C1 = 1/(s1-1)*Z1*Z1';

% Show that they don't commute

C0*C1-C1*C0

% Now build a filter

[u0,s,v] = svd(Z0*Z0','econ');

C0f = u0'*C1

[ut, st, vt] = svd(C0f*C0f')
