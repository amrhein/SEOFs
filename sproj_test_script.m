% sproj_test_script
% Seems to work at least dimensionally. Output from this experiment is a
% 5x10 matrix, corresponding to the number of patterns times the number of
% times.

data    = randn(20,10,13);
time    = 1:10;
patts   = randn(20,5);
binsize = 1;

% Function to compute "snapshot" projections of patterns onto ensemble
% model output
% INPUTS
% data      Modal data matrix with dimensions (space,time,nens)
% time      Time axis of model output
% patts     Matrix of spatial patterns (space along rows) to project
% binsize   Odd (symmetric) length of bin in time in which to compute
%           SEOFs
%
% OUTPUTS
% wts_ts    Time series (indexed by column) of weights corresponding to
%           different patterns (indexed by row)
% t         Time for weight time series
%
% We throw away padding at the beginning and end so that all
% computations use the same number of times.
%
% DEA 10/20

pad              = floor(binsize/2);
[~,td,~]         = size(data);

t                = [];
wts_ts           = [];

% Define a vector of time indices not in padded regions
tis = (pad+1):(td-pad);

for ii = 1:length(tis)

    ti = tis(ii);

    % Define a time range
    tr           = (ti-pad):(ti+pad);

    % Select time window
    dat          = data(:,tr,:);

    % Compute weights and save
    wts_ts(:,ii) = mk_sproj(dat,patts);
    t(ii)        = time(ti);
end
