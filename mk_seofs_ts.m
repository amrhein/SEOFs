function[EOF_ts,SV_ts,t] = mk_seofs_ts(data,time,neofs,binsize)
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
    % EOF_ts    Time series (indexed by space, EOF index, time) of leading
    %           EOFs
    % SV_ts     Time series (indexed by EOF index, time) of leading
    %           singular values
    % t         Time corresponding to EOF_ts and SV_ts
    %
    % We throw away padding at the beginning and end so that all
    % computations use the same number of times.
    %
    % DEA 10/20
    
    pad                = floor(binsize/2);
    [~,td,~]           = size(data);
    
    t                  = [];
    EOF_ts             = [];
    SV_ts              = [];

    % Define a vector of time indices not in padded regions
    tis = (pad+1):(td-pad);
    
    for ii = 1:length(tis)
        
        ti = tis(ii);
        
        % Define a time range
        tr             = (ti-pad):(ti+pad);
        
        % Select time window
        dat            = data(:,tr,:);
        
        % Compute weights and save
        [sU,sS]        = mk_seofs(dat,neofs);
        EOF_ts(:,:,ii) = sU;
        SV_ts(:,ii)   = sS;
        t(ii)          = time(ti);
        
    end

end

function [sU,sS] = mk_seofs(dat,neofs)
    % Function to compute "snapshot" EOFs from ensemble model output
    %
    % INPUTS
    % data      Modal data matrix with dimensions (space,time,nens)
    % neofs     Number of snapshot EOFs to output as a function of time.
    %
    % OUTPUTS
    % sU        Left singular vectors (EOFs) computed over ensemble number
    %           and time for this particular time bin
    % sS        Same as sU, but a vector of singular values

    [sd,td,nd]   = size(dat);
    datnm        = reshape(dat,sd,td*nd)-mean(reshape(dat,sd,td*nd),2);
    [sUa,sSa,~]  = svd(datnm,'econ');
    sU           = sUa(:,1:neofs);
    sS           = diag(sSa(1:neofs,1:neofs));

end

