function[wts_ts,t] = mk_sproj_ts(data,time,patts,binsize)
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

end

function [wts] = mk_sproj(dat,patts)
    % Function to compute "snapshot" projections of patterns onto ensemble
    % model output
    % INPUTS
    % data      Modal data matrix with dimensions (space,time,nens)
    % patts     Matrix of spatial patterns (space along rows) to project
    %
    % OUTPUTS
    % wts       Vector of SVD-like weights computed for each pattern by
    %           summing projections of ensemble number and time
    %
    % Goal is to get a weighting like an EOF. So the procedure is to
    % project each pattern onto the data, ending up with a vector (indexed
    % by time and nens). Then normalize that vector so that v'v*v = 1 (like
    % a PC). The normalization is the weight.

    [sd,td,nd] = size(dat);
    datnm      = reshape(dat,sd,td*nd)-mean(reshape(dat,sd,td*nd),2);
    proj       = datnm'*patts;
    wts        = sqrt(sum(proj.*proj,2));

end
