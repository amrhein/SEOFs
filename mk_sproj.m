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
    wts        = sqrt(sum(proj.*proj,1));


