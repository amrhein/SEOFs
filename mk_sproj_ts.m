function[wts_ts,t] = mk_sproj_ts(data,time,patts,binsize)

    
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