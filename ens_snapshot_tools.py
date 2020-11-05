# ens_snapshot_tools.py
# D Amrhein, November 2020

import numpy as np

def mk_seofs_ts(data,time,neofs,binsize):
    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output

    INPUTS
    data      Modal data matrix with dimensions (space,time,nens)
    time      Time axis of model output
    patts     Matrix of spatial patterns (space along rows) to project
    binsize   Odd (symmetric) length of bin in time in which to compute
              SEOFs
    
    OUTPUTS
    EOF_ts    Time series (indexed by space, EOF index, time) of leading
              EOFs
    SV_ts     Time series (indexed by EOF index, time) of leading
              singular values
    t         Time corresponding to EOF_ts and SV_ts

    We throw away padding at the beginning and end so that all
    computations use the same number of times.
    """
    
    pad                = np.floor(binsize/2) #problem here w indent/unindent
    [td,_,_]           = data.shape
    
    t                  = []
    EOF_ts             = []
    SV_ts              = []

    # Define a vector of time indices not in padded regions
    tis = np.arange(pad,td-pad)
    
    for ii in np.arange(tis):
        
        ti = tis[ii]
        
        # Define a time range
        tr             = np.arange(ti-pad,ti+pad+1)
        
        # Select time window
        dat            = data[tr,:,:]
        
        # Compute weights and save
        sU,sS          = mk_seofs(dat,neofs)
        EOF_ts[:,:,ii] = sU
        SV_ts[:,ii]    = sS
        t[ii]          = time(ti)

    return EOF_ts, SV_ts, t


def mk_seofs(dat,neofs):
     """
     Function to compute "snapshot" EOFs from ensemble model output
    
     INPUTS
     data      Modal data matrix with dimensions (time,space,nens)
     neofs     Number of snapshot EOFs to output as a function of time.
    
     OUTPUTS
     sU        Left singular vectors (EOFs) computed over ensemble number
               and time for this particular time bin
     sS        Same as sU, but a vector of singular values
     """
    
    # Change dimensions of dat from (time, space, nens) to (space,time,nens) to allow for reshaping
    dats         = np.transpose(dat,(1,2,0))
    [td,sd,nd]   = dat.shape

    # Reshape so that the second axis is a combination of time and ensemble dimensions.
    # Default is 'C' indexing which should leave the time dimension intact.
    datr         = dats.reshape((td,sd*nd))
    
    # Remove the mean over time and ensembles
    datnm        = datr - datr.mean(axis=1, keepdims=True)
    
    # Compute SVD
    [sUa,sSa,~]  = np.linalg.svd(datnm,full_matrices=False)
    
    # Return desired number of singular values and vectors
    sU           = sUa[:,1:neofs]
    sS           = sSa[1:neofs]

    return sU, sS


def mk_sproj_ts(data,time,patts,binsize):

    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output
    
    INPUTS
    data      Modal data matrix with dimensions (space,time,nens)
    time      Time axis of model output
    patts     Matrix of spatial patterns (space along rows) to project
    binsize   Odd (symmetric) length of bin in time in which to compute
              SEOFs
    
    OUTPUTS
    wts_ts    Time series (indexed by column) of weights corresponding to
              different patterns (indexed by row)
    t         Time for weight time series
    We throw away padding at the beginning and end so that all
    computations use the same number of times.
    """
    
    pad                = np.floor(binsize/2)
    [td,_,_]           = data.shape
    
    t                  = []
    wts_ts             = []

    # Define a vector of time indices not in padded regions
    tis = np.arange(pad,td-pad)
    
    for ii in np.arange(tis):
        
        ti = tis[ii]
        
        # Define a time range
        tr             = np.arange(ti-pad,ti+pad+1)
        
        # Select time window
        dat            = data[tr,:,:]
        
        # Compute weights and save
        wts_ts[][:,ii]   = mk_sproj(dat,patts)
        t[ii]          = time[ti]

    return wts_ts, t

def mk_sproj(dat,patts):

    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output
    
    INPUTS
    data      Modal data matrix with dimensions (space,time,nens)
    patts     Matrix of spatial patterns (space along rows) to project
    
    OUTPUTS
    wts       Vector of SVD-like weights computed for each pattern by
              summing projections of ensemble number and time
    
    Goal is to get a weighting like an EOF. So the procedure is to
    project each pattern onto the data, ending up with a vector (indexed
    by time and nens). Then normalize that vector so that v'v*v = 1 (like
    a PC). The normalization is the weight.
    """
    
    # Change dimensions of dat from (time, space, nens) to (space,time,nens) to allow for reshaping
    dats         = np.transpose(dat,(1,2,0))
    [td,sd,nd]   = dat.shape

    # Reshape so that the second axis is a combination of time and ensemble dimensions.
    # Default is 'C' indexing which should leave the time dimension intact.
    datr         = dats.reshape((td,sd*nd))
    
    # Remove the mean over time and ensembles
    datnm        = datr - datr.mean(axis=1, keepdims=True)

    # Compute projection of each pattern onto all ensemble members
    proj       = datnm.T.dot(patts);
    
    # Compute how much of the variance across ensemble members is accounted for by the pattern
    wts        = np.sqrt(np.sum(proj*proj,0));
 
    return wts