# ens_snapshot_tools.py
# D Amrhein, November 2020

import numpy as np

def mk_seofs_ts(data,time,neofs,binsize):
    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output

    INPUTS
    data      Modal data matrix with dimensions (time,space,nens)
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
    
    pad                = int(np.floor(binsize/2))
    [td,sd,_]           = data.shape

    # Define a vector of time indices not in padded regions
    tis = np.arange(pad,td-pad)
    ltis               = len(tis) 

    t                  = np.empty(ltis)
    EOF_ts             = np.empty([ltis,sd,neofs])
    SV_ts              = np.empty([ltis,neofs])

    for ii,ti in enumerate(tis):

        # Define a time range
        tr             = np.arange(ti-pad,ti+pad+1)

        # Select time window
        dat            = data[tr,:,:]

        # Compute weights and save
        sU,sS          = mk_seofs(dat,neofs)
        EOF_ts[ii,:,:] = sU
        SV_ts[ii,:]   = sS
        t[ii]         = time[ti]
        
    return EOF_ts, SV_ts, t


def mk_seofs(dat,neofs):
    """
    Function to compute "snapshot" EOFs from ensemble model output
   
    INPUTS
    dat       Modal data matrix with dimensions (time,space,nens)
    neofs     Number of snapshot EOFs to output as a function of time.
   
    OUTPUTS
    sU        Left singular vectors (EOFs) computed over ensemble number
             and time for this particular time bin
    sS        Same as sU, but a vector of singular values
    """

    # Change dimensions of dat from (time, space, nens) to (space,time,nens) to allow for reshaping
    dats         = np.transpose(dat,(1,0,2))
    [sd,td,nd]   = dats.shape

    # Reshape so that the second axis is a combination of time and ensemble dimensions.
    # Default is 'C' indexing which should leave the time dimension intact.
    datr         = dats.reshape((sd,td*nd))

    # Remove the mean over time and ensembles
    datnm        = datr - datr.mean(axis=1, keepdims=True)

    # Compute SVD
    [sUa,sSa,_]  = np.linalg.svd(datnm,full_matrices=False)

    # Return desired number of singular values and vectors
    sU           = sUa[:,:neofs]
    sS           = sSa[:neofs]

    return sU, sS


def mk_sproj_ts(data,time,patts,binsize):

    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output

    INPUTS
    data      Modal data matrix with dimensions (time,space,nens)
    time      Time axis of model output
    patts     Matrix of spatial patterns (space in columns) to project
    binsize   Odd (symmetric) length of bin in time in which to compute
              SEOFs

    OUTPUTS
    wts_ts    Time series (indexed by column) of weights corresponding to
              different patterns (indexed by row)
    t         Time for weight time series
    We throw away padding at the beginning and end so that all
    computations use the same number of times.
    """

    pad                = int(np.floor(binsize/2))
    [td,sd,_]          = data.shape
    [_,pd]             = patts.shape


    # Define a vector of time indices not in padded regions
    tis                = np.arange(pad,td-pad)
    ltis               = len(tis) 
    t                  = np.empty(ltis)
    wts_ts             = np.empty([ltis,pd])

    for ii,ti in enumerate(tis):

        # Define a time range
        tr             = np.arange(ti-pad,ti+pad+1)

        # Select time window
        dat            = data[tr,:,:]

        # Compute weights and save
        wts_ts[ii,:]   = mk_sproj(dat,patts)
        t[ii]          = time[ti]

    return wts_ts, t

def mk_sproj(dat,patts):

    """
    Function to compute "snapshot" projections of patterns onto ensemble
    model output

    INPUTS
    data      Modal data matrix with dimensions (time,space,nens)
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
    dats         = np.transpose(dat,(1,0,2))
    [sd,td,nd]   = dats.shape

    # Reshape so that the second axis is a combination of time and ensemble dimensions.
    # Default is 'C' indexing which should leave the time dimension intact.
    datr         = dats.reshape((sd,td*nd))

    # Remove the mean over time and ensembles
    datnm        = datr - datr.mean(axis=1, keepdims=True)
    
    # Compute projection of each pattern onto all ensemble members
    proj       = datnm.T.dot(patts);
    


    # Compute how much of the variance across ensemble members is accounted for by the pattern
    wts        = np.sqrt(np.sum(proj*proj,0));

    return wts

def mk_covs(data,time,binsize):
    
    """
    Constructs a time series of covariance matrices from a data set with time, space, and ensemble dimensions.
    In practice, "data" must have a small spatial dimension because we are explicitly storing covariance matrices.
    As such, for large model output it is recommended that data be first projected into a reduced basis.
    
    INPUTS
    data      Modal data matrix with dimensions (time,space,nens)
    time      Time axis of model output
    binsize   Odd (symmetric) length of bin in time in which to compute
              SEOFs

    OUTPUTS
    C         3d matrix of time-evolving spatial covariances computed over ensemble space and, for binsize>1, a moving time window
    t         Time for covariance matrix time series
  
    """

    pad                = int(np.floor(binsize/2))
    [td,sd,_]          = data.shape

    # Define a vector of time indices not in padded regions. These will be our bin centers.
    tis                = np.arange(pad,td-pad)
    ltis               = len(tis) 
    
    t                  = np.empty(ltis)
    C                  = np.empty([ltis,sd,sd])

    for ii,ti in enumerate(tis):
        
        # Define a time range
        tr             = np.arange(ti-pad,ti+pad+1)

        # Select time window
        dat            = data[tr,:,:]
        
        # Change dimensions of dat from (time, space, nens) to (space,time,nens) to allow for reshaping
        dats         = np.transpose(dat,(1,0,2))
        [sd,td,nd]   = dats.shape

        # Reshape so that the second axis is a combination of time and ensemble dimensions.
        # Default is 'C' indexing which should leave the time dimension intact.
        datr         = dats.reshape((sd,td*nd))

        # Remove the mean over time and ensembles
        datnm        = datr - datr.mean(axis=1, keepdims=True)
        
        #import pdb
        #pdb.set_trace()
        
        # Save the covariance matrix. Warning -- only do this with reduced space!
        C[ii,:,:] = 1/(td*nd-1)*datnm.dot(datnm.T)
        
        # Corresponding time
        t[ii]          = time[ti]

    # Remove the time mean covariance
    # Cnm        = C - C.mean(axis=1, keepdims=True)

    # Compute dominant changes to covariance
    # [uC,sC,vC] = np.linalg.svd(Cnm, full_matrices=False)

    return C,t

def reduce_space(data,nEOF):
    
    """
    Projects a field (time, space, nens) onto its nEOF leading EOFs.
    
    INPUTS
    data      Modal data matrix with dimensions (space,time,nens)
    nEOF      Number of EOFs retained

    OUTPUTS
    rbdor     Reduced-space (time, eof index, nens) data matrix
    ur        EOFs used to project reduced-space back into full state
    s         Full vector of singular values
  
    """

    # Change dimensions of dat from (time, space, nens) to (space,time,nens) to allow for reshaping
    dats         = np.transpose(data,(1,0,2))
    [sd,td,nd]   = dats.shape

    # Reshape so that the second axis is a combination of time and ensemble dimensions.
    # Default is 'C' indexing which will leave the time dimension intact.
    datr         = dats.reshape((sd,td*nd))

    # Compute EOFs as a reduced basis
    [u,s,vt] = np.linalg.svd(datr,full_matrices=False)

    # This is the output in the reduced basis. Keep nEOF
    # rbd          = (vt[:nEOF,:]*s[:nEOF,None]).T

    # This is the output in the reduced basis. Keep nEOF
    rbd          = (vt[:nEOF,:]*s[:nEOF,None])

    # Reshape into original dimensions
    rbdo         = rbd.reshape(nEOF,td,nd)

    # Reorder dimensions like the original
    rbdor = np.transpose(rbdo,(1,0,2))

    # These are the columns of u that are useful
    ur           = u[:,:nEOF]
    
    return rbdor, ur, s

def KLdiv(C0,C1,m0,m1):
    """
    Computes Kullback-Leibler divergence between two (full-rank) Gaussian processes with sample
    covariances matrices C0 and C1 and sample means m0 and m1. C0 and C1 must be full rank.
    See https://en.wikipedia.org/wiki/Kullback–Leibler_divergence

    INPUTS
    C0    (space, space) First covariance matrix
    C1    (space, space) Second covariance matrix
    m0    (space) First mean vector
    m1    (space) Second mean vector

    OUTPUTS
    kld   Kullback-Leibler divergence
 
    """
    C1i = np.linalg.inv(C1)
    kld = 1/2 * ( np.trace(C1i.dot(C0)) + (m0-m1).T.dot(C1i).dot(m0-m1) + np.log(np.linalg.det(C1)/np.linalg.det(C1)) )

    return kld
