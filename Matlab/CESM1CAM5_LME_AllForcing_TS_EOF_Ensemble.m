%% look at mean, EOF of ensemble spread
%%
clc
clear
close all
%% define annual time for plotting later
Time_LME = 850:1849;
%%
LME_850_1850={...
'b.e11.BLMTRC5CN.f19_g16.001.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.002.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.003.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.004.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.005.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.006.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.007.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.008.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.009.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.010.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.011.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.012.cam.h0.TS.085001-184912.nc',...
'b.e11.BLMTRC5CN.f19_g16.013.cam.h0.TS.085001-184912.nc',...
};

TITLES={'ALLr1','ALLr2','ALLr3','ALLr4','ALLr5','ALLr6','ALLr7','ALLr8','ALLr9','ALLr10','ALLr11','ALLr12','ALLr13'}; 

nmodels=size(LME_850_1850,2);
%% define LAT/LON region for analysis once, make lat correction/weighting for later
CESMLME=nc2struct('b.e11.BLMTRC5CN.f19_g16.001.cam.h0.TS.085001-184912.nc');
lat=find(CESMLME.lat.data>= -90 & CESMLME.lat.data <= 90);
lon=find(CESMLME.lon.data>= 0 & CESMLME.lon.data<= 360);
LON=CESMLME.lon.data(lon);
LAT=CESMLME.lat.data(lat);
LAT_Corrections=sqrt(cosd(LAT)); %make lat corrections
LAT_Corrections_Global=repmat(LAT_Corrections,1,size(LON,1));
[nlon,nlat,nt]=size(month2annual_mean(CESMLME.TS.data(lon,lat,:)));
%% pre-fill matrix with NaNs for looping through data, then loop
CESM_TimeSeries_All13=nan(nlon,nlat,nt,nmodels);
% load, save data in for loop
progressbar('CESM LME: Load, filter, save data in for loop')
for i=1:nmodels
    CESMLME=nc2struct(char(LME_850_1850(i)));
    region=CESMLME.TS.data(lon,lat,:);
    region=month2annual_mean(region); %monthly to annual data
    region=RemoveMean3D(region); %Remove long-term mean
    % save at end of for loop
    CESM_TimeSeries_All13(:,:,:,i)=region; %last dimension will be ensemble member
% progressbar at end of for loop
progressbar(i/nmodels)
end
%% now remove the ensemble mean
nens = nmodels; %how many ens members for taking mean?

%first, average together ensemble members
CESM_Mean_ALL=nanmean(CESM_TimeSeries_All13(:,:,:,1:nens),4);
%remove ens mean
CESM_TimeSeries_All13_RemEnsMean = CESM_TimeSeries_All13 - CESM_Mean_ALL;
%lat weight
CESM_TimeSeries_All13_RemEnsMean_Weighted = CESM_TimeSeries_All13_RemEnsMean.*LAT_Corrections_Global';
%% run Dan's code

data = reshape(CESM_TimeSeries_All13_RemEnsMean_Weighted,nlon*nlat,nt,nens);
time = 1:1000;
neofs = 12;
binsize = 5;

[EOF_ts,SV_ts,t] = mk_seofs_ts(data,time,neofs,binsize);
%[EOF_ts,SV_ts,t] = mk_seofs_ts(data,time,neofs,binsize)
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
%%
figure()

numplots = 9;
for i = 1:numplots
    subplot(numplots,1,i)
    plot(t,SV_ts(i,:))
    hold on
end

    %%
patts = EOF_ts(:,1,1);
[wts_ts,t] = mk_sproj_ts(data,time,patts,binsize);
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
%%