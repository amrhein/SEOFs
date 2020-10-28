function [RegionalMean_LatWeighted]=LatWeightRegionalMean(X3D,LON,LAT);
% function calculates cosine of latitude, weights global/regional mean by
% the cosine of latitude (divide by weighting sums)
% written by L Parsons, 2020/01/14
[nlon,nlat,nt]=size(X3D);

LAT_Corrections=sqrt(cosd(LAT)); %make lat corrections
LAT_Corrections_Global=repmat(LAT_Corrections,1,size(LON,1));
%LAT_Corrections_Global_Sum=nansum(LAT_Corrections_Global,2);

RegionalMean_LatWeighted=nan(1,nt);

    for t=1:nt %loop through each time step, find where valid data for weighting/sums
        DataLocations=(~isnan(squeeze(X3D(:,:,t))))*1;
        LAT_Corrections_Global_Sum=nansum(nansum(LAT_Corrections_Global.*DataLocations')); %make NaN points zero
        Data_Annual_OneYear_LatCorrected=X3D(:,:,t).*LAT_Corrections_Global';
        Annual_OneYear_Reshaped_Sum_LatCorrected=nansum(reshape(Data_Annual_OneYear_LatCorrected,nlon*nlat,1));
        Annual_OneYear_Reshaped_Sum_DivWeight=Annual_OneYear_Reshaped_Sum_LatCorrected/LAT_Corrections_Global_Sum;
        RegionalMean_LatWeighted(t)=Annual_OneYear_Reshaped_Sum_DivWeight;
    end

end