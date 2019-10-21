%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      Extract the 11 x 10 area from the 80 x 80 data frame    %                                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: J. J. Zhang
%%% Function:
%%% read in 80 x 80 pixels data frame
%%% do the energy calibration
%%% extract the 11 x 10 pixels area at the center
%%% save the 110-pixel data (binary file, ped-subed, reordered, energy calibrated)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars
close all
clc

iSave=1; %enable saving operations
iPlot=0;%enable plotting operations
file_start = 161;
file_end = 166;

folderName_cali = 'E:\CSC\CoAm02\CoAm_500fps_30minrepeat_002Results_det\';
folderName='E:\CSC\Ba\';
common_name='Ba_340fps_30min';

no_of_bins = 1000;
range = [0 600];%range in energies, not ADU!
dx = range(2)/no_of_bins;
bins = dx/2:dx:range(2)-dx/2;%equal bins 
ff = 8000/no_of_bins;%multiplicative factor between bin and ADU because 1 bin=8ADU

%read bad pixel map
fileID = fopen([folderName_cali,'BadPixels.txt'],'r');
mask= fscanf(fileID,'%f');
fclose(fileID);
indBad=find(mask==0);

%read in gain and offset information
fileID = fopen([folderName_cali,'Gain.txt'],'r');
gain= fscanf(fileID,'%f');
fclose(fileID);
gain=reshape(gain,[80,80]);%multiply by 4 because 1 bin=8ADU

fileID = fopen([folderName_cali,'Offset.txt'],'r');
offset= fscanf(fileID,'%f');
fclose(fileID);
offset=reshape(offset,[80,80]);

fileID = fopen([folderName_cali,'Threshold_1.txt'],'r');
mask= fscanf(fileID,'%f');
mask = reshape(mask,[80,80]);
fclose(fileID);

nn=0; mm=0;
dataIn=zeros(1,6400);

%%   select 4*4 pixels area on the whole detector surface(80,80), center energy fall in (Threshold_low, Threshold_high)
select = zeros(10,10);

for rr = file_start:file_end
       % read the files 
       disp(['file#',num2str(rr)]);
       filename_raw=['E:\CSC\Ba\Ba_340fps_30minrepeat_00',num2str(rr-1),'\Ba_340fps_30min_ped_sub.bin'];
       fid = fopen([filename_raw]);
       
       if iSave
           filename_ordered_ped_sub = ['F:\CSC_porcessing\Ba_100\Ba_340fps_30minrepeat_00',num2str(rr-1),'_encali.bin'];
           fid2= fopen(filename_ordered_ped_sub, 'w');
       end
       
       frame = zeros(80,80);  %create empty frame
       tic;
       
       while ~feof(fid)
           % Sort Data into 240x80 pixels with correct orientation
           dataIn = fread(fid,6400,'double');
           
           if(~isempty(dataIn))
               
                frame=reshape(dataIn, 80, 80);   
                frame=frame-offset;%apply energy calibration
                frame=frame./gain; 
                dataIn_calibrated=frame(:);%delete bad pixels
                dataIn_calibrated(indBad)=0;%set to 0 the bad pixels

                indices1=find(dataIn_calibrated>=range(2) | dataIn_calibrated<0); %eliminate the spikes
                dataIn_calibrated(indices1) = 0; %if not valid set to 0

                frame = reshape(dataIn_calibrated,[80,80]);
                select = frame(36:45,35:45);
                
                if iSave
                        fwrite(fid2, select(:),'double');
                end
                  
            end
        end
        
        fclose(fid);
        delete([filename_raw]);
        if iSave
            fclose(fid2);    %close current files   
        end
end


