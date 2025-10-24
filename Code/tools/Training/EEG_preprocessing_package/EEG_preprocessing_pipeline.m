
%% 1. initialize 

clear; close all; clc

raw_data_path = "\\wks3\pr_breska\el-Slim\random assaf data (controls)";  % The file selection dialog will open on this folder

% necessary functions (in matlab path):
% from fieldtrip: read_biosemi_bdf, read_24bit (plus all the read_24bit mex files), ReadBioSemiTriggerChannel
% from EEGlab (should have full toolbox as folder): runica, topoplot (plus '.locs' file, e.g. head71.locs)
% Custom: HPF, LPF, remove_line_noise, multichanplot, rdi, get_neighbor_electrodes



%% 2. choose data file, load and append
clc

if ~exist('raw_data_path','var')
    raw_data_path=cd;
end

% select files of a single subject
[data_file_list, ~] = uigetfile({'*.bdf','BDF file (*.bdf)' ;'*.*', 'All files (*.*)'},'Select raw file/s',raw_data_path,'multiselect','on');
if isscalar(data_file_list)
    % No file was selected. 
else
    if ~iscell(data_file_list)        
        data_file_list = {data_file_list}; % Single file was selected - put it in a 1x1 cell array
    end
end
nSessions = length(data_file_list);

% create data structure
SDATA = struct;
SDATA.info = struct;
SDATA.metadata = struct;
SDATA.events = struct;

EEGdata=[];
% run over sessions, incase multiple data files were selected
disp('reading data files')
tic
for sessIdx = 1:nSessions
    
    fileName = fullfile(raw_data_path,data_file_list{sessIdx});
    hdr = read_biosemi_bdf(fileName);
    session_data = read_biosemi_bdf(fileName,hdr, 1,hdr.nSamples,1:hdr.nChans);
    session_data = single(session_data)';
    EEGdata=[EEGdata; session_data];
    toc
    
end
SDATA.data=EEGdata;

% log info
SDATA.info.sampling_rate = hdr.Fs;
SDATA.info.channel_labels = hdr.label;

% add channel numbers
for chan=1:length(SDATA.info.channel_labels)
    SDATA.info.channel_labels{chan}=[SDATA.info.channel_labels{chan} ' (' num2str(chan) ')'];
end

%get status channel info
statusChans = ReadBioSemiTriggerChannel(SDATA.data(:,end));
SDATA.events.triggerChannel=statusChans.Triggers;
SDATA.events.blockStartRecording=statusChans.StartEpoch;
SDATA.metadata.statusChan=SDATA.data(:,end);
SDATA.data=SDATA.data(:,1:end-1);
SDATA.info.channel_labels=SDATA.info.channel_labels(1:end-1);
SDATA.info.fileNames=data_file_list;

removeEXG8=input('remove EGX8? (1=yes): '); % external electrode that is often not used - inspect before removing!
if removeEXG8
    SDATA.data=SDATA.data(:,1:end-1);
    SDATA.info.channel_labels=SDATA.info.channel_labels(1:end-1);
end
    

SDATA.metadata.bad_electrodes=[];
SDATA.metadata.good_electrodes=1:size(SDATA.data,2);
SDATA.metadata.analysisStageDone=2;

clear EEGdata session_data    



%% 3. Mark bad electrodes

%% 3.1: inspect data
hpf_cutoff_for_disp=0.1;
show_data = HPF(double(SDATA.data),SDATA.info.sampling_rate,hpf_cutoff_for_disp,3); % for visualization only

multichanplot(show_data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);

clear show_data


%% 3.2: mark bad electrodes

clc
bad_elec=input('input numbers of bad electrodes (in [], separated by commas): ');
SDATA.metadata.bad_electrodes=unique([SDATA.metadata.bad_electrodes, bad_elec]);

SDATA.metadata.good_electrodes=setdiff(SDATA.metadata.good_electrodes, SDATA.metadata.bad_electrodes);
SDATA.metadata.analysisStageDone=3;



%% 4. reference
reference_channels=[69,70]; %linked mastoids
% reference_channels=[71]; %nose
clc

ref_chans = reference_channels(~ismember(reference_channels,SDATA.metadata.bad_electrodes));
if isempty(ref_chans)
    disp('cannot reference - requested reference channels were marked as bad electrodes')
else
    SDATA.data = bsxfun(@minus,SDATA.data,mean(SDATA.data(:,ref_chans),2));
end

if length(ref_chans)<length(reference_channels)
    disp('some of the reference channels were marked as bad electrodes')
end

SDATA.metadata.reference=ref_chans;
disp('reference done')
SDATA.metadata.analysisStageDone=4;



%% 5. detrend blocks and concatenate
edgeWinForBlockEnd = 10;

% divide data into blocks
block_start=[find(SDATA.events.blockStartRecording)' size(SDATA.data,1)+1];
num_blocks = length(block_start)-1;
data_block = cell(num_blocks,1);
for block = 1:num_blocks
    data_block{block} = SDATA.data(block_start(block):block_start(block+1)-1,:);
end

% deternd blocks, using smoothed end point (instead of just the last sample)
for block = 1:num_blocks
    [Nts, Nch] = size(data_block{block});
    for ch = 1:Nch
        trend = linspace(mean(data_block{block}(1:edgeWinForBlockEnd,ch)),mean(data_block{block}((end-edgeWinForBlockEnd+1):end,ch)),Nts)';
        data_block{block}(:,ch) = data_block{block}(:,ch) - trend;
    end
end

% concatenate blocks
temp_data = [];
for block = 1:num_blocks
    temp_data = [temp_data ; data_block{block}];
end
SDATA.data = temp_data;
clear temp_data data_block trend
disp('detrend done')
SDATA.metadata.analysisStageDone=5;



%% 6. filter: high-pass + line (for electrical frequency in the current country)
hpf_freq = 0.1;
filt_deg = 3;
line_noise_freq=60;
LNFwindow=10;

tic; SDATA.data = single(HPF(double(SDATA.data),SDATA.info.sampling_rate,hpf_freq,filt_deg)); toc
tic; SDATA.data = remove_line_noise(SDATA.data,line_noise_freq,SDATA.info.sampling_rate,LNFwindow); toc % A larger window makes for a much smaller "notch filter" effect.

multichanplot(SDATA.data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);

dataBeforeICA=SDATA.data;
disp('filter done')
SDATA.metadata.analysisStageDone=6;



%% 7. ICA

%% 7.1: prepare for ICA
verSupChanName='EXG4 (68)';
verInfChanName='EXG3 (67)';
horRightChanName='EXG2 (66)';
horLeftChanName='EXG1 (65)';

clc

manual_select_train_set=1;
train_set_range_secs=[10 130];

% create bipolar EOG channels
VEOG = SDATA.data(:,strcmp(SDATA.info.channel_labels,verSupChanName))-SDATA.data(:,strcmp(SDATA.info.channel_labels,verInfChanName));
HEOG = SDATA.data(:,strcmp(SDATA.info.channel_labels,horRightChanName))-SDATA.data(:,strcmp(SDATA.info.channel_labels,horLeftChanName));

% remove bad electrodes to not contaminate all components
dataForICA=SDATA.data(:,SDATA.metadata.good_electrodes);
clean_channel_labels=SDATA.info.channel_labels(SDATA.metadata.good_electrodes);

% define training set using multichanplot function
if ~manual_select_train_set
    ica_train_set=dataForICA(train_set_range_secs(1)*SDATA.info.sampling_rate:train_set_range_secs(2)*SDATA.info.sampling_rate,:);
else    
    clean_channels_labels_with_bipolar=[{'VEOG'}; {'HEOG'}; clean_channel_labels];
    show_data = [VEOG HEOG dataForICA];
    
    train_set_idx=multichanplot(show_data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', clean_channels_labels_with_bipolar, 'ylim', [-100 100]);
    ica_train_set=dataForICA(train_set_idx,:);
    
end
disp(['length of ICA train data is ' num2str(size(ica_train_set,1)/SDATA.info.sampling_rate) ' seconds'])

clear show_data


%% 7.2: run ICA
    
tic
[weights,sphere,~,~,~,~,~]  = runica(ica_train_set','pca',size(ica_train_set,2)-1);
toc

unmix = weights*sphere;   % comps x chans
mix = pinv(unmix);

if ~isreal(mix)
    error('Warning: bad mixing matrix (complex values). Try reducing training data dimensions');
end

SDATA.ica.unmix = unmix;
SDATA.ica.mix = mix;

C = (SDATA.ica.unmix * dataForICA')';

cCorr=corr([C VEOG HEOG]);
figure;
subplot(2,1,1); plot(cCorr(end-1,1:end-2)); title('component correlations with VEOG')
subplot(2,1,2); plot(cCorr(end,1:end-2)); title('component correlations with HEOG')


%% 7.3: inspect ICA component time course
% C = (SDATA.ica.unmix * dataForICA')';

clc

component_labels_with_bipolar=cell(size(C,2),1);
for comp=1:length(component_labels_with_bipolar)
    component_labels_with_bipolar{comp}=num2str(comp);
end

component_labels_with_bipolar=[component_labels_with_bipolar; {'VEOG'}; {'HEOG'}];

scalingFactorV=std(VEOG)/mean(std(C));
scalingFactorH=std(HEOG)/mean(std(C));
show_C = [C 2*VEOG/scalingFactorV 2*HEOG/scalingFactorH];
clear C

multichanplot(show_C, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', component_labels_with_bipolar);

clear show_C
clc


%% 7.4: inspect ICA component topography
clc
h=figure('Position', [500 50 500 500]); 

done=0;
while ~done
    
    clc
    compToPlot=input('input number of component to plot: ');
        
    compTopo=zeros(length(SDATA.info.channel_labels),1);
    compTopo(SDATA.metadata.good_electrodes)=SDATA.ica.mix(:,compToPlot);
    compTopo(SDATA.metadata.bad_electrodes)=NaN;
    
    topoplot(compTopo,'head71.locs','electrodes','on','style','map','shading','interp'); 
    figure(h.Number);

    userInp=input('done inspecting? (1=yes): ');
    if userInp==1
        done=1;
    end
    
end


%% 7.5: remove components and reconstruct data
clc
remove_comp=input('input numbers of components to remove (in [], separated by commas): ');

Ncomps = size(SDATA.ica.mix,2);
cmp = true(Ncomps,1);
cmp(remove_comp) = false;
reconstructedData = ( SDATA.ica.mix * diag(cmp) * SDATA.ica.unmix * dataForICA' )';


%% 7.6: Inspect reconstruction
clc

show_data = [VEOG HEOG dataForICA];
multichanplot(show_data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', clean_channels_labels_with_bipolar, 'ylim', [-100 100]);
title('Original data')
show_data = [VEOG HEOG reconstructedData];
multichanplot(show_data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', clean_channels_labels_with_bipolar, 'ylim', [-100 100]);
title('Reconstructed data')
clear show_data


%% 7.7: approve reconstruction

acceptReconst=input('Accept reconstruction? (1=yes): ');
if acceptReconst==1
    SDATA.data(:,SDATA.metadata.good_electrodes) = single(reconstructedData);
end

clear reconstructedData dataForICA
SDATA.metadata.analysisStageDone=7;



%% 8. semi-automatic artifact rejection %%

%% 8.1: automatic artifact detection
lpf_cutoff_for_rdi = 100;

abs_amp_threshold = 100;
low_act_allowed = 0.5;
low_act_interval = 100;
minmax_allowed = 100;
minmax_interval=200;
art_margin = 200; % total width of time window (in msec) around artifact time point that would be marked as bad

clc

art_margin_samples=round(0.5*art_margin*SDATA.info.sampling_rate/1000); % convert artifact margin to samples

channelsForRDI=SDATA.metadata.good_electrodes(SDATA.metadata.good_electrodes<65); % run only on scalp channels
lpf_data_for_RDI = LPF(double(SDATA.data),SDATA.info.sampling_rate,lpf_cutoff_for_rdi,3); % applied on data for artifact search, not to actual data

disp('Running artifact detection')
tic; amp_artMat = rdi(lpf_data_for_RDI,'channels',channelsForRDI,'m',[minmax_allowed minmax_interval art_margin_samples art_margin_samples], 'e',[-abs_amp_threshold abs_amp_threshold art_margin_samples art_margin_samples], 'l',[low_act_allowed low_act_interval art_margin_samples art_margin_samples]); toc

amp_art = any(amp_artMat,2);
figure; plot(1:size(amp_artMat,2), 100*mean(amp_artMat)); title(['% data rejected per electrode (' num2str(100*mean(amp_art), 3) '% rejected overall)']);
ylim([0 max(100*mean(amp_artMat))*1.2])

clear lpf_data_for_RDI


%% 8.2: inspect artifacts and modify marking

% add bipolar channels for to see where EOG artifact were and whether they were fully removed
data_rdi_withBipolar = [SDATA.data(:,channelsForRDI) VEOG HEOG];
amp_artMat_withBiploar = [amp_artMat(:,channelsForRDI) zeros(size(amp_artMat,1),2)];
channel_labels_RDI_withBiploar=[SDATA.info.channel_labels(channelsForRDI); {'VEOG'}; {'HEOG'}];

amp_art_inspected = multichanplot(data_rdi_withBipolar,10,'srate',SDATA.info.sampling_rate,'markdata',amp_art,'channelnames',channel_labels_RDI_withBiploar, 'ylim', [-100 100], 'markIndivChan', amp_artMat_withBiploar);

amp_artMat_clean=amp_artMat.*repmat(amp_art_inspected,1, size(amp_artMat,2));
figure; plot(1:size(amp_artMat_clean,2), 100*mean(amp_artMat_clean)); title('percent data rejected per electrode'); ylim([0 max(100*mean(amp_artMat))*1.2])

disp(['Percent data rejected: ' num2str(100*mean(amp_art_inspected)) '%'])


%% 8.3: accept artifact marking
clc
acceptInspection=input('Accept artifact marking? (1=yes): ');
if acceptInspection==1
    SDATA.metadata.artifacts = amp_art_inspected;
end
SDATA.metadata.analysisStageDone=8;



%% 9. Interpolate bad electrodes

%% 9.1: interpolate bad electrodes
dataClean=SDATA.data;

if ~isempty(SDATA.metadata.bad_electrodes)
    for e = SDATA.metadata.bad_electrodes
        neighbors = get_neighbor_electrodes(e);
        neighbors(ismember(neighbors,SDATA.metadata.bad_electrodes)) = [];
        dataClean(:,e) = nanmean(dataClean(:,neighbors),2);
    end
    
    multichanplot(SDATA.data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);
    multichanplot(dataClean, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);

end

%% 9.2: interpolate additional electrodes (if necessary)
close all
clc

interpolateElec=input('input numbers of electrodes to interpolate (in [], separated by commas): ');
interpolateElec=[interpolateElec, find(isnan(dataClean(1,:)))];
if ~isempty(interpolateElec)
    for e = interpolateElec
        neighbors = get_neighbor_electrodes(e);
        neighbors(ismember(neighbors,SDATA.metadata.bad_electrodes)) = [];
        dataClean(:,e) = nanmean(dataClean(:,neighbors),2);
    end
    
    multichanplot(SDATA.data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);
    multichanplot(dataClean, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);
end

%% 9.3: accept interpolation
clc
acceptInspection=input('Accept interpolation? (1=yes): ');
if acceptInspection==1
    SDATA.data = dataClean;
end
SDATA.metadata.analysisStageDone=9;


%% 10. save SData
clc
fileNameSave=[SDATA.info.fileNames{1}(1:end-4) '_pp.mat'];
disp('Saving')
save(fileNameSave, 'SDATA')

disp(['done saving s' num2str(subject_ids)])

%%





