%% TIME-FREQUENCY LOGISTIC REGRESSION ANALYSIS


%% 1. INITIALIZATION AND LOAD MASTER TABLE

%clear; close all; clc
disp('--- Piloting GLMM Prediction Model ---');

%--- Configuration ---
DEFAULT_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Data'; % Default folder where your processed data is saved
%example christina data: \\wks3\pr_breska\el-Christina\Backup Copy Christina\PF_Poster\Data\EEG\
% christina eeg trigger list "\\wks3\pr_breska\el-Christina\SxA\SxA_Data\EEGTriggerList.docx"

% --- 1. Use UIGETFILE for Interactive Selection (GUI Dialog) ---
[filename, filepath] = uigetfile({'*.mat','MATLAB Data File (*.mat)' ;'*.*', 'All files (*.*)'},...
                                    'Select Clean Preprocessed SDATA File', DEFAULT_PATH);

if isequal(filename, 0)
    disp('No file selected. Aborting script.');
    return; 
end

master_table_file = fullfile(filepath, filename);


% Load the MasterTable (contains SubjectID, AlphaPower, StimIntensity, SubjectiveOutcome)
load(master_table_file, 'MasterTable');
disp('Master Table loaded successfully.');

% Ensure data types are correct for the GLMM function
MasterTable.SubjectID = categorical(MasterTable.SubjectID);
MasterTable.SubjectiveOutcome = logical(MasterTable.SubjectiveOutcome); 

% Display total N and the first few rows
disp(['Total trials in model: ' num2str(size(MasterTable, 1))]);
head(MasterTable);

%copy in case I need it quickly instead of reloading dataset file
MasterTable_copy = MasterTable;



%% GLOBAL VARIABLES

DO_BASELINE_CORRECTION = true;
Fs = 1024; % Assuming Fs is 1024 Hz
n_channels = 71; %change if needed
PRE_EVENT_SEC = 0.5; % Assumed pre-stimulus window
N_TIME_BINS = 1;
N_FREQ_BINS = 1;
alpha_freq_range = [8 12];
time_window_sec = [-0.020 0];


% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 28, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};


%% BASELINE_CORRECTION

if DO_BASELINE_CORRECTION
    for i = 1:height(MasterTable)
        MasterTable.AlphaAmplitude{i} = baseline_correction(MasterTable.AlphaAmplitude{i}, Fs, PRE_EVENT_SEC);
    end
    disp('All single-trial Alpha features are now baseline-normalized (dB).');
end

%% AVERAGE DATA OF ROI

% 1. Average across the Channel Dimension (Dimension 3)
% Slices the power envelope to the current ROI channels and averages their values.
for i = 1:height(MasterTable)
        MasterTable.AlphaAmplitude{i} = squeeze(mean(MasterTable.AlphaAmplitude{i}(:, :, currentROI), 3));
end

%% DIFFERENT DATASETS    

% Convert the categorical column to its underlying numeric codes (1, 2, 3)
condition_codes = double(MasterTable.Condition);

% Now use the numerical codes for the comparison logic

% 1. MasterTable_Predictive (Condition < 3: Keep Rhythm and Interval)
MasterTable_Predictive = MasterTable(condition_codes < 3, :);

% 2. MasterTable_Irregular (Condition == 3: Keep Irregular)
MasterTable_Irregular = MasterTable(condition_codes == 3, :);

%% TIME-FREQUENCY REGRESSION MAP

tic
[BetaMap, PValueMap, TimeBins, FreqBins] = tf_regression_map(MasterTable_Predictive, Fs, time_window_sec, alpha_freq_range, N_TIME_BINS, N_FREQ_BINS);
toc


%%

