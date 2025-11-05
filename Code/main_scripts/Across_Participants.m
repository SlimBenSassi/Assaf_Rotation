%% 1. INITIALIZATION AND FILE SELECTION

clear; close all; clc
disp('--- Starting Multi-Subject Data Aggregation for GLMM ---');


% --- Configuration ---
DEFAULT_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Data'; % Default folder where your processed data is saved
%example christina data: \\wks3\pr_breska\el-Christina\Backup Copy Christina\PF_Poster\Data\EEG\
% christina eeg trigger list "\\wks3\pr_breska\el-Christina\SxA\SxA_Data\EEGTriggerList.docx"



% 1. Use uigetfile to select multiple subjects' result files
[filenames, filepath] = uigetfile({'*.mat','MATLAB Data File (*.mat)' ;'*.*', 'All files (*.*)'},...
                                    'Select All Subject Alpha Results Files', DEFAULT_PATH, 'MultiSelect', 'on');

if isequal(filenames, 0)
    disp('No files selected. Aborting.');
    return; 
end

if ~iscell(filenames)
    filenames = {filenames};
end

N_SUBJECTS = length(filenames);
disp(['Selected ' num2str(N_SUBJECTS) ' subject files. Proceeding to loop...']);


%% --- Global Variables ---

Fs = SDATA.info.sampling_rate;
data_matrix = SDATA.data; % [Time x Channels]
channel_labels = SDATA.info.channel_labels;
n_channels = size(data_matrix, 2);
events = SDATA.events.triggerChannel; % Raw Status Channel vector
alpha_freq_range = [8, 12]; % Alpha band for filtering (Hz)

% --- Time Variables --- %
pred_window_s = 0.100; % 200ms pre-stimulus prediction window (used for non-time resolved average across window)
PRE_EVENT_SEC = 0.9; %taking all time since warning signal 
POST_EVENT_SEC = 0.3;
total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;
pre_samples = round(PRE_EVENT_SEC * Fs);
pred_start_sample = pre_samples + round(-pred_window_s * Fs); % e.g., sample 410 at 1024Hz
pred_end_sample = pre_samples;

% ---  Event Codes --- %
target_codes = [012, 013, 014, 015, 016, 022, 023, 024, 025, 026, 212, 213, 214, 215, 216, 222, 223, 224, 225, 226]; % (01X means contrast X+1) Targets: Rhythm Target Contrast 4,5,6,7 Right, same but Left (contrasts around threshold) 
%target_codes = [014, 015, 016, 024, 025, 026, 214, 215, 216, 224, 225, 226];
%targer_codes = [014, 015, 016, 024, 025, 026];
report_unseen_code = [231, 241]; % Subjective report code for 'Did Not See' for rhythm 231, for interval 241
report_seen_codes = [232, 233, 234, 242, 243, 244]; % Subjective report
%codes for 'Saw' (Hit) for rhythm 232, for interval 242
%report_seen_codes = [232, 233, 234];

max_target_trials_per_condition = 1000; 

% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};


