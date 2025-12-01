%% Extract rest of features (follows logic of the script ACROSS_PARTICIPANTS)



% ---------------------------------------------------------------
% MERGE baseline/orientation features into big EEG dataset
%
% Requirements:
%   • main_data.mat           → contains all_alpha_power_raw, all_subject_ids, ...
%   • baseline_features.mat   → contains B_baseline_raw, B_isi, B_orientation,
%                               B_orientation_reseponse, B_subject_ids
% ---------------------------------------------------------------

clear; clc;

%% ============================
%       LOAD DATASETS
% =============================
%main_file = 'main_data.mat';       % <-- YOUR MAIN EEG FILE
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


disp('Loading main EEG dataset...');
load(master_table_file, 'MasterTable');        % loads all_alpha_power_raw, all_subject_ids, etc.


%% --- Global Variables ---

alpha_freq_range = [13, 40]; % Alpha band for filtering (Hz)

% --- Time Variables --- %
Fs = 1024; %change if needed
n_channels = 71; %change if needed
pred_window_s = 0.200; % 200ms pre-stimulus prediction window (used for non-time resolved average across window)
PRE_EVENT_SEC = 0.500; %taking all time since warning signal 
POST_EVENT_SEC = 0.1;
total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;
pre_samples = round(PRE_EVENT_SEC * Fs);
pred_start_sample = pre_samples + round(-pred_window_s * Fs); % e.g., sample 410 at 1024Hz
pred_end_sample = pre_samples;

% ---  Event Codes --- %
%target_codes = [011, 012, 013, 014, 015, 016, 017, 021, 022, 023, 024, 025, 026, 027, 211, 212, 213, 214, 215, 216, 217, 221, 222, 223, 224, 225, 226, 227];  % (01X means contrast X+1) Targets: Rhythm Target Contrast 4,5,6,7 Right, same but Left (contrasts around threshold) 
%target_codes = [014, 015, 016, 024, 025, 026, 214, 215, 216, 224, 225, 226];
%target_codes = [014, 015, 016, 024, 025, 026];
target_codes = [010:029 210:229]; %all predictive
report_unseen_code = [231, 241]; % Subjective report code for 'Did Not See' for rhythm 231, for interval 241
report_seen_codes = [232, 233, 234, 242, 243, 244]; % Subjective report
warning_codes_of_interest = [071 072 073]; %for irregular
all_warning_codes = [071, 072, 073];
%report_unseen_code = [231, 241];
%report_seen_codes = [23:253];
%codes for 'Saw' (Hit) for rhythm 232, for interval 242
%report_seen_codes = [232, 233, 234];

max_target_trials_per_condition = 1000; 

% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 28, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};


%% ===================================================
%      Allocate NEW merged variables
% ====================================================
MasterTable = MasterTable(MasterTable.SubjectID == categorical("102"), :);

all_subject_ids = MasterTable.SubjectID;
all_baseline_raw              = cell(size(all_subject_ids));   % cell of baseline windows
all_isi_from_baseline         = zeros(size(all_subject_ids));
all_orientation               = zeros(size(all_subject_ids));
all_orientation_reseponse     = zeros(size(all_subject_ids));


%% ===================================================
%  3. Loop over subjects to compute baseline features
% ===================================================

% all_subject_ids is a categorical column
subjects = cellstr(unique(all_subject_ids));

disp(['Processing ' num2str(length(subjects)) ' subjects for baseline features...']);

%for s = 13:length(subjects)
for s = 1:1
    %subjects = categories(all_subject_ids);  % now a cell array of strings
    subj = subjects{s};
    disp(['Processing subject ' subj ' (' num2str(s) ' of ' num2str(length(subjects)) ')']);
    
    % --- 3A. Find indices of all trials for this subject in main dataset
    idx_main = find(all_subject_ids == subj);
    n_trials_main = length(idx_main);
    
    % --- 3B. Load subject EEG structure
    current_filename = ['EEG_SxA_Subj' subj '_Session2_pp.mat']; % adjust if needed
    full_external_path = fullfile('\\wks3\pr_breska\el-Christina\SxA\SxA_Data\EEG Preprocessed', [current_filename]);
    load(full_external_path, 'SDATA');  % loads SDATA
    reref_to_nose(SDATA);
    
    % --- 3C. Call extract_baseline_orientation() for this subject
    [baseline_raw_subj, isi_subj, orientation_subj, orientation_reseponse_subj, outcome_missing, n_trials_extracted] = ...
        extract_baseline_orientation(SDATA, warning_codes_of_interest, all_warning_codes, subj);
    
   % is_not_nan = ~isnan(orientation_reseponse_subj) ;
    %baseline_raw_subj =  baseline_raw_subj(is_not_nan);
   % isi_subj = isi_subj(is_not_nan);
   % orientation_subj = orientation_subj(is_not_nan);
   % orientation_reseponse_subj = orientation_reseponse_subj(is_not_nan);

    [latencies, ~, ~, ~, ~, ~, ~, n_trials_selected] = select_single_trials_behavior(SDATA,warning_codes_of_interest,all_warning_codes, subj);
    
    is_good = is_good_trial(SDATA, latencies, total_epoch_samples, pred_end_sample);

    latencies = latencies(is_good & ~outcome_missing);
    baseline_raw_subj =  baseline_raw_subj (is_good & ~outcome_missing);
    isi_subj = isi_subj(is_good & ~outcome_missing);
    orientation_subj = orientation_subj(is_good & ~outcome_missing);
    orientation_reseponse_subj = orientation_reseponse_subj(is_good & ~outcome_missing);

    
 

    % FIXED padding (100 ms before + 100 ms after)
    PAD_MS = 100;  
    PAD_SAMPLES = round((PAD_MS / 1000) * Fs);
    
    alpha_baseline_tf = tf_and_trim_baseline_fixedpad( ...
                            baseline_raw_subj, ...
                            Fs, ...
                            alpha_freq_range, ...
                            PAD_SAMPLES);

    for t = 1:length(idx_main)
        all_baseline_raw{idx_main(t)} = alpha_baseline_tf(:,:,:,t);
    end
    
    % --- 3E. Fill new columns in the correct positions
    %all_baseline_raw(idx_main)          = baseline_raw_subj;
    all_orientation(idx_main)           = orientation_subj;
    all_orientation_reseponse(idx_main) = orientation_reseponse_subj;
    all_isi_from_baseline(idx_main)     = isi_subj;

    
    disp(['Subject ' subj ' merged successfully.']);
    
end

%% ===================================================
%  4. Save the updated dataset
% ===================================================


MasterTable.Baseline = all_baseline_raw(:);
MasterTable.Orientation = all_orientation(:);
MasterTable.Response = all_orientation_reseponse(:);
MasterTable.IrregularTargetTime = all_isi_from_baseline(:);

disp('Aggregation complete. Master table created for GLMM.');
head(MasterTable)


RESULTS_DIR = fullfile('C:\Users\ssassi\Desktop\Assaf_Rotation', 'Results'); 

% 3. Create the final filename
final_save_path = fullfile(RESULTS_DIR, ['GLMM_Master_Table_with_baseline2nd_half.mat']);

% 4. Save the table
save(final_save_path, 'MasterTable', '-v7.3');
disp(['Master table saved to: ' final_save_path]);

%% 

for i=1:height(MasterTable)
    MasterTable.Baseline{i} = squeeze(mean(MasterTable.Baseline{i}(:,:,currentROI), 3));
end

disp('Aggregation complete. Master table created for GLMM.');
head(MasterTable)


RESULTS_DIR = fullfile('C:\Users\ssassi\Desktop\Assaf_Rotation', 'Results'); 
                
% 3. Create the final filename
final_save_path = fullfile(RESULTS_DIR, ['GLMM_Master_Table_Occipital_with_baseline.mat']);

% 4. Save the table
save(final_save_path, 'MasterTable', '-v7.3');
disp(['Master table saved to: ' final_save_path]);
