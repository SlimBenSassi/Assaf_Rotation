%% 1. INITIALIZATION AND FILE SELECTION

clear; close all; clc
disp('--- Starting Multi-Subject Data Aggregation for GLMM , TODO FIX LINE NUMBERS MAY NOT EXCEED ---');


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

alpha_freq_range = [8, 12]; % Alpha band for filtering (Hz)

% --- Time Variables --- %
Fs = 1024; %change if needed
n_channels = 71; %change if needed
pred_window_s = 0.100; % 200ms pre-stimulus prediction window (used for non-time resolved average across window)
PRE_EVENT_SEC = 0.5; %taking all time since warning signal 
POST_EVENT_SEC = 0.1;
total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;
pre_samples = round(PRE_EVENT_SEC * Fs);
pred_start_sample = pre_samples + round(-pred_window_s * Fs); % e.g., sample 410 at 1024Hz
pred_end_sample = pre_samples;

% ---  Event Codes --- %
target_codes = [011, 012, 013, 014, 015, 016, 017, 021, 022, 023, 024, 025, 026, 027, 211, 212, 213, 214, 215, 216, 217, 221, 222, 223, 224, 225, 226, 227]; % (01X means contrast X+1) Targets: Rhythm Target Contrast 4,5,6,7 Right, same but Left (contrasts around threshold) 
%target_codes = [014, 015, 016, 024, 025, 026, 214, 215, 216, 224, 225, 226];
%target_codes = [014, 015, 016, 024, 025, 026];
report_unseen_code = [231, 241]; % Subjective report code for 'Did Not See' for rhythm 231, for interval 241
report_seen_codes = [232, 233, 234, 242, 243, 244]; % Subjective report
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

%%

% 1. INITIALIZE AGGREGATION VECTORS

% These vectors will accumulate the data from ALL subjects
all_subject_ids = {};       % Will store subject labels as strings
all_alpha_power = [];       % Will store the single-trial Alpha power predictor (X1)
all_stim_intensity_raw = [];    % Will store the Stimulus Intensity RAW
all_stim_intensity = [];    % Will store the Stimulus Intensity (X2)
all_subjective_outcome = [];% Will store the binary subjective outcome (Y)


% --- START LOOP ---
for sub_idx = 1:N_SUBJECTS
    
    current_filename = filenames{sub_idx};
    full_path = fullfile(filepath, current_filename);
    disp(['-----------------------------------------------------']);
    
    % --- A. EXTRACT SUBJECT ID (CRUCIAL ADDITION) ---
    % 1. Search for the pattern _SubjXXX_ and extract the number.
    %    sscanf searches the string for the integer after the pattern Subj.
    extracted_id = sscanf(current_filename, 'EEG_SxA_Subj%d_', 1); 
    
    % 2. Convert to string for use as a categorical variable later.
    if ~isempty(extracted_id)
        subject_id_str = num2str(extracted_id);
    else
        % Fallback if ID is not found (use the loop index)
        subject_id_str = ['Subj' num2str(sub_idx)]; 
    end

    disp(['STARTING SUBJECT ' subject_id_str ': ' current_filename]);

    % --- B. Load Subject's Data ---
    % Loads SDATA, which includes data_matrix, Fs, and events.
    load(full_path);

    % --- B. Define/Update Global Variables for Current Subject ---
    % These variables MUST be updated inside the loop as they change per file.
    Fs = SDATA.info.sampling_rate;
    data_matrix = SDATA.data;
    n_channels = size(data_matrix, 2);
    
    % reref to noise if not already
    SDATA = reref_to_nose(SDATA);

    
    % --- C. Trial Selection and Filtering (Function Calls) ---
    % 1. Find and link targets to responses (The complex selection logic)
    [latencies, codes, intensities, outcomes, n_trials] = select_single_trials(SDATA, target_codes, report_unseen_code, report_seen_codes);
    
    % 2. Downsample trials (Optional, using the max limit)
    % [latencies, codes, outcomes, n_trials] = downsample_trials(latencies, codes, outcomes, MAX_TARGET_TRIALS); 

    % 3. Reject trials overlapping with artifact mask
    [latencies, codes, intensities, outcomes, n_trials] = reject_artifact_trials(SDATA, latencies, codes, intensities, outcomes, total_epoch_samples, pre_samples);
    
    % 4. Epoching and Time-Frequency Analysis (The most complex step)
    all_epochs_padded = epoch_with_padding(data_matrix, Fs, n_channels, n_trials, latencies, total_epoch_samples, pre_samples, alpha_freq_range);
    
    alpha_power_envelope = tf_and_trim(all_epochs_padded, Fs, alpha_freq_range);

    
    % 1. Slice the 4D matrix to the prediction window and ROI
    % We select the time window (Dim 1), all frequencies (Dim 2), and the ROI channels (Dim 3).
    sliced_predictors = alpha_power_envelope(pred_start_sample:pred_end_sample, :, currentROI, :);

    % 2. Calculate the FINAL mean across Time, Freq, and Channels
    % The result is a 1x1x1xN_Trials array, which needs to be squeezed.
    % We are averaging over Dimension 1 (Time), Dimension 2 (Freq), and Dimension 3 (Channels).
    current_alpha_power = squeeze(mean(mean(mean(sliced_predictors, 1), 2), 3)); 
    
    % 3. CONVERT TO COLUMN VECTORS FOR AGGREGATION
    % The final aggregated data must be N_Trials x 1 column vectors.
    current_alpha_power = current_alpha_power(:); 
    current_subjective_outcome = outcomes(:);
    current_stim_intensity = intensities(:);

    % 5. STANDARDIZATION (Z-SCORING WITHIN SUBJECT) for alpha power
    
    % Apply Z-score to Alpha Power (X1) and stimintensity
    current_alpha_power_z = zscore_vector(current_alpha_power);
    current_stim_intensity_z = zscore_vector(current_stim_intensity);

% --- 6. Append to Master Vectors (The Aggregation) ---
    % Append the Z-scored versions!
    all_alpha_power = [all_alpha_power; current_alpha_power_z];
    all_stim_intensity = [all_stim_intensity; current_stim_intensity_z];
    all_stim_intensity_raw = [all_stim_intensity_raw; current_stim_intensity];

    % 4. Append to Master Vectors (The Aggregation) ---
    % NOTE: Subject ID must be stored as a cell array of strings for GLMM.
    all_subject_ids = [all_subject_ids; repmat({subject_id_str}, n_trials, 1)];
    %all_alpha_power = [all_alpha_power; current_alpha_power];
    %all_stim_intensity = [all_stim_intensity; current_stim_intensity];
    all_subjective_outcome = [all_subjective_outcome; current_subjective_outcome];
    
    disp(['Subject ' subject_id_str ' processed. ' num2str(n_trials) ' trials appended.']);
% --- END OF SUBJECT LOOP ---

end



%% 3. FINALIZATION AND INSPECTION (Creating the Master Table)


% 1. Calculate the dynamic predictor column name
alpha_predictor_name = strcat('AlphaPower_Avg_', num2str(pred_window_s * 1000), 'ms');% NOTE: We use [] and strings/char arrays to guarantee compatibility here.


% 2. Create the VariableNames cell array explicitly
column_names = {'SubjectiveOutcome', alpha_predictor_name, 'StimIntensity', 'StimIntensityRaw', 'SubjectID'};


% --- Now the table creation is simplified and robust ---
MasterTable = table(all_subjective_outcome, all_alpha_power, all_stim_intensity, all_stim_intensity_raw, all_subject_ids, 'VariableNames', column_names); % <-- Uses clean variable


% 2. Convert SubjectID and Outcome to Categorical/Logical for GLMM (Crucial)
MasterTable.SubjectID = categorical(MasterTable.SubjectID);
MasterTable.SubjectiveOutcome = logical(MasterTable.SubjectiveOutcome); % Convert 0/1 to logical

disp('Aggregation complete. Master table created for GLMM.');
head(MasterTable)

% --- 1. Locate the Repository Root Folder ---
% We use the which command to find the location of the current script, 
% then use fileparts() to step back to the repository's root directory.
%script_path = fileparts(which(mfilename)); % Gets the directory of the current running script
%repo_root = fileparts(fileparts(script_path)); % Assumes script is 2 levels down (main_scripts)

% 2. Construct the full path to the 03_RESULTS folder
RESULTS_DIR = fullfile("C:\Users\ssassi\Desktop\Assaf_Rotation", 'Results'); 

% 3. Create the final filename
final_save_path = fullfile(RESULTS_DIR, 'GLMM_Master_Table_pilot_biggest_zscored_and_rawstim.2.0.mat');

% 4. Save the table
save(final_save_path, 'MasterTable', '-v7.3');
disp(['Master table saved to: ' final_save_path]);