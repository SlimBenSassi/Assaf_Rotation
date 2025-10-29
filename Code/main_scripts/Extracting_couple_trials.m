%% Exercise: Comparing Second-beat ERPs: rhythm vs interval, short vs long
%% 1. LOAD PREPROCESSED DATA

%file reading code from ERP.m script

clear; close all; clc
disp('--- Extracting a couple trials ---');

% --- Configuration ---
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

full_file_path = fullfile(filepath, filename);

% --- 2. Load the Data ---
try
    % Load only the SDATA variable from the selected .mat file
    load(full_file_path, 'SDATA');
    disp(['Loaded clean data from: ' filename]);
catch ME
    disp(['ERROR: Failed to load SDATA structure from ' filename]);
    disp(['MATLAB Error: ' ME.message]);
    error('Aborting due to critical data loading failure.');
end

%% --- Extract Global Variables ---
Fs = SDATA.info.sampling_rate;
data_matrix = SDATA.data; % [Time x Channels]
channel_labels = SDATA.info.channel_labels;
n_channels = size(data_matrix, 2);
events = SDATA.events.triggerChannel; % Raw Status Channel vector
alpha_freq_range = [8, 12]; % Alpha band for filtering (Hz)
pred_window_s = 0.100; % 100ms pre-stimulus prediction window

% --- Target Event Codes ---
codes_of_interest = [014, 024]; % Targets: Rhythm Short/Long, Interval Short/Long
report_unseen_code = 231; % Subjective report code for 'Did Not See'
report_seen_codes = [232, 233, 234]; % Subjective report codes for 'Saw' (Hit)

% --- 1. Extract Latencies and Codes from the Status Channel ---
status_vector = events;
trigger_indices = find(status_vector ~= 0); 
trigger_codes = status_vector(trigger_indices);

% --- 2. Link Targets to Subjective Outcomes (CRITICAL STEP) ---
% Initialize final lists
final_target_latencies = [];
final_target_codes = [];
y_subjective_outcome = []; % 0 = Unseen (Miss), 1 = Seen (Hit)

search_window = round(10 * Fs); % Search up to 500ms after target for response

for i = 1:length(trigger_codes)
    current_code = trigger_codes(i);
    
    if ismember(current_code, codes_of_interest)
        % Found a target event (014, 024, etc.)
        target_latency = trigger_indices(i);
        
        % Search for the next subjective report event
        found_report = false;
        
        % Search the next few indices (up to end of response window)
        for j = i + 1 : min(i + 10, length(trigger_codes)) 
            report_code = trigger_codes(j);
            
            if ismember(report_code, [report_unseen_code, report_seen_codes])
                % Found a report event (231, 232, etc.)
                
                % Check if the report occurred within a reasonable time window (e.g., 500ms)
                if trigger_indices(j) - target_latency < search_window
                    
                    final_target_latencies = [final_target_latencies; target_latency];
                    final_target_codes = [final_target_codes; current_code];
                    
                    % Determine binary outcome: 1 if Saw (232/233/234), 0 if Unseen (231)
                    if report_code == report_unseen_code
                        y_subjective_outcome = [y_subjective_outcome; 0]; % Unseen/Miss
                    else
                        y_subjective_outcome = [y_subjective_outcome; 1]; % Seen/Hit
                    end
                    found_report = true;
                    break; % Stop searching for reports once one is found
                end
            end
        end
        % Skip trials where no report was found (e.g., participant timed out)
    end
end
%%
% --- 3. Downsample to 20 Trials for Speedrun ---
target_trials_per_condition = 6; % Use a smaller number for faster testing
target_codes_for_downsample = [014, 024]; % Focus on Short Rhythm vs Short Interval comparison

% Find indices for target conditions
idx_112_full = find(final_target_codes == target_codes_for_downsample(1));
idx_212_full = find(final_target_codes == target_codes_for_downsample(2));

% Select first N trials of each group
idx_112_sample = idx_112_full(1:min(length(idx_112_full), target_trials_per_condition));
idx_212_sample = idx_212_full(1:min(length(idx_212_full), target_trials_per_condition));

final_sample_indices = [idx_112_sample; idx_212_sample];

% Final Trial Data
epoch_latencies = final_target_latencies(final_sample_indices);
epoch_codes = final_target_codes(final_sample_indices);
y_subjective_outcome = y_subjective_outcome(final_sample_indices);
n_trials = length(final_sample_indices);

disp(['Selected ' num2str(n_trials) ' trials for high-precision analysis.']);


%% 2. ALPHA POWER EXTRACTION (Filter, Hilbert, Extract Envelope)

% --- Configuration ---
PRE_EVENT_SEC = 0.5; 
POST_EVENT_SEC = 0.3;
total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;

% --- 1. Bandpass Filter (Alpha 8-12 Hz) ---
filter_order = 3; 
[b, a] = butter(filter_order, alpha_freq_range / (Fs/2));
data_filtered_alpha = filtfilt(b, a, double(data_matrix)); 

% --- 2. Epoching the Filtered Data ---
pre_samples = round(PRE_EVENT_SEC * Fs);
all_epochs_alpha = zeros(total_epoch_samples, n_channels, n_trials);

% for trial_idx = 1:n_trials
%     center_idx = epoch_latencies(trial_idx); 
%     start_idx = center_idx - pre_samples;
%     end_idx = center_idx + total_epoch_samples - 1;
% 
%     if start_idx >= 1 && end_idx <= size(data_matrix, 1)
%         all_epochs_alpha(:, :, trial_idx) = data_filtered_alpha(start_idx:end_idx, :);
%     end
% end

for trial_idx = 1:n_trials
    center_idx = epoch_latencies(trial_idx); 
    
    % The correct calculation for start and end indices:
    start_idx = center_idx - pre_samples;
    end_idx = start_idx + total_epoch_samples - 1; % Total length should be total_epoch_samples
    
    if start_idx >= 1 && end_idx <= size(data_matrix, 1)
        
        % Extract the segment
        % The right side is [end_idx - start_idx + 1] samples long.
        % This must equal total_epoch_samples (819).
        all_epochs_alpha(:, :, trial_idx) = data_filtered_alpha(start_idx:end_idx, :);
    end
end

disp('Alpha-band epoching complete.');


% --- 3. Hilbert Transform and Power Calculation ---
alpha_analytic = hilbert(all_epochs_alpha);
% Power is magnitude squared, often log-transformed (we use simple power here)
alpha_power_envelope = abs(alpha_analytic).^2;


%% 3. FEATURE EXTRACTION AND HYPOTHESIS TESTING

% --- 1. Define the Prediction Window ---
pred_start_sample = pre_samples + round(-pred_window_s * Fs); % e.g., sample 410 at 1024Hz
pred_end_sample = pre_samples; % sample 512 (0ms)

% --- 2. Define the Alpha ROI (Cz proxy) ---
pz_channel_idx = 48; 

% --- 3. Extract the Single-Trial Predictor (X) ---
% Average the Alpha power across the time window and the Pz channel for each trial.
% Result is [N_trials x 1] vector of mean Alpha power in the final 100ms.
alpha_power_predictor = squeeze(mean(mean(alpha_power_envelope(pred_start_sample:pred_end_sample, pz_channel_idx, :), 1), 2)); 


% --- 4. Hypothesis Testing (T-Test: Seen vs. Unseen) ---

% Split the alpha power predictor into two groups based on the subjective outcome (0 or 1)
alpha_seen_power = alpha_power_predictor(y_subjective_outcome == 1);
alpha_missed_power = alpha_power_predictor(y_subjective_outcome == 0);

disp('--- Running T-Test (Seen vs. Unseen) ---');
if length(alpha_seen_power) > 1 && length(alpha_missed_power) > 1
    
    [h, p, ci, stats] = ttest2(alpha_seen_power, alpha_missed_power);
    
    disp(['T-Test Result:']);
    disp(['   Mean Power (Seen): ' num2str(mean(alpha_seen_power), 3)]);
    disp(['   Mean Power (Missed): ' num2str(mean(alpha_missed_power), 3)]);
    disp(['   T-stat = ' num2str(stats.tstat, 3) ', P-value = ' num2str(p, 4)]);
    
    % --- Plotting the Prediction ---
    figure;
    scatter(alpha_power_predictor, y_subjective_outcome, 50, 'filled');
    title('Single-Trial Alpha Power Predicting Subjective Outcome (0=Miss, 1=Seen)');
    xlabel(['Alpha Power (Avg ' num2str(pred_window_s*1000) 'ms pre-stim)']);
    ylabel('Subjective Outcome');
    ylim([-0.1 1.1]);
    
else
    disp('Warning: Not enough trials in one group to run T-test.');
end


%% 3. TIME-RESOLVED FEATURE EXTRACTION AND HYPOTHESIS TESTING

% --- 1. Configuration ---
pz_channel_idx = 48; % Cz proxy channel
Fs = round(total_epoch_samples / (PRE_EVENT_SEC + POST_EVENT_SEC)); % Recalculate Fs from epoch length
time_points = 1:total_epoch_samples; 

% --- 2. Define the Pre-Stimulus Time Window ---
% We will analyze the entire pre-stimulus period (-500ms to 0ms)
pred_start_time_idx = time_points(pred_start_sample);
pred_end_time_idx = time_points(pred_end_sample);
time_window_samples = pred_end_sample - pred_start_sample + 1;

% --- 3. Initialize Output Matrices ---
% This will store the T-statistic and P-value for EVERY time point.
t_map_vector = zeros(time_window_samples, 1);
p_map_vector = zeros(time_window_samples, 1);
time_vec_pred = erp_time_vec(pred_start_sample:pred_end_sample);

disp(['Running time-resolved T-test across ' num2str(time_window_samples) ' time points...']);
tic

% --- 4. THE TIME-RESOLVED LOOP ---
for t_idx = 1:time_window_samples
    
    % --- A. Extract Predictor for the current time point ---
    % Squeeze removes single dimensions. We take only the current time index (t_idx).
    % Result is a [N_trials x 1] vector of Alpha Power at this specific millisecond.
    alpha_power_at_t = squeeze(alpha_power_envelope(pred_start_sample + t_idx - 1, pz_channel_idx, :));

    % --- B. Split into Groups ---
    alpha_seen_power = alpha_power_at_t(y_subjective_outcome == 1);
    alpha_missed_power = alpha_power_at_t(y_subjective_outcome == 0);
    
    % --- C. Run T-Test ---
    if length(alpha_seen_power) > 1 && length(alpha_missed_power) > 1
        % ttest2 returns the p-value first, then the stats structure
        [h, p_val, ~, stats] = ttest2(alpha_seen_power, alpha_missed_power);
        
        t_map_vector(t_idx) = stats.tstat;
        p_map_vector(t_idx) = p_val;
    else
        t_map_vector(t_idx) = NaN; % Assign NaN if not enough trials
    end
end
toc
disp('Time-resolved T-map computation finished.');


%% 4. PLOT TIME-RESOLVED T-MAP

figure;

% --- 1. Plot the T-statistic over time ---
plot(time_vec_pred, t_map_vector, 'b', 'LineWidth', 2);
hold on;

% --- 2. Add Significance Threshold (Alpha = 0.05) ---
% We calculate the threshold based on the T-distribution degrees of freedom (N_trials - 2)
df = n_trials - 2;
if df > 0
    t_critical_pos = tinv(0.975, df); % T-value for p < 0.05 two-tailed
    t_critical_neg = tinv(0.025, df);
    
    % Plot horizontal significance lines
    line(xlim, [t_critical_pos t_critical_pos], 'Color', 'r', 'LineStyle', ':');
    line(xlim, [t_critical_neg t_critical_neg], 'Color', 'r', 'LineStyle', ':');
end

% --- 3. Aesthetics ---
line([0 0], ylim, 'Color', 'k', 'LineStyle', '--'); % Vertical line at stimulus onset
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', '-'); % Horizontal line at T=0
title(['Time-Resolved T-Map (Alpha Power: Seen vs. Unseen) at Ch ' num2str(pz_channel_idx)]);
xlabel('Time relative to stimulus (s)');
ylabel('T-Statistic (Predictive Strength)');
grid on;
hold off;
