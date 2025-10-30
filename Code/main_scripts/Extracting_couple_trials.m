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

%% --- Global Variables ---

Fs = SDATA.info.sampling_rate;
data_matrix = SDATA.data; % [Time x Channels]
channel_labels = SDATA.info.channel_labels;
n_channels = size(data_matrix, 2);
events = SDATA.events.triggerChannel; % Raw Status Channel vector
alpha_freq_range = [8, 12]; % Alpha band for filtering (Hz)
pred_window_s = 0.300; % 100ms pre-stimulus prediction window
% ---  Event Codes ---
target_codes = [014, 024]; % Targets: Rhythm Short/Long, Interval Short/Long
report_unseen_code = 231; % Subjective report code for 'Did Not See'
report_seen_codes = [232, 233, 234]; % Subjective report codes for 'Saw' (Hit)
max_target_trials_per_condition = 6; 


%% 1.1 Finding trials of interest

% --- 1. Extract Latencies and Codes from the Status Channel ---
status_vector = events;
trigger_indices = find(status_vector ~= 0); 
trigger_codes = status_vector(trigger_indices);

% --- 2. Link Targets to Subjective Outcomes  ---
% Initialize final lists
final_target_latencies = [];
final_target_codes = [];
y_subjective_outcome = []; % 0 = Unseen (Miss), 1 = Seen (Hit)

search_window = round(10 * Fs); % Search up to 10ms after target for response

for i = 1:length(trigger_codes)
    current_code = trigger_codes(i);
    
    if ismember(current_code, target_codes)
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


% Calculate total number of successful target-response pairings
total_pairings = length(final_target_codes);

% Count occurrences of each target code that made it into the final list
unique_targets = unique(final_target_codes);

disp('--- Target-Response Linking Summary ---');
disp(['Total successfully linked trials (Targets + Responses): ' num2str(total_pairings)]);

% Loop through the specific codes of interest to display individual counts
for code = target_codes
    count = sum(final_target_codes == code);
    disp(['  > Code ' num2str(code) ' found: ' num2str(count) ' trials.']); %TODO: make this show 0s too
end
disp('---------------------------------------');

%% Optional: 1.2 Taking less trials to run stuff quickly


% --- 1. Initialize Storage for Sampled Indices ---
sampled_indices_cell = cell(1, length(target_codes));
disp('--- Optional: Taking less trials to run stuff quickly ---');

% --- 2. Loop Through Each Target Code ---
for i = 1:length(target_codes)
    code = target_codes(i);
    
    % Find all indices for the current target code (e.g., all 014 trials)
    idx_full = find(final_target_codes == code);
    
    % Select the first N trials (up to target_trials_per_condition)
    n_available = length(idx_full);
    n_sample = min(n_available, max_target_trials_per_condition);
    
    % Store the sampled indices in the cell array
    sampled_indices_cell{i} = idx_full(1:n_sample);
    
    disp(['  > Code ' num2str(code) ': Sampled ' num2str(n_sample) ' of ' num2str(n_available) ' available trials.']);
end

% --- 3. Consolidate Indices and Data ---
% Vertically stack all sampled indices into one master list
final_sample_indices = vertcat(sampled_indices_cell{:});

% Final Trial Data is sliced using the master list
epoch_latencies = final_target_latencies(final_sample_indices);
epoch_codes = final_target_codes(final_sample_indices);
y_subjective_outcome = y_subjective_outcome(final_sample_indices);

n_trials = length(final_sample_indices);
disp(['Selected ' num2str(n_trials) ' total trials for analysis.']);


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

%% 2.bis ALPHA POWER EXTRACTION (Lab's TFhilbert Method)

% --- Configuration ---
PRE_EVENT_SEC = 0.5; 
POST_EVENT_SEC = 0.3;
total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;


% We analyze only the single Alpha band (8-12 Hz) for this single-trial test.
frequencies_to_analyze = [alpha_freq_range(1): alpha_freq_range(2)]; % Use 10 Hz as the center frequency
filt_width_octaves = 0.2; 
causalFilt = false; % Use zero-phase filtering

% --- 1. Epoching the Clean Data (Output: [Time x Channels x Trials]) ---
pre_samples = round(PRE_EVENT_SEC * Fs);
all_epochs_alpha = zeros(total_epoch_samples, n_channels, n_trials);

for trial_idx = 1:n_trials
    center_idx = epoch_latencies(trial_idx); 
    start_idx = center_idx - pre_samples;
    end_idx = start_idx + total_epoch_samples - 1;
    
    if start_idx >= 1 && end_idx <= size(data_matrix, 1)
        all_epochs_alpha(:, :, trial_idx) = data_matrix(start_idx:end_idx, :);
    end
end
disp('Alpha-band epoching complete.');

% --- 2. TFhilbert Call (CRITICAL DIMENSIONAL SWAP) ---
% The function processes trials sequentially. We must process each channel across all trials.
% We reshape the 3D data into [Trials*Time x Channels] and then process it, 
% or process trial-by-trial, which is cleaner.

% The function expects [Trials/Channels x Time]. We will call it in a simple loop 
% over the channels, processing time across trials.

% Initialize the final power envelope matrix [Time x Channels x Trials]
alpha_power_envelope = zeros(total_epoch_samples, length(frequencies_to_analyze), n_channels, n_trials);

for ch = 1:n_channels
    % Extract all time courses for the current channel across all trials: [Time x 1 x Trials]
    channel_data = squeeze(all_epochs_alpha(:, ch, :)); 
    
    % Call the lab's function: Input must be [Trials x Time]
    % Transpose the channel_data [Time x Trials] to [Trials x Time]
    [hil_amp, ~] = TFhilbert(frequencies_to_analyze, filt_width_octaves, channel_data', Fs, causalFilt);
    
    % TFhilbert output is [Freqs x Time x Trials]. 
    % We store the power (amp^2) back into the final matrix:
    alpha_power_envelope(:, :, ch, :) = permute(hil_amp.^2, [2 1 3]);
end

disp('TFhilbert processing complete. Power envelope calculated.');


%% 3. FEATURE EXTRACTION AND HYPOTHESIS TESTING

% --- 1. Define the Prediction Window ---
pred_start_sample = pre_samples + round(-pred_window_s * Fs); % e.g., sample 410 at 1024Hz
pred_end_sample = pre_samples; % sample 512 (0ms)

% --- 2. Define the Alpha ROI (Cz proxy) ---
cz_channel_idx = 48; 

% --- 3. Extract the Single-Trial Predictor (X) ---
% Average the Alpha power across the time window and the Pz channel for each trial.
% Result is [N_trials x 1] vector of mean Alpha power in the final 100ms.
%alpha_power_predictor = squeeze(mean(mean(mean(alpha_power_envelope(pred_start_sample:pred_end_sample, :, cz_channel_idx, :), 1), 2), 4));

% 1. Slice the window: [Time x Freqs x 1 x Trials]
sliced_power = alpha_power_envelope(pred_start_sample:pred_end_sample, :, cz_channel_idx, :);

% 2. Average only across Time (Dim 1) and Frequencies (Dim 2)
mean_power_across_tf = squeeze(mean(mean(sliced_power, 1), 2));
% The result of this average is now [Trials x 1] (because the first two dimensions collapsed)

% 3. Final assignment: Ensure it's a column vector [N_trials x 1]
alpha_power_predictor = reshape(mean_power_across_tf, n_trials, 1);

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


%% 4. TIME-RESOLVED FEATURE EXTRACTION AND HYPOTHESIS TESTING

% --- 1. Configuration ---
cz_channel_idx = 48; % Cz proxy channel
Fs = round(total_epoch_samples / (PRE_EVENT_SEC + POST_EVENT_SEC)); % Recalculate Fs from epoch length
time_points = 1:total_epoch_samples; 

% --- 2. Define the Pre-Stimulus Time Window ---
% We will analyze the entire pre-stimulus period (-500ms to 0ms)
pred_start_time_idx = time_points(pred_start_sample);
pred_end_time_idx = time_points(pred_end_sample);
time_window_samples = pred_end_sample - pred_start_sample + 1;
pred_start_sample_idx = 1; 
pred_end_sample_idx = pre_samples; 

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
    %alpha_power_at_t = squeeze(alpha_power_envelope(pred_start_sample + t_idx - 1, cz_channel_idx, :));
    % Indexing: (Time_Index, ALL Freqs, Channel_Index, All_Trials)
    alpha_power_at_pixel_4D = squeeze(alpha_power_envelope(pred_start_sample_idx + t_idx - 1, :, cz_channel_idx, :)); 
    
    % FIX: Average the power across the Frequency dimension (Dimension 1 of the squeezed output)
    alpha_power_at_t = mean(alpha_power_at_pixel_4D, 1);
    % Output is now a 1xN_Trials vector of average Alpha power for this time point.
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


%% 4.1 PLOT TIME-RESOLVED T-MAP

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
title(['Time-Resolved T-Map (Alpha Power: Seen vs. Unseen) at Ch ' num2str(cz_channel_idx)]);
xlabel('Time relative to stimulus (s)');
ylabel('T-Statistic (Predictive Strength)');
grid on;
hold off;

%% 5.1 TIME-FREQUENCY (T-MAP) FEATURE EXTRACTION AND HYPOTHESIS TESTING

% --- 1. Configuration ---
cz_channel_idx = 48; % cz proxy channel (The analysis is run on one channel at a time)

% --- Variables for the Map Axes ---
n_time_points = size(alpha_power_envelope, 1); % Total time points in the epoch
n_freqs = size(alpha_power_envelope, 2);      % Total frequencies analyzed (5 in your case)

% Define the Pre-Stimulus Time Window (Samples)
pred_start_sample_idx = 1; 
pred_end_sample_idx = pre_samples; 

% Slice the time-frequency data to the prediction window
time_window_samples = pred_end_sample_idx - pred_start_sample_idx + 1;

% --- 2. Initialize Output Matrix ---
% The T-Map will be a 2D matrix: [Time Points x Frequencies]
t_map_matrix = zeros(time_window_samples, n_freqs);
p_map_matrix = zeros(time_window_samples, n_freqs);

disp(['Running T-test across ' num2str(time_window_samples) ' time points and ' num2str(n_freqs) ' frequencies...']);
tic

% --- 3. THE TIME-FREQUENCY LOOP ---
for f_idx = 1:n_freqs % Outer Loop: Iterate through each of the 5 frequencies
    
    for t_idx = 1:time_window_samples % Inner Loop: Iterate through each time point
        
        % --- A. Extract Predictor for the current (t, f) pixel ---
        % Extract the power vector across all trials at this specific time and frequency.
        % The index is alpha_power_envelope(Time, Freq, Trial)
        alpha_power_at_pixel = squeeze(alpha_power_envelope(pred_start_sample_idx + t_idx - 1, f_idx, cz_channel_idx, :));

        % --- B. Split into Groups ---
        alpha_seen_power = alpha_power_at_pixel(y_subjective_outcome ==1); % commented out just bcz the slicing of the array is already
        %done in section 3, same for next line
        alpha_missed_power = alpha_power_at_pixel(y_subjective_outcome == 0);
        
        % --- C. Run T-Test (Prediction Test) ---
        if length(alpha_seen_power) > 1 && length(alpha_missed_power) > 1
            [h, p_val, ~, stats] = ttest2(alpha_seen_power, alpha_missed_power);
            
            % Store the T-statistic and P-value in the matrix
            t_map_matrix(t_idx, f_idx) = stats.tstat;
            p_map_matrix(t_idx, f_idx) = p_val;
        else
            t_map_matrix(t_idx, f_idx) = NaN; % Not enough trials
        end
    end % End Time Loop (t_idx)
    
end % End Frequency Loop (f_idx)
toc
disp('Time-Frequency T-map computation finished.');


%% 5.2. PLOT TIME-FREQUENCY T-MAP (Visualization)

figure('Name', 'Predictive T-Map');

% --- Configuration for Plot Axes ---
time_vec_pred = time_points(pred_start_sample_idx : pred_end_sample_idx);
freq_vec = [8:12]; % We only analyzed 8, 10, 12 Hz, so we use the range for plotting.
if length(freq_vec) > n_freqs % Adjust if the simple [8:12] is too long for the 5 freqs
    freq_vec = linspace(alpha_freq_range(1), alpha_freq_range(2), n_freqs);
end

% --- Plotting the T-Map ---
imagesc(time_vec_pred, freq_vec, t_map_matrix'); 
% CRITICAL: Transpose t_map_matrix' for imagesc: [Freq x Time]

axis xy; % Corrects the Y-axis direction (low frequencies at bottom)
colorbar;
caxis([-4 4]); % Standard T-statistic range for visualization
colormap('jet'); % High-contrast colormap (e.g., 'jet' or 'parula')

% --- Aesthetics ---
line([0 0], ylim, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--'); % Vertical line at stimulus onset
title(['Time-Frequency T-Map (Alpha Power Predicts Awareness) at Ch ' num2str(cz_channel_idx)]);
xlabel('Time relative to stimulus (s)');
ylabel('Frequency (Hz)');