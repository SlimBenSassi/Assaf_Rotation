%% Exercise: Comparing Second-beat ERPs: rhythm vs interval, short vs long
%% 1. LOAD PREPROCESSED DATA

%file reading code from ERP.m script

clear; close all; clc
disp('--- Exercise ---');

% --- Configuration ---
DEFAULT_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Data'; % Default folder where your processed data is saved
%example christina data: \\wks3\pr_breska\el-Christina\Backup Copy Christina\PF_Poster\Data\EEG\

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

% --- Extract Global Variables ---
Fs = SDATA.info.sampling_rate;
data_matrix = SDATA.data; % [Time x Channels]
channel_labels = SDATA.info.channel_labels;
n_channels = size(data_matrix, 2);
%events = SDATA.events.triggerChannel;

%% 2. EPOCHING 


% --- CONFIGURATION ---
baseline_window_s = 0.5; % 500ms baseline (e.g., -500ms to 0ms)
erp_window_s = 1.0;      % 1000ms post-event (e.g., 0ms to 1000ms)
PRE_EVENT_SEC = -baseline_window_s; 
POST_EVENT_SEC = erp_window_s;
codes_of_interest = [112, 122, 212, 222];

% --- 1. Extract Latencies and Codes from the Status Channel ---
status_vector = SDATA.events.triggerChannel; % This is the [Time x 1] vector

% Find the sample indices where an event occurred (value is non-zero)
trigger_indices = find(status_vector ~= 0); 

% Get the actual event code (value) at those specific indices
trigger_codes = status_vector(trigger_indices);

% --- 2. Filter Epochs of Interest ---
% Find the indices corresponding to our target event codes (112, 222, etc.)
idx_epochs_of_interest = ismember(trigger_codes, codes_of_interest);

% --- 3. Final List of Epoch Parameters ---
epoch_latencies = trigger_indices(idx_epochs_of_interest);
epoch_codes = trigger_codes(idx_epochs_of_interest); % This preserves the 112 vs 222 distinction

n_total_epochs = length(epoch_latencies);
disp(['Found ' num2str(n_total_epochs) ' total trials of interest.']);


%% 3. EPOCHING LOGIC

% --- 1. Calculate Samples ---
pre_samples = round(-PRE_EVENT_SEC * Fs); 
post_samples = round(POST_EVENT_SEC * Fs);
total_epoch_samples = pre_samples + post_samples; % Total time points in the epoch

% --- 2. Initialize 3D Epoch Array and Labels ---
all_epochs = zeros(total_epoch_samples, n_channels, n_total_epochs);
epoch_labels = zeros(n_total_epochs, 1); % CRITICAL: Saves the event code

% --- 3. The Epoching Loop ---
for trial_idx = 1:n_total_epochs
    
    center_idx = epoch_latencies(trial_idx); % The time point (sample index) of the event
    
    start_idx = center_idx - pre_samples;
    end_idx = center_idx + post_samples - 1;
    
    % Safety check: ensure indices are within bounds
    if start_idx >= 1 && end_idx <= size(data_matrix, 1)
        
        % Extract the segment
        all_epochs(:, :, trial_idx) = data_matrix(start_idx:end_idx, :);
        
        % Store the original event code as the label
        epoch_labels(trial_idx) = epoch_codes(trial_idx);
    end
end

disp('Epoching complete. Epoch labels saved for classification.');
% all_epochs (3D data matrix) and epoch_labels (1D vector of codes) are ready!

%% BASELINE CORRECTION (Crucial Missing Step)

% Define baseline end point using pre_samples (the time point corresponding to 0s)
baseline_end_sample = pre_samples; 

% 1. Calculate the mean of the pre-stimulus period (across dimension 1: Time)
% We assume the baseline window runs from the start of the epoch (sample 1) up to the stimulus (pre_samples).
baseline_mean = mean(all_epochs(1:baseline_end_sample, :, :), 1); 

% 2. Subtract the baseline mean from all data points across all trials
% The result is saved to the variable the rest of your script expects:
all_epochs_corrected = bsxfun(@minus, all_epochs, baseline_mean);

disp('Baseline correction applied successfully. Data is now ready for averaging.');

%% 4. GROUP EPOCHS AND CALCULATE CONDITION ERPs

% Codes: [Type][Length][Target] -> 1=Rhythm/2=Interval, 1=Short/2=Long

% --- Define the four conditions based on your labels (epoch_codes) ---
idx_Rhythm_Short = (epoch_labels == 112); 
idx_Rhythm_Long  = (epoch_labels == 122); 
idx_Interval_Short = (epoch_labels == 212); 
idx_Interval_Long  = (epoch_labels == 222); 

% --- Calculate Grand Average for Each Condition ---
% Average across the 3rd dimension (Trials) for the selected indices
ERP_Rhythm_Short = mean(all_epochs_corrected(:, :, idx_Rhythm_Short), 3); 
ERP_Rhythm_Long  = mean(all_epochs_corrected(:, :, idx_Rhythm_Long), 3);
ERP_Interval_Short = mean(all_epochs_corrected(:, :, idx_Interval_Short), 3);
ERP_Interval_Long  = mean(all_epochs_corrected(:, :, idx_Interval_Long), 3);


%% 5. GENERATE 2x2 COMPARISON PLOT

% --- Configuration ---
Fs = SDATA.info.sampling_rate;
% ERP time vector (calculated previously in Section 3)
erp_time_vec = linspace(PRE_EVENT_SEC, POST_EVENT_SEC, size(ERP_Rhythm_Short, 1));
% Find a central-parietal channel (where P3/N400 effects are often largest)
channel_idx = 31; % Using Channel  Pz

figure('Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]); % Wide figure

% --- Standard Plotting Setup ---
plot_setup = @(t) [line([0 0], ylim, 'Color', 'k', 'LineStyle', '--'); ...
                   line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':'); ...
                   set(gca, 'YDir', 'reverse'), title(t), xlabel('Time (s)'), ylabel('Amplitude (\muV)')];


% --- SUBPLOT 1: RHYTHM SHORT vs. LONG (Interval Length Effect in Rhythm) ---
subplot(2, 2, 1);
plot(erp_time_vec, ERP_Rhythm_Short(:, channel_idx), 'b', 'LineWidth', 2, 'DisplayName', 'Short (112)');
hold on;
plot(erp_time_vec, ERP_Rhythm_Long(:, channel_idx), 'r', 'LineWidth', 2, 'DisplayName', 'Long (122)');
plot_setup('Rhythm: Short vs. Long Interval');
legend('show', 'Location', 'SouthEast');


% --- SUBPLOT 2: INTERVAL SHORT vs. LONG (Interval Length Effect in Memory) ---
subplot(2, 2, 2);
plot(erp_time_vec, ERP_Interval_Short(:, channel_idx), 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'Short (212)');
hold on;
plot(erp_time_vec, ERP_Interval_Long(:, channel_idx), 'Color', [0.8 0.2 0.8], 'LineWidth', 2, 'DisplayName', 'Long (222)');
plot_setup('Interval: Short vs. Long Interval');
legend('show', 'Location', 'SouthEast');


% --- SUBPLOT 3: RHYTHM vs. INTERVAL (Short Duration, Structure Effect) ---
subplot(2, 2, 3);
plot(erp_time_vec, ERP_Rhythm_Short(:, channel_idx), 'b', 'LineWidth', 2, 'DisplayName', 'Rhythm (112)');
hold on;
plot(erp_time_vec, ERP_Interval_Short(:, channel_idx), 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'Interval (212)');
plot_setup('Short Duration: Rhythm vs. Interval Structure');
legend('show', 'Location', 'SouthEast');


% --- SUBPLOT 4: RHYTHM vs. INTERVAL (Long Duration, Structure Effect) ---
subplot(2, 2, 4);
plot(erp_time_vec, ERP_Rhythm_Long(:, channel_idx), 'r', 'LineWidth', 2, 'DisplayName', 'Rhythm (122)');
hold on;
plot(erp_time_vec, ERP_Interval_Long(:, channel_idx), 'Color', [0.8 0.2 0.8], 'LineWidth', 2, 'DisplayName', 'Interval (222)');
plot_setup('Long Duration: Rhythm vs. Interval Structure');
legend('show', 'Location', 'SouthEast');


%% 6. SAVE FINAL FIGURE AND DATA

% Save the full ERP matrix for later statistical analysis
save(fullfile('Results', 'second_beats_final_ERP_matrix.mat'), 'ERP_matrix', 'erp_time_vec', '-v7.3');
disp('ERP matrix saved to Results/second_beats_final_ERP_matrix.mat');

% Save the comparison figure as a PNG
figure_filename_png = fullfile('Results', 'Second beats: Final_2x2_ERP_Comparison.png');
print(gcf, figure_filename_png, '-dpng', '-r300'); 
disp(['Comparison figure saved to Results/Second beats: Final_2x2_ERP_Comparison.png']);