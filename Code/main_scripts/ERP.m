%% ERP ANALYSIS SPEEDRUN: CNV and P3 Components

% --- GOAL: Calculate and visualize two key timing ERPs (P3 and CNV) ---
%% 1. INITIALIZE AND LOAD CLEAN DATA

clear; close all; clc
disp('--- Starting ERP Speedrun---');

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
%%

% --- CRITICAL FIX: Load REAL Event Markers ---
% Now using the indices extracted from the Status Channel in the preprocessing script.
if ~isfield(SDATA.events, 'trigger_indices') || isempty(SDATA.events.trigger_indices)
    error('No trigger_indices found in SDATA. Please fix preprocessing event extraction.');
end

event_indices = SDATA.events.trigger_indices; % <--- USES REAL INDICES
n_trials = length(event_indices);

disp(['Loaded ' num2str(n_trials) ' actual trials at ' num2str(Fs) ' Hz.']);


%% 2. EPOCHING AND BASELINE CORRECTION

% We will use a wide window to capture both CNV and P3 components.
PRE_EVENT_SEC = -1.0; % 1000ms pre-event (CNV preparation)
POST_EVENT_SEC = 1.0;  % 1000ms post-event (P3)

% --- Epoching Logic ---
pre_samples = round(-PRE_EVENT_SEC * Fs); 
total_epoch_samples = round((POST_EVENT_SEC - PRE_EVENT_SEC) * Fs);

all_epochs = zeros(total_epoch_samples, n_channels, n_trials);
total_samples_in_data = size(data_matrix, 1);

for trial_idx = 1:n_trials
    
    center_idx = event_indices(trial_idx);
    start_idx = center_idx - pre_samples;
    end_idx = start_idx + total_epoch_samples - 1;
    
    % Ensure indexing is within bounds (safety check for start/end)
    if start_idx >= 1 && end_idx <= total_samples_in_data
        all_epochs(:, :, trial_idx) = data_matrix(start_idx:end_idx, :);
    else
        % Skip trials that are too close to the beginning or end of the recording
        disp(['Warning: Skipping trial ' num2str(trial_idx) ' due to boundary.']);
    end
end
disp('Epoching complete.');


% --- Baseline Correction ---
baseline_mean = mean(all_epochs(1:pre_samples, :, :), 1); % Mean of baseline samples
all_epochs_corrected = bsxfun(@minus, all_epochs, baseline_mean);
disp('Baseline correction complete.');


%% 3. CALCULATE GRAND AVERAGE ERP (P3 and CNV)

% Average across the 3rd dimension (Trials) to get the ERP matrix [Time x Channels]
ERP_matrix = mean(all_epochs_corrected, 3);

% Create ERP time vector (for plotting X-axis)
erp_time_vec = linspace(PRE_EVENT_SEC, POST_EVENT_SEC, size(ERP_matrix, 1));


%% 4. PLOT KEY ERP COMPONENTS (Cz/Pz Comparison)

% --- Find key channels (using generic indices for simulation) ---
% Cz proxy (Central/Frontal for CNV): Channel 5
channel_cz_idx = 48; 
% Pz proxy (Parietal-Central for P3): Channel 10 
channel_pz_idx = 31; 

figure('Units', 'normalized', 'Position', [0.1 0.1 0.7 0.7]); % Large Figure

% --- Subplot 1: P3 Component Check (Target Detection) ---
subplot(2, 1, 1);
plot(erp_time_vec, ERP_matrix(:, channel_pz_idx), 'r', 'LineWidth', 2);
hold on;
line([0 0], ylim, 'Color', 'k', 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
xlim([PRE_EVENT_SEC, POST_EVENT_SEC]);
title(['P3 Component Check (Target Response) - Channel Pz Proxy (' num2str(channel_pz_idx) ')']);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
text(0.3, max(ylim)*0.7, 'Expected P3 Peak: 300-600ms', 'FontSize', 9);
set(gca, 'YDir', 'reverse'); % Negative Upward
hold off;

% --- Subplot 2: CNV Component Check (Temporal Preparation) ---
subplot(2, 1, 2);
plot(erp_time_vec, ERP_matrix(:, channel_cz_idx), 'b', 'LineWidth', 2);
hold on;
line([0 0], ylim, 'Color', 'k', 'LineStyle', '--');
line(xlim, [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
xlim([PRE_EVENT_SEC, POST_EVENT_SEC]);
title(['CNV Component Check (Temporal Expectation) - Channel Cz Proxy (' num2str(channel_cz_idx) ')']);
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
text(-0.5, min(ylim)*0.7, 'Expected CNV: Slow Negative Deflection Pre-Zero', 'FontSize', 9);
set(gca, 'YDir', 'reverse'); % Negative Upward
hold off;


%% 5. SAVE FINAL FIGURE AND DATA

% Save the full ERP matrix for Monday morning analysis
save(fullfile('Results', 'final_ERP_matrix.mat'), 'ERP_matrix', 'erp_time_vec', '-v7.3');
disp('ERP matrix saved to 03_RESULTS/final_ERP_matrix.mat');

% Save the comparison figure as a PNG
figure_filename_png = fullfile('Results', 'CNV_P3_Comparison.png');
print(gcf, figure_filename_png, '-dpng', '-r300'); 
disp(['Comparison figure saved to 03_RESULTS/CNV_P3_Comparison.png']);

disp('--- ERP Speedrun Complete. Data and Figure Saved. ---');
