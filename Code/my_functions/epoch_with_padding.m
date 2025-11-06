function all_epochs_padded = epoch_with_padding(data_matrix, Fs, n_channels, n_trials, epoch_latencies, total_epoch_samples, pre_samples, alpha_freq_range)
% EPOCH_DATA_WITH_PADDING Slices continuous EEG data, adding buffer samples 
% to the beginning and end of each trial to prevent edge artifacts during filtering.
%
% INPUTS:
%   data_matrix: Continuous data [Time x Channels].
%   epoch_latencies: Sample indices for trial starts (N_trials x 1).
%   Fs: Sampling rate.
%   ... (other parameters for sizing and padding)
%
% OUTPUTS:
%   all_epochs_padded: The final 3D padded epoch array [Padded Time x Channels x Trials].

% --- 1. CONFIGURATION FOR PADDING ---
LOWEST_FREQ = alpha_freq_range(1); 
PAD_CYCLES = 2; 
PAD_SAMPLES = round(PAD_CYCLES * Fs / LOWEST_FREQ);
disp(['Padding ' num2str(PAD_SAMPLES) ' samples per side for edge artifact control.']);


% Define the PADDED size
total_epoch_samples_padded = total_epoch_samples + 2 * PAD_SAMPLES;
pre_samples_padded = pre_samples + PAD_SAMPLES; % Shifted 0ms point


% --- 2. EPOCHING WITH PADDING ---
disp('Starting padded epoching...');

all_epochs_padded = zeros(total_epoch_samples_padded, n_channels, n_trials); 

for trial_idx = 1:n_trials
    center_idx = epoch_latencies(trial_idx); 
    
    % Calculate padded indices
    start_idx = center_idx - pre_samples_padded; 
    end_idx = start_idx + total_epoch_samples_padded - 1; 
    
    % Safety check: if indices are out of bounds (near start/end of file)
    if start_idx >= 1 && end_idx <= size(data_matrix, 1)
        all_epochs_padded(:, :, trial_idx) = data_matrix(start_idx:end_idx, :);
    else
        % --- Zero-pad trials that are too close to the file edges (robust fallback) ---
        disp(['Warning: Trial ' num2str(trial_idx) ' out of bounds. Zero padding applied.']);
        segment = zeros(total_epoch_samples_padded, n_channels);
        
        % Calculate where to place the data in the padded segment
        data_start = max(1, start_idx);
        data_end = min(size(data_matrix, 1), end_idx);
        
        pad_start = data_start - start_idx + 1;
        pad_end = data_end - start_idx + 1;
        
        segment(pad_start:pad_end, :) = data_matrix(data_start:data_end, :);
        all_epochs_padded(:, :, trial_idx) = segment;
    end
end
disp('Padded epoching complete.');
end