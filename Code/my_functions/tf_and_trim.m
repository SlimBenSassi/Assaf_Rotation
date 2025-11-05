function alpha_power_envelope = tf_and_trim(all_epochs_alpha, Fs, alpha_freq_range)
% Summary of this function goes here
%   Detailed explanation goes here
% runs tfHilbert (lab's function) then trimms padded data


% --- 2. TFhilbert Call (CRITICAL DIMENSIONAL SWAP) ---
% The function processes trials sequentially. We must process each channel across all trials.
% We reshape the 3D data into [Trials*Time x Channels] and then process it, 
% or process trial-by-trial, which is cleaner.

% The function expects [Trials/Channels x Time]. We will call it in a simple loop 
% over the channels, processing time across trials.
dimensions = size(all_epochs_alpha); % assuming t x freq x trial
total_epoch_samples_padded = dimensions(1);
n_channels = dimensions(2);
n_trials = dimensions(3);

% We analyze only the single Alpha band (8-12 Hz) for this single-trial test.
frequencies_to_analyze = [alpha_freq_range(1): alpha_freq_range(2)]; % Use 10 Hz as the center frequency
filt_width_octaves = 0.2; 
causalFilt = false; % Use zero-phase filtering


% Initialize the final power envelope matrix [Time x Channels x Trials]
alpha_power_envelope = zeros(total_epoch_samples_padded, length(frequencies_to_analyze), n_channels, n_trials);

for ch = 1:n_channels
    % Extract all time courses for the current channel across all trials: [Time x 1 x Trials]
    channel_data = squeeze(all_epochs_alpha(:, ch, :)); 
    
    % Call the lab's function: Input must be [Trials x Time]
    % Transpose the channel_data [Time x Trials] to [Trials x Time]
    [hil_amp, ~] = TFhilbert(frequencies_to_analyze, filt_width_octaves, channel_data', Fs, causalFilt);
    
    % TFhilbert output is [Freqs x Time x Trials]. 
    % We store the power (amp^2) back into the final matrix:
    alpha_power_envelope(:, :, ch, :) = permute(hil_amp, [2 1 3]); %hil_amp.^2 for power, hil_amp for amplitude
end

disp('TFhilbert processing complete. Power envelope calculated.');

% --- TRIMMING STEP (MUST OCCUR AFTER TFhilbert) ---
disp('Trimming edge padding...');

% --- 1. CONFIGURATION FOR PADDING ---
LOWEST_FREQ = alpha_freq_range(1); 
PAD_CYCLES = 2; 
PAD_SAMPLES = round(PAD_CYCLES * Fs / LOWEST_FREQ);

% Define the samples to keep (from sample PAD_SAMPLES + 1 up to the end minus PAD_SAMPLES)
start_trim_idx = PAD_SAMPLES + 1 ;
end_trim_idx = total_epoch_samples_padded - PAD_SAMPLES;

% Slice the power envelope to remove the padding in the Time dimension (Dimension 1)
alpha_power_envelope = alpha_power_envelope(start_trim_idx:end_trim_idx, :, :, :);

% Verify the size is back to normal (total_epoch_samples)
disp(['Final time points after trimming: ' num2str(size(alpha_power_envelope, 1))]);


end