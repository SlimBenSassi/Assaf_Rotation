function alpha_power_envelope = tf_and_trim_baseline_fixedpad(baseline_cell, Fs, alpha_freq_range, PAD_SAMPLES)
% TF + TRIM for baseline segments (cell input)
% baseline_cell:  cell array, each cell = [Time x Channels]
% PAD_SAMPLES:    number of samples to trim from start + end (fixed)

% ------------------------------
% 1. Convert cell → 3D matrix
% ------------------------------
n_trials = numel(baseline_cell);
example = baseline_cell{1};
n_time = size(example,1);
n_channels = size(example,2);

all_epochs = zeros(n_time, n_channels, n_trials);
for t = 1:n_trials
    all_epochs(:,:,t) = baseline_cell{t};
end

% Now all_epochs is [time x channels x trials]

disp(['Running TFhilbert on baseline: ' num2str(n_time) ' samples, ' ...
      num2str(n_channels) ' channels, ' num2str(n_trials) ' trials']);

% ------------------------------
% 2. TFhilbert (same as your code)
% ------------------------------
frequencies_to_analyze = alpha_freq_range(1):alpha_freq_range(2);
filt_width_octaves = 0.2;
causalFilt = false;

alpha_power_envelope = zeros(n_time, length(frequencies_to_analyze), n_channels, n_trials);

for ch = 1:n_channels
    channel_data = squeeze(all_epochs(:,ch,:));    % [time x trials]
    [hil_amp, ~] = TFhilbert(frequencies_to_analyze, filt_width_octaves, channel_data', Fs, causalFilt);
    alpha_power_envelope(:,:,ch,:) = permute(hil_amp, [2 1 3]);
end

% ------------------------------
% 3. FIXED TRIMMING
% ------------------------------
start_idx = PAD_SAMPLES + 1;
end_idx   = n_time - PAD_SAMPLES;

alpha_power_envelope = alpha_power_envelope(start_idx:end_idx, :, :, :);

disp(['Trimmed baseline: final samples = ' num2str(size(alpha_power_envelope,1))]);

end
