% %% 2. ALPHA POWER EXTRACTION (Filter, Hilbert, Extract Envelope) 
% 
% % --- Configuration ---
% PRE_EVENT_SEC = 0.9; 
% POST_EVENT_SEC = 0.3;
% total_epoch_samples = round((PRE_EVENT_SEC + POST_EVENT_SEC) * Fs);
% erp_time_vec = (0:total_epoch_samples - 1) / Fs - PRE_EVENT_SEC;
% 
% % --- 1. Bandpass Filter (Alpha 8-12 Hz) ---
% filter_order = 3; 
% [b, a] = butter(filter_order, alpha_freq_range / (Fs/2));
% data_filtered_alpha = filtfilt(b, a, double(data_matrix)); 
% 
% % --- 2. Epoching the Filtered Data ---
% pre_samples = round(PRE_EVENT_SEC * Fs);
% all_epochs_alpha = zeros(total_epoch_samples, n_channels, n_trials);
% 
% for trial_idx = 1:n_trials
%     center_idx = epoch_latencies(trial_idx); 
% 
%     % The correct calculation for start and end indices:
%     start_idx = center_idx - pre_samples;
%     end_idx = start_idx + total_epoch_samples - 1; % Total length should be total_epoch_samples
% 
%     if start_idx >= 1 && end_idx <= size(data_matrix, 1)
% 
%         % Extract the segment
%         % The right side is [end_idx - start_idx + 1] samples long.
%         % This must equal total_epoch_samples (819).
%         all_epochs_alpha(:, :, trial_idx) = data_filtered_alpha(start_idx:end_idx, :);
%     end
% end
% 
% disp('Alpha-band epoching complete.');
% 
% 
% % --- 3. Hilbert Transform and Power Calculation ---
% alpha_analytic = hilbert(all_epochs_alpha);
% % Power is magnitude squared, often log-transformed (we use simple power here)
% alpha_power_envelope = abs(alpha_analytic).^2;