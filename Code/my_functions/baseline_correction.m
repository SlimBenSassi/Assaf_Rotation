function power_db = baseline_correction(power_cube, Fs, pre_event_sec, baseline_start_sec, baseline_end_sec)
% NORMALIZE_POWER_DB Converts raw power into decibels (dB) relative to a baseline.
%
% INPUTS:
%   power_cube: 3D array [Time x Freqs x Channels] for a single trial,
%   handles 2D Time x Freqs too
%   Fs: Sampling rate.
%   pre_event_sec: Duration of the pre-event window (e.g., 0.5s).

% --- 1. Define Baseline Samples ---

time_zero_sample = round(pre_event_sec * Fs); 

baseline_start_sample = time_zero_sample + round(baseline_start_sec * Fs); 
baseline_end_sample = time_zero_sample + round(baseline_end_sec * Fs);


% --- 2. Calculate Baseline Mean (across Time Dimension) ---
% The baseline is averaged only across the time dimension (Dim 1).
% Result is [1 x Freqs x Channels] (the baseline average for each Freq/Chan combination).
if length(size(power_cube)) == 4
    mean_baseline_power = mean(power_cube(baseline_start_sample:baseline_end_sample, :, :), 1);
else
    mean_baseline_power = mean(power_cube(baseline_start_sample:baseline_end_sample, :), 1);
end

% --- 3. Shield against zero/negative power (Crucial for log) ---
epsilon = eps;
mean_baseline_power(mean_baseline_power <= epsilon) = epsilon; 
power_cube(power_cube <= epsilon) = epsilon; 

% --- 4. Apply Log-Ratio Normalization (dB Conversion) ---
% dB Power = 10 * log10(Power / BaselinePower)
power_db = 20 * log10(bsxfun(@rdivide, power_cube, mean_baseline_power)); %20 because we are actually using amplitude and not power

end