function [BetaMap, PValueMap, TimeBins, FreqBins] = tf_regression_map(MasterTable, Fs, time_window_sec, freq_range_hz, N_TIME_BINS, N_FREQ_BINS, outcome)
% COMPUTE_TF_REGRESSION_MAP Fits a GLMM for every time-frequency bin and returns a map of Beta coefficients.
% This is the functional equivalent of the Time-Frequency Regression Map.
%
% INPUTS (Assumed): MasterTable (with AlphaAmplitude cell array), Fs, and analysis windows.

disp('--- Starting Time-Frequency GLMM Map (TFRM) ---');

% --- 1. CONFIGURATION AND INITIALIZATION ---
PRE_EVENT_SEC = 0.500; % Assumed baseline start from your pipeline
N_TRIALS_TOTAL = size(MasterTable, 1);

% --- A. Define Time and Frequency Bins ---
time_start_sec = time_window_sec(1); % -0.200s
time_end_sec = time_window_sec(2);   % 0s

% Calculate sample indices and bin sizes
time_zero_sample = round(PRE_EVENT_SEC * Fs); 
start_sample = time_zero_sample + round(time_start_sec * Fs); 
end_sample = time_zero_sample + round(time_end_sec * Fs);
total_window_samples = end_sample - start_sample + 1;

samples_per_time_bin = floor(total_window_samples / N_TIME_BINS);
freqs_per_bin = (freq_range_hz(2) - freq_range_hz(1)) / N_FREQ_BINS;

% Initialize output matrices
BetaMap = zeros(N_TIME_BINS, N_FREQ_BINS);
PValueMap = zeros(N_TIME_BINS, N_FREQ_BINS);

% --- B. Extract the Master 4D Data Blob (Cell Column) ---
% We assume the 3D data cube [Time x Freqs x 1] is in the table.
% This must be unpacked into a single numeric array for easy indexing.
try
    % Convert the cell column back to a numeric array [Time x Freq x Trial]
    alpha_data_cube = cat(3, MasterTable.AlphaAmplitude{:}); 
    disp(['Unpacked 3D data cube size: ' num2str(size(alpha_data_cube))]);
catch ME
    error('Could not unpack AlphaAmplitude cell array. Check final column type in MasterTable.');
end

%Calculate the FINAL mean across Time, Freq, and Channels
% The result is a 1x1x1xN_Trials array, which needs to be squeezed.
% We are averaging over Dimension 1 (Time), Dimension 2 (Freq), and Dimension 3 (Channels).
sliced_power = alpha_data_cube(start_sample:end_sample, :, :); 
% Average raw power across Time (1), Freq (2), and Channels (3)
mean_alpha_predictor = squeeze(mean(mean(sliced_power, 1), 2));

T_temp = MasterTable;
T_temp.AlphaPower = double(mean_alpha_predictor(:)); % Predictor X1
T_temp.StimIntensityZ = double(T_temp.StimIntensityZ); % X2 (already Z-scored)

if strcmp(outcome, 'Subjective')
   model_formula = 'SubjectiveOutcome ~ AlphaPower + (1|SubjectID)';
else
   model_formula = 'ObjectiveOutcome ~ AlphaPower + (1|SubjectID)';
end

% Fit GLMM (Only need the final P-value and Beta)
glme = fitglme(T_temp, model_formula, ...
                       'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace');
disp(glme.Coefficients);


% --- 2. THE BETA-MAP REGRESSION LOOP (Time x Frequency) ---

for f_bin = 1:N_FREQ_BINS
    
    % Define the current frequency range indices
    %freq_start_idx = round(freq_range_hz(1) + (f_bin - 1) * freqs_per_bin);
    %freq_end_idx = round(freq_range_hz(1) + f_bin * freqs_per_bin);
    %disp(freq_start_idx:freq_end_idx)
    
    for t_bin = 1:N_TIME_BINS
        
        % A. Define the current time bin samples
        time_bin_start_sample = start_sample + (t_bin - 1) * samples_per_time_bin;
        time_bin_end_sample = time_bin_start_sample + samples_per_time_bin - 1;
        
        % B. Feature Extraction: Slice and Average for this specific [t, f] bin
        % Data is [Time x Freqs x Channels x Trials]
        %sliced_power = alpha_data_cube(time_bin_start_sample:time_bin_end_sample, ...
                                       %freq_start_idx:freq_end_idx, :);
        sliced_power = alpha_data_cube(time_bin_start_sample:time_bin_end_sample, f_bin, :);
        % Average raw power across Time (1), Freq (2), and Channels (3)
        current_alpha_predictor = squeeze(mean(mean(sliced_power, 1), 2));
        
        % C. Prepare Temporary Table (Single Predictor + Controls)
        % Create a temporary table by replacing the Z-scored AlphaPower column
        T_temp = MasterTable;
        T_temp.AlphaPower = double(current_alpha_predictor(:)); % Predictor X1
        T_temp.StimIntensitZ = double(T_temp.StimIntensityZ); % X2 (already Z-scored)


        % D. Fit GLMM (Only need the final P-value and Beta)
        glme = fitglme(T_temp, model_formula, ...
                       'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace', 'Verbose', 0);
        
        % E. Extract Results (Find the AlphaPower Main Effect Beta/P-Value)
        T_coeff = glme.Coefficients;
        alpha_row = strcmp(T_coeff.Name, 'AlphaPower');

        BetaMap(t_bin, f_bin) = T_coeff.Estimate(alpha_row);
        PValueMap(t_bin, f_bin) = T_coeff.pValue(alpha_row);
        
    end
end

% --- 3. Finalize Output Axes ---
TimeBins = linspace(time_start_sec, time_end_sec, N_TIME_BINS);
FreqBins = linspace(freq_range_hz(1), freq_range_hz(2), N_FREQ_BINS);

disp('Time-Frequency Regression Map computation finished.');

figure('Name', 'Regression Beta Map');

% --- 1. Plot the Beta Map using imagesc ---
% We plot the BetaMap directly. No need for transpose if the TFR function
% outputted [Time x Freqs]. We rely on the axes labels to handle orientation.
imagesc(TimeBins, FreqBins, BetaMap'); 
% NOTE: Assuming BetaMap needs to be transposed (') to align [Freqs x Time] for imagesc.
axis xy; % CRITICAL: Flips the Y-axis so low frequencies are at the bottom.
% Reverse X-axis direction
%set(gca, 'YDir', 'reverse');
colorbar;

% --- Set Color Limits (Crucial for symmetrical interpretation) ---
% Find the max absolute value to center the color map symmetrically around zero.
max_abs_beta = max(abs(BetaMap(:)));
clim([-max_abs_beta, max_abs_beta]); 
colormap('jet'); % Use a high-contrast diverging colormap (jet or parula)
hold on; 

% --- 1. Create a logical mask for significance (p < 0.05) ---
significant_mask = (PValueMap' < 0.1); % 0.025 because our hypothesis is one-tailed 

% 2. Plot the contour lines using the mask
[~, h_contour] = contour(TimeBins, FreqBins, significant_mask, [0.5 0.5], 'LineWidth', 3, 'LineColor', 'w', 'LineStyle', ':');

% --- Aesthetics and Markers ---
hold on;
line([0 0], ylim, 'Color', 'w', 'LineWidth', 2, 'LineStyle', '--'); % Vertical line at stimulus onset (t=0)
%set(gca, 'XDir', 'reverse'); % Sets Time (X-axis) to flow backward (Standard ERP)
title(['Time-Frequency Beta Map (Alpha Power Predicts ' outcome ' Outcome)'], 'FontSize', 10);
xlabel('Time relative to stimulus (s)');
ylabel('Frequency (Hz)');
grid on;
hold off;
end