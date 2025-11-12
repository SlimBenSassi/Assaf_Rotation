function [accuracy_per_bin] = accuracy_per_bin(data, actual_outcome, predicted_outcome, threshold)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if threshold < 0.5
    baseline_accuracy = 1 - threshold;
else
    baseline_accuracy = threshold;
end

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = data.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin = zeros(N_BINS, 1);

% --- 3. Loop Through Each Unique Intensity Level ---
for i = 1:N_BINS
    current_level = intensity_levels(i);
    
    % Find all trials belonging to this exact intensity level
    idx_level = (intensity_data_raw == current_level);
    
    % Extract the predictions and actual outcomes for this level
    actual_level = actual_outcome(idx_level);
    predicted_level = predicted_outcome(idx_level);
    
    N_level_trials = length(actual_level);
    %disp(N_level_trials);
    
    if N_level_trials > 0
        % Calculate accuracy: (Correct Predictions) / (Total Trials in Bin)
        accuracy_per_bin(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin);
h.FaceColor = [0.1 0.4 0.7]; 

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x)], min(intensity_levels):max(intensity_levels), 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [baseline_accuracy baseline_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;

end