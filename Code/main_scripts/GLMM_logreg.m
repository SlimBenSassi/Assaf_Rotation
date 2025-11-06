%% 1. INITIALIZATION AND LOAD MASTER TABLE

clear; close all; clc
disp('--- Piloting GLMM Prediction Model ---');

%--- Configuration ---
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

master_table_file = fullfile(filepath, filename);


% Load the MasterTable (contains SubjectID, AlphaPower, StimIntensity, SubjectiveOutcome)
load(master_table_file, 'MasterTable');
disp('Master Table loaded successfully.');

% Ensure data types are correct for the GLMM function
MasterTable.SubjectID = categorical(MasterTable.SubjectID);
MasterTable.SubjectiveOutcome = logical(MasterTable.SubjectiveOutcome); 

% Display total N and the first few rows
disp(['Total trials in model: ' num2str(size(MasterTable, 1))]);
head(MasterTable);


%% OUTLIER REJECTION (Filtering Data Before GLMM)

disp('Checking and removing global outliers (AlphaPower > +/- 3 SD)...');

% --- 1. Calculate Global Statistics (Across ALL trials/subjects) ---
global_mean = mean(MasterTable.AlphaPower_Avg_200ms);
global_std = std(MasterTable.AlphaPower_Avg_200ms);
threshold = 3; % The standard 3 SD limit

% --- 2. Create Logical Mask for Outliers ---
% Find indices where AlphaPower is outside the [mean - 3*std, mean + 3*std] range
is_outlier = (MasterTable.AlphaPower_Avg_200ms > global_mean + threshold * global_std) | ...
             (MasterTable.AlphaPower_Avg_200ms < global_mean - threshold * global_std);

% --- 3. Filter the MasterTable ---
N_original = size(MasterTable, 1);
MasterTable(is_outlier, :) = []; % Remove all rows (trials) marked as outliers

N_removed = N_original - size(MasterTable, 1);
disp(['Total trials removed due to outliers: ' num2str(N_removed)]);
disp(['Remaining trials for GLMM: ' num2str(size(MasterTable, 1))]);


%% A. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) SIMPLE

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_200ms + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_200ms= double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);


% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_simple = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');

disp(glme_simple.Coefficients);


%% A.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_200ms);
alpha_std = std(MasterTable.AlphaPower_Avg_200ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_simple.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_200ms'));


% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_alpha * alpha_plot_range ;
       
% 3. Convert Log-Odds to Probability (The S-Curve Transformation)
% Probability = 1 / (1 + exp(-LogOdds))
probability_seen = 1 ./ (1 + exp(-log_odds));

% --- 4. Plotting ---
figure('Units', 'normalized', 'Position', [0.2 0.2 0.4 0.6]);
plot(alpha_plot_range, probability_seen, 'b', 'LineWidth', 3);
hold on;
plot(alpha_plot_range, probability_seen>=0.5, 'r', 'LineWidth', 3);
hold on;


% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% A.2 PREDICTION ACCURACY AND CLASSIFICATION

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(glme_simple, MasterTable);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= 0.5); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(MasterTable.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy, 0.5]; % Compare Accuracy to Chance (0.5)

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;


%% A.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_simple, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
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
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;

%% B. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_200ms * StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_200ms = double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_complex = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');

disp(glme_complex.Coefficients);


%% B.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_200ms);
alpha_std = std(MasterTable.AlphaPower_Avg_200ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_complex.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_200ms'));
beta_intensity = T.Estimate(strcmp(T.Name, 'StimIntensity'));
beta_interaction = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_200ms:StimIntensity'));

% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_alpha * alpha_plot_range + ...
           beta_intensity * fixed_intensity + ...
           beta_interaction * (alpha_plot_range * fixed_intensity);
       
% 3. Convert Log-Odds to Probability (The S-Curve Transformation)
% Probability = 1 / (1 + exp(-LogOdds))
probability_seen = 1 ./ (1 + exp(-log_odds));

% --- 4. Plotting ---
figure('Units', 'normalized', 'Position', [0.2 0.2 0.4 0.6]);
plot(alpha_plot_range, probability_seen, 'b', 'LineWidth', 3);
hold on;
plot(alpha_plot_range, probability_seen>=0.5, 'r', 'LineWidth', 3);
hold on;

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;


%% B.2 PREDICTION ACCURACY AND CLASSIFICATION

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(glme_complex, MasterTable);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= 0.5); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(MasterTable.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy, 0.5]; % Compare Accuracy to Chance (0.5)

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;


%% B.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_complex, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
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
    disp(N_level_trials);
    
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
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level)');
ylabel('Accuracy (Proportion Correct)');
grid on;

%% C. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX 2 : with random slopes

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1 + AlphaPower|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_200ms * StimIntensity + (1 + AlphaPower_Avg_200ms | SubjectID)';


% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_200ms = double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_complex2 = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');

disp(glme_complex2.Coefficients);


%% C.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_200ms);
alpha_std = std(MasterTable.AlphaPower_Avg_200ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_complex2.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_200ms'));
beta_intensity = T.Estimate(strcmp(T.Name, 'StimIntensity'));
beta_interaction = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_200ms:StimIntensity'));

% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_alpha * alpha_plot_range + ...
           beta_intensity * fixed_intensity + ...
           beta_interaction * (alpha_plot_range * fixed_intensity);
       
% 3. Convert Log-Odds to Probability (The S-Curve Transformation)
% Probability = 1 / (1 + exp(-LogOdds))
probability_seen = 1 ./ (1 + exp(-log_odds));

% --- 4. Plotting ---
figure('Units', 'normalized', 'Position', [0.2 0.2 0.4 0.6]);
plot(alpha_plot_range, probability_seen, 'b', 'LineWidth', 3);
hold on;
plot(alpha_plot_range, probability_seen>=0.5, 'r', 'LineWidth', 3);
hold on;

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% C.2 PREDICTION ACCURACY AND CLASSIFICATION

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(glme_complex2, MasterTable);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= 0.5); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(MasterTable.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy, 0.5]; % Compare Accuracy to Chance (0.5)

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;


%% C.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_complex2, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
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
    disp(N_level_trials);
    
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
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level)');
ylabel('Accuracy (Proportion Correct)');
grid on;

%% D. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM): CONTROL

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_200ms= double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);


% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_control = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');

disp(glme_simple.Coefficients);


%% D.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.StimIntensity);
alpha_std = std(MasterTable.StimIntensity);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
%fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_control.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_stim = T.Estimate(strcmp(T.Name, 'StimIntensity'));


% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_stim * alpha_plot_range ;
       
% 3. Convert Log-Odds to Probability (The S-Curve Transformation)
% Probability = 1 / (1 + exp(-LogOdds))
probability_seen = 1 ./ (1 + exp(-log_odds));

% --- 4. Plotting ---
figure('Units', 'normalized', 'Position', [0.2 0.2 0.4 0.6]);
plot(alpha_plot_range, probability_seen, 'b', 'LineWidth', 3);
hold on;
plot(alpha_plot_range, probability_seen>=0.5, 'r', 'LineWidth', 3);
hold on;


% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% D.2 PREDICTION ACCURACY AND CLASSIFICATION

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(glme_control, MasterTable);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= 0.5); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(MasterTable.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy, 0.5]; % Compare Accuracy to Chance (0.5)

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;


%% D.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_control, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
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
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;

