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


%% JUST TO CHECK

MasterTable = MasterTable(MasterTable.StimIntensityRaw <= 6, :); 
MasterTable = MasterTable(MasterTable.StimIntensityRaw >= 4, :); 


disp('Filtered: Removed all trials with raw intensity > 5 and < 3 to focus analysis on threshold.');

%% OUTLIER REJECTION (Filtering Data Before GLMM)

disp('Checking and removing global outliers (AlphaPower > +/- 3 SD)...');

% --- 1. Calculate Global Statistics (Across ALL trials/subjects) ---
global_mean = mean(MasterTable.AlphaPower_Avg_100ms);
global_std = std(MasterTable.AlphaPower_Avg_100ms);
threshold = 3; % The standard 3 SD limit

% --- 2. Create Logical Mask for Outliers ---
% Find indices where AlphaPower is outside the [mean - 3*std, mean + 3*std] range
is_outlier = (MasterTable.AlphaPower_Avg_100ms > global_mean + threshold * global_std) | ...
             (MasterTable.AlphaPower_Avg_100ms < global_mean - threshold * global_std);

% --- 3. Filter the MasterTable ---
N_original = size(MasterTable, 1);
MasterTable(is_outlier, :) = []; % Remove all rows (trials) marked as outliers

N_removed = N_original - size(MasterTable, 1);
disp(['Total trials removed due to outliers: ' num2str(N_removed)]);
disp(['Remaining trials for GLMM: ' num2str(size(MasterTable, 1))]);


%% Baseline Accuracy 

% 1. Count the number of 'Seen' trials (1) and 'Unseen' trials (0)
N_Total = length(MasterTable.SubjectiveOutcome);
N_seen = sum(MasterTable.SubjectiveOutcome == 1);
N_unseen = sum(MasterTable.SubjectiveOutcome == 0);

% 2. Calculate the proportion of the largest class (the maximum chance performance)
max_chance_accuracy = max(N_seen, N_unseen) / N_Total;

if N_seen < N_unseen
    threshold = 1-max_chance_accuracy;  
else 
    threshold = max_chance_accuracy;
end

disp(['Baseline Performance (Max Chance Guess): ' num2str(max_chance_accuracy * 100, 3) '%']);

% --- Now update the plotting line to use this new variable ---

% Original Chance Line: line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
%line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);

%% A. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) SIMPLE

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms= double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);


% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_simple = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_simple.Coefficients);


%% A.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_100ms);
alpha_std = std(MasterTable.AlphaPower_Avg_100ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_simple.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms'));


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
plot(alpha_plot_range, probability_seen>=threshold, 'r', 'LineWidth', 3);
hold on;


% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_100ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% A.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome, predicted_outcome, accuracy1] = predict_outcome(MasterTable, glme_simple, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome, predicted_outcome);

%% A.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_simple, MasterTable) >= threshold); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin1 = zeros(N_BINS, 1);

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
        accuracy_per_bin1(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin1(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin1);
h.FaceColor = [0.1 0.4 0.7]; 

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x)], min(intensity_levels):max(intensity_levels), 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;


%% Permutation Test
tic

N_PERMUTATIONS = 10; % Standard number of permutations
null_accuracies = zeros(N_PERMUTATIONS, 1);
MasterTable_original = MasterTable; % Keep a copy of the original data

disp(['Running Permutation Test (N = ' num2str(N_PERMUTATIONS) ')...']);

for i = 1:N_PERMUTATIONS
    
    % % 1. SHUFFLE THE PREDICTOR (Breaks the link between Alpha and Outcome)
    % % We shuffle the AlphaPower column (X) while keeping the Outcome (Y) fixed.
    shuffled_alpha_power = MasterTable_original.AlphaPower_Avg_100ms(randperm(size(MasterTable_original, 1)));
    % 
    % % 2. Create the Null Table
    MasterTable_null = MasterTable_original;
    MasterTable_null.AlphaPower_Avg_100ms = shuffled_alpha_power; % Inject shuffled predictor

    % 1. SHUFFLE THE OUTCOME (The clean fix for complex models)
    % Create a randomly permuted index list based on the total number of trials.
    %shuffled_indices = randperm(size(MasterTable_original, 1));
    
    % 2. Create the Null Table by injecting the shuffled outcome (Y)
    %MasterTable_null = MasterTable_original;
    % We shuffle the SubjectiveOutcome column (Y) while keeping X1 and X2 in place.
    %MasterTable_null.SubjectiveOutcome = MasterTable_null.SubjectiveOutcome(shuffled_indices);
    
    % 3. RUN THE NULL MODEL (Recalculate the GLMM on shuffled data)
    % IMPORTANT: This step is slow!
    glme_null = fitglme(MasterTable_null, model_formula, ...
                        'Distribution', 'Binomial', 'Link', 'logit', 'Verbose', 0, ...
                        'FitMethod','Laplace');
    
    % 4. GET NULL ACCURACY
    predicted_outcome_null = (predict(glme_null, MasterTable_null) >= threshold); 
    actual_outcome_null = double(MasterTable_null.SubjectiveOutcome);
    
    null_accuracies(i) = sum(predicted_outcome_null == actual_outcome_null) / length(actual_outcome_null);
end

% --- Final Significance Check ---
real_accuracy = accuracy1; % Use your calculated accuracy
p_value_permutation = sum(null_accuracies >= real_accuracy) / N_PERMUTATIONS;

disp('--- PERMUTATION TEST RESULT ---');
disp(['Real Accuracy: ' num2str(real_accuracy * 100, 3) '%']);
disp(['95th Percentile of Null (Chance Threshold): ' num2str(prctile(null_accuracies, 95) * 100, 3) '%']);
disp(['Permutation P-value (P_perm): ' num2str(p_value_permutation, 4)]);

toc

%% A.5 PLOT PERMUTATION TEST NULL DISTRIBUTION

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

% --- 1. Plot the Null Distribution (Histogram) ---
h = histogram(null_accuracies, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); 
hold on;

% --- 2. Define Critical Thresholds ---
% Calculate the 95th percentile (the significance cut-off)
chance_threshold_95 = prctile(null_accuracies, 95); 

% --- 3. Add Vertical Lines ---
% A. Add the line for your actual, achieved accuracy (62.34%)
line([real_accuracy real_accuracy], ylim, 'Color', 'b', 'LineWidth', 3, 'DisplayName', 'Actual Model Accuracy');

% B. Add the line for the 95th percentile (the chance boundary)
line([chance_threshold_95 chance_threshold_95], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', '95% Significance Threshold');

% --- 4. Aesthetics ---
title('Model Performance vs. Null Distribution (Permutation Test)', 'FontSize', 14);
xlabel('Prediction Accuracy (Proportion Correct)');
ylabel('Frequency (N of Null Models)');
legend('show', 'Location', 'NorthWest');
hold off;



%% B. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms * StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms = double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_alpha_stim_inter = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_alpha_stim_inter.Coefficients);


%% B.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_100ms);
alpha_std = std(MasterTable.AlphaPower_Avg_100ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_alpha_stim_inter.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms'));
beta_intensity = T.Estimate(strcmp(T.Name, 'StimIntensity'));
beta_interaction = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms:StimIntensity'));

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
plot(alpha_plot_range, probability_seen>=threshold, 'r', 'LineWidth', 3);
hold on;

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_100ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;


%% B.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome, predicted_outcome, accuracy2] = predict_outcome(MasterTable, glme_alpha_stim_inter, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome, predicted_outcome);


%% B.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_alpha_stim_inter, MasterTable) >= threshold); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin2 = zeros(N_BINS, 1);

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
        accuracy_per_bin2(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin2(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin2);
h.FaceColor = [0.1 0.4 0.7]; 

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;


%% Permutation Test
tic

N_PERMUTATIONS = 10; % Standard number of permutations
null_accuracies = zeros(N_PERMUTATIONS, 1);
MasterTable_original = MasterTable; % Keep a copy of the original data

disp(['Running Permutation Test (N = ' num2str(N_PERMUTATIONS) ')...']);

for i = 1:N_PERMUTATIONS
    
    % 1. SHUFFLE THE OUTCOME (The clean fix for complex models)
    % Create a randomly permuted index list based on the total number of trials.
    shuffled_indices = randperm(size(MasterTable_original, 1));
    
    % 2. Create the Null Table by injecting the shuffled outcome (Y)
    MasterTable_null = MasterTable_original;
    % We shuffle the SubjectiveOutcome column (Y) while keeping X1 and X2 in place.
    MasterTable_null.SubjectiveOutcome = MasterTable_null.SubjectiveOutcome(shuffled_indices);
    
    % 3. RUN THE NULL MODEL (Recalculate the GLMM on shuffled data)
    % IMPORTANT: This step is slow!
    glme_null = fitglme(MasterTable_null, model_formula, ...
                        'Distribution', 'Binomial', 'Link', 'logit', 'Verbose', 0, ...
                        'FitMethod','Laplace');
    
    % 4. GET NULL ACCURACY
    predicted_outcome_null = (predict(glme_null, MasterTable_null) >= threshold); 
    actual_outcome_null = double(MasterTable_null.SubjectiveOutcome);
    
    null_accuracies(i) = sum(predicted_outcome_null == actual_outcome_null) / length(actual_outcome_null);
end

% --- Final Significance Check ---
real_accuracy = accuracy2; % Use your calculated accuracy
p_value_permutation = sum(null_accuracies >= real_accuracy) / N_PERMUTATIONS;

disp('--- PERMUTATION TEST RESULT ---');
disp(['Real Accuracy: ' num2str(real_accuracy * 100, 3) '%']);
disp(['95th Percentile of Null (Chance Threshold): ' num2str(prctile(null_accuracies, 95) * 100, 3) '%']);
disp(['Permutation P-value (P_perm): ' num2str(p_value_permutation, 4)]);

toc

%% B.5 PLOT PERMUTATION TEST NULL DISTRIBUTION

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

% --- 1. Plot the Null Distribution (Histogram) ---
h = histogram(null_accuracies, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); 
hold on;

% --- 2. Define Critical Thresholds ---
% Calculate the 95th percentile (the significance cut-off)
chance_threshold_95 = prctile(null_accuracies, 95); 

% --- 3. Add Vertical Lines ---
% A. Add the line for your actual, achieved accuracy (62.34%)
line([real_accuracy real_accuracy], ylim, 'Color', 'b', 'LineWidth', 3, 'DisplayName', 'Actual Model Accuracy');

% B. Add the line for the 95th percentile (the chance boundary)
line([chance_threshold_95 chance_threshold_95], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', '95% Significance Threshold');

% --- 4. Aesthetics ---
title('Model Performance vs. Null Distribution (Permutation Test)', 'FontSize', 14);
xlabel('Prediction Accuracy (Proportion Correct)');
ylabel('Frequency (N of Null Models)');
legend('show', 'Location', 'NorthWest');
hold off;

%% C. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX 2 : with random slopes

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1 + AlphaPower|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms * StimIntensity + (1 + AlphaPower_Avg_100ms | SubjectID)';


% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms = double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_alpha_stim_inter_rs = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_alpha_stim_inter_rs.Coefficients);


%% C.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_100ms);
alpha_std = std(MasterTable.AlphaPower_Avg_100ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_alpha_stim_inter_rs.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms'));
beta_intensity = T.Estimate(strcmp(T.Name, 'StimIntensity'));
beta_interaction = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms:StimIntensity'));

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
plot(alpha_plot_range, probability_seen>=threshold, 'r', 'LineWidth', 3);
hold on;

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_100ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% C.2 PREDICTION ACCURACY AND CLASSIFICATION


% Predicting Outcome and Accuracy
[actual_outcome, predicted_outcome, accuracy3] = predict_outcome(MasterTable, glme_alpha_stim_inter_rs, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome, predicted_outcome);


%% C.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_alpha_stim_inter_rs, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin3 = zeros(N_BINS, 1);

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
        accuracy_per_bin3(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin3(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin3);
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
MasterTable.AlphaPower_Avg_100ms= double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);


% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_stim = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_stim.Coefficients);


%% D.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
stim_mean = mean(MasterTable.StimIntensity);
stim_std = std(MasterTable.StimIntensity);
stim_plot_range = linspace(stim_mean - 4*stim_std, stim_mean + 4*stim_std, 100);
stimraw_mean = mean(MasterTable.StimIntensityRaw);
stimraw_std = std(MasterTable.StimIntensityRaw);
stimraw_plot_range = linspace(stimraw_mean - 4*stimraw_std, stimraw_mean + 4*stimraw_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
%fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_stim.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_stim = T.Estimate(strcmp(T.Name, 'StimIntensity'));


% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_stim * stim_plot_range ;
       
% 3. Convert Log-Odds to Probability (The S-Curve Transformation)
% Probability = 1 / (1 + exp(-LogOdds))
probability_seen = 1 ./ (1 + exp(-log_odds));

% --- 4. Plotting ---
figure('Units', 'normalized', 'Position', [0.2 0.2 0.4 0.6]);
plot(stimraw_plot_range, probability_seen, 'b', 'LineWidth', 3);
hold on;
plot(stimraw_plot_range, probability_seen>=threshold, 'r', 'LineWidth', 3);
hold on;


% Add points for the raw data groups (for visual context)
scatter(MasterTable.StimIntensityRaw, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% D.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome, predicted_outcome, accuracy4] = predict_outcome(MasterTable, glme_stim, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome, predicted_outcome);

%% D.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_stim, MasterTable) >= 0.5); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin4 = zeros(N_BINS, 1);

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
        accuracy_per_bin4(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin4(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin4);
h.FaceColor = [0.1 0.4 0.7]; 

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], [min(intensity_levels) :max(intensity_levels)], 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;



%% Permutation Test
tic

N_PERMUTATIONS = 10; % Standard number of permutations
null_accuracies = zeros(N_PERMUTATIONS, 1);
MasterTable_original = MasterTable; % Keep a copy of the original data

disp(['Running Permutation Test (N = ' num2str(N_PERMUTATIONS) ')...']);

for i = 1:N_PERMUTATIONS
    
    % % 1. SHUFFLE THE PREDICTOR (Breaks the link between Alpha and Outcome)
    % % We shuffle the AlphaPower column (X) while keeping the Outcome (Y) fixed.
    shuffled_stim = MasterTable_original.StimIntensity(randperm(size(MasterTable_original, 1)));
    shuffled_stimraw = MasterTable_original.StimIntensityRaw(randperm(size(MasterTable_original, 1)));

    % 
    % % 2. Create the Null Table
    MasterTable_null = MasterTable_original;
    MasterTable_null.StimIntensity = shuffled_stim; % Inject shuffled predictor
    MasterTable_null.StimIntensityRaw = shuffled_stimraw;

    % 1. SHUFFLE THE OUTCOME (The clean fix for complex models)
    % Create a randomly permuted index list based on the total number of trials.
    %shuffled_indices = randperm(size(MasterTable_original, 1));
    
    % 2. Create the Null Table by injecting the shuffled outcome (Y)
    %MasterTable_null = MasterTable_original;
    % We shuffle the SubjectiveOutcome column (Y) while keeping X1 and X2 in place.
    %MasterTable_null.SubjectiveOutcome = MasterTable_null.SubjectiveOutcome(shuffled_indices);
    
    % 3. RUN THE NULL MODEL (Recalculate the GLMM on shuffled data)
    % IMPORTANT: This step is slow!
    glme_null = fitglme(MasterTable_null, model_formula, ...
                        'Distribution', 'Binomial', 'Link', 'logit', 'Verbose', 0, ...
                        'FitMethod','Laplace');
    
    % 4. GET NULL ACCURACY
    predicted_outcome_null = (predict(glme_null, MasterTable_null) >= threshold); 
    actual_outcome_null = double(MasterTable_null.SubjectiveOutcome);
    
    null_accuracies(i) = sum(predicted_outcome_null == actual_outcome_null) / length(actual_outcome_null);
end

% --- Final Significance Check ---
real_accuracy = accuracy4; % Use your calculated accuracy
p_value_permutation = sum(null_accuracies >= real_accuracy) / N_PERMUTATIONS;

disp('--- PERMUTATION TEST RESULT ---');
disp(['Real Accuracy: ' num2str(real_accuracy * 100, 3) '%']);
disp(['95th Percentile of Null (Chance Threshold): ' num2str(prctile(null_accuracies, 95) * 100, 3) '%']);
disp(['Permutation P-value (P_perm): ' num2str(p_value_permutation, 4)]);

toc

%% D.5 PLOT PERMUTATION TEST NULL DISTRIBUTION

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

% --- 1. Plot the Null Distribution (Histogram) ---
h = histogram(null_accuracies, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); 
hold on;

% --- 2. Define Critical Thresholds ---
% Calculate the 95th percentile (the significance cut-off)
chance_threshold_95 = prctile(null_accuracies, 95); 

% --- 3. Add Vertical Lines ---
% A. Add the line for your actual, achieved accuracy (62.34%)
line([real_accuracy real_accuracy], ylim, 'Color', 'b', 'LineWidth', 3, 'DisplayName', 'Actual Model Accuracy');

% B. Add the line for the 95th percentile (the chance boundary)
line([chance_threshold_95 chance_threshold_95], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', '95% Significance Threshold');

% --- 4. Aesthetics ---
title('Model Performance vs. Null Distribution (Permutation Test)', 'FontSize', 14);
xlabel('Prediction Accuracy (Proportion Correct)');
ylabel('Frequency (N of Null Models)');
legend('show', 'Location', 'NorthWest');
hold off;


%% E. BOTH PREDICTORS BUT NO INTERACTION


% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms + StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms = double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_alpha_stim = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod', 'Laplace');

disp('Model fitting complete.');

disp(glme_alpha_stim.Coefficients);

%% E.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_100ms);
alpha_std = std(MasterTable.AlphaPower_Avg_100ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_alpha_stim.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms'));
beta_intensity = T.Estimate(strcmp(T.Name, 'StimIntensity'));
beta_interaction = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms:StimIntensity'));

% 2. Calculate the Log-Odds for the plotting range
% LogOdds = beta0 + beta_alpha*X1 + beta_intensity*X2 + beta_interaction*(X1*X2)
log_odds = beta_0 + ...
           beta_alpha * alpha_plot_range + ...
           beta_intensity * fixed_intensity;
       
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
scatter(MasterTable.AlphaPower_Avg_100ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;


%% E.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome, predicted_outcome, accuracy5] = predict_outcome(MasterTable, glme_alpha_stim, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome, predicted_outcome);


%% E.3. QUICK PLOT: ACCURACY PER INTENSITY LEVEL (BINS)


% --- 1. Get Predicted Probabilities and Actual Outcomes ---
predicted_outcome = (predict(glme_alpha_stim, MasterTable) >= threshold); 
actual_outcome = double(MasterTable.SubjectiveOutcome);

% --- 2. Define Bins and Group Data ---
% CRITICAL: Use the raw intensity values for grouping.
intensity_data_raw = MasterTable.StimIntensityRaw; 
intensity_levels = unique(intensity_data_raw); % Finds the 7 experimental levels (e.g., 2, 3, 4...)
N_BINS = length(intensity_levels);

% Initialize Output
accuracy_per_bin5 = zeros(N_BINS, 1);

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
        accuracy_per_bin5(i) = sum(predicted_level == actual_level) / N_level_trials;
    else
        accuracy_per_bin5(i) = NaN;
    end
end

figure('Units', 'normalized', 'Position', [0.2 0.2 0.45 0.5]);

% Plot the accuracy values
h = bar(accuracy_per_bin5);
h.FaceColor = [0.1 0.4 0.7]; 

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x+1)], 1:N_BINS, 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Model Accuracy by Standardized Stimulus Intensity Bin', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Accuracy (Proportion Correct)');
grid on;


%% Permutation Test
tic

N_PERMUTATIONS = 10; % Standard number of permutations
null_accuracies = zeros(N_PERMUTATIONS, 1);
MasterTable_original = MasterTable; % Keep a copy of the original data

disp(['Running Permutation Test (N = ' num2str(N_PERMUTATIONS) ')...']);

for i = 1:N_PERMUTATIONS
    
    % 1. SHUFFLE THE OUTCOME (The clean fix for complex models)
    % Create a randomly permuted index list based on the total number of trials.
    shuffled_indices = randperm(size(MasterTable_original, 1));
    
    % 2. Create the Null Table by injecting the shuffled outcome (Y)
    MasterTable_null = MasterTable_original;
    % We shuffle the SubjectiveOutcome column (Y) while keeping X1 and X2 in place.
    MasterTable_null.SubjectiveOutcome = MasterTable_null.SubjectiveOutcome(shuffled_indices);
    
    % 3. RUN THE NULL MODEL (Recalculate the GLMM on shuffled data)
    % IMPORTANT: This step is slow!
    glme_null = fitglme(MasterTable_null, model_formula, ...
                        'Distribution', 'Binomial', 'Link', 'logit', 'Verbose', 0, ...
                        'FitMethod','Laplace');
    
    % 4. GET NULL ACCURACY
    predicted_outcome_null = (predict(glme_null, MasterTable_null) >= threshold); 
    actual_outcome_null = double(MasterTable_null.SubjectiveOutcome);
    
    null_accuracies(i) = sum(predicted_outcome_null == actual_outcome_null) / length(actual_outcome_null);
end

% --- Final Significance Check ---
real_accuracy = accuracy5; % Use your calculated accuracy
p_value_permutation = sum(null_accuracies >= real_accuracy) / N_PERMUTATIONS;

disp('--- PERMUTATION TEST RESULT ---');
disp(['Real Accuracy: ' num2str(real_accuracy * 100, 3) '%']);
disp(['95th Percentile of Null (Chance Threshold): ' num2str(prctile(null_accuracies, 95) * 100, 3) '%']);
disp(['Permutation P-value (P_perm): ' num2str(p_value_permutation, 4)]);

toc

%% E.5 PLOT PERMUTATION TEST NULL DISTRIBUTION

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

% --- 1. Plot the Null Distribution (Histogram) ---
h = histogram(null_accuracies, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); 
hold on;

% --- 2. Define Critical Thresholds ---
% Calculate the 95th percentile (the significance cut-off)
chance_threshold_95 = prctile(null_accuracies, 95); 

% --- 3. Add Vertical Lines ---
% A. Add the line for your actual, achieved accuracy (62.34%)
line([real_accuracy real_accuracy], ylim, 'Color', 'b', 'LineWidth', 3, 'DisplayName', 'Actual Model Accuracy');

% B. Add the line for the 95th percentile (the chance boundary)
line([chance_threshold_95 chance_threshold_95], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', '95% Significance Threshold');

% --- 4. Aesthetics ---
title('Model Performance vs. Null Distribution (Permutation Test)', 'FontSize', 14);
xlabel('Prediction Accuracy (Proportion Correct)');
ylabel('Frequency (N of Null Models)');
legend('show', 'Location', 'NorthWest');
hold off;


%% COMPARING MODELS


%% LIKELIHOOD RATIO TEST (Complex vs. Control Model)

disp('--- Running Likelihood Ratio Test (LRT) ---');

% The Control Model (glme_control) is the simpler, nested model (H0).
% The Complex Model (glme_complex) is the reference model (H1).

% --- CRITICAL STEP: Compare the two fitted models ---
% This tests the hypothesis: Is the variance explained by AlphaPower * StimIntensity necessary?
comparison_table = compare(glme_alpha_stim, glme_alpha_stim_inter);

disp('LRT Results:');
disp(comparison_table);

% --- Extract the critical P-value ---
% The LRT P-value (in the last row/column of the table) is the formal test result.
lrt_p_value = comparison_table.pValue(2); % Assumes the comparison P-value is in the second row

disp('------------------------------------------');
disp(['CRITICAL LRT P-VALUE (Proving Alpha is Necessary): ' num2str(lrt_p_value, 4)]);

if lrt_p_value < 0.05
    disp('CONCLUSION: The addition of AlphaPower and its interaction term is STATISTICALLY JUSTIFIED (p < 0.05).');
    disp('Your Alpha feature is necessary for modeling subjective awareness.');
else
    disp('CONCLUSION: AlphaPower does NOT significantly improve the model compared to Stimulus Intensity alone.');
end
disp('------------------------------------------');



%% PLOT OVERALL PREDICTION ACCURACY COMPARISON

% Define the overall accuracy scores for all models
overall_accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5];

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

bar(overall_accuracies, 'FaceColor', [0.1 0.4 0.7]);
hold on;

% Define labels for the X-axis
model_labels = {'alpha', 'alpha+stim+inter', 'alpha+stim+inter+rs', 'stim', 'alpha+stim'};

set(gca, 'XTickLabel', model_labels);
ylim([0.45 1.0]); % Set Y-limit to a meaningful range

line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'Max Chance Baseline');

title('Overall Accuracy of All Fitted GLMMs', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
hold off;


%% PLOT ACCURACY BY CONTRAST LEVEL (Grouped Bars)

% --- Combine all accuracy vectors into a single matrix [N_Bins x N_Models] ---
% Note: All vectors must be the same length (N_Bins = 7 levels).
accuracy_matrix = [accuracy_per_bin1, accuracy_per_bin2, accuracy_per_bin3, accuracy_per_bin4, accuracy_per_bin5];

figure('Units', 'normalized', 'Position', [0.05 0.05 0.8 0.7]);

% Plot grouped bars (each group is a contrast level, and each bar is a model)
bar(accuracy_matrix); 

% Set X-axis labels based on the raw contrast levels (e.g., 2, 3, 4, 5, 6, 7)
% Assuming your raw intensity levels are 1 through 7 and that N_Bins = 7
contrast_labels = arrayfun(@(x) ['Level ' num2str(x)], 1:size(accuracy_matrix, 1), 'UniformOutput', false);

set(gca, 'XTickLabel', contrast_labels);
ylim([0.45 1.0]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '--','DisplayName', 'Max Chance Baseline');

title('Prediction Accuracy by Contrast Level (All Models)', 'FontSize', 14);
xlabel('Stimulus Intensity Level');
ylabel('Accuracy (Proportion Correct)');
legend(model_labels, 'Location', 'NorthEast');
grid on;


%% 6. CROSS-VALIDATION (5-Fold Predictive Accuracy Check)

% --- Configuration ---
K_FOLDS = 10; % Standard number of folds
P_THRESHOLD = threshold; % Unbiased classification threshold

disp('--- Starting 5-Fold Cross-Validation ---');

% 1. Prepare Data Indices (Crucial for reproducible splitting)

% Create a random permutation of the trial indices
random_order = randperm(size(MasterTable, 1)); 

% Shuffle the MasterTable rows (This fixes the ordered data problem)
MasterTable = MasterTable(random_order, :); 

N_TOTAL_TRIALS = size(MasterTable, 1);
CVO = cvpartition(N_TOTAL_TRIALS, 'KFold', K_FOLDS);

% ... (rest of the cross-validation loop continues) ...

% Initialize storage for accuracy results
accuracy_complex = zeros(K_FOLDS, 1);
accuracy_control = zeros(K_FOLDS, 1);

% --- 2. The K-Fold Loop ---
tic
for k = 1:K_FOLDS
    
    % --- A. Split Data ---
    % Get indices for training and testing sets for the current fold
    idx_train = CVO.training(k);
    idx_test = CVO.test(k);
    
    T_train = MasterTable(idx_train, :);
    T_test = MasterTable(idx_test, :);
    
    disp(['Processing Fold ' num2str(k) '/' num2str(K_FOLDS) '...']);
    
    % --- B. TRAIN MODELS (Fit only on the training set) ---
    
    % Model 1: Complex Model (Alpha * Intensity)
    formula_complex = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms + StimIntensity + (1|SubjectID)';
    glme_complex_k = fitglme(T_train, formula_complex, 'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace', 'Verbose', 0);
    
    % Model 2: Control Model (Intensity only)
    formula_control = 'SubjectiveOutcome ~ StimIntensity * AlphaPower_Avg_100ms + (1|SubjectID)';
    glme_control_k = fitglme(T_train, formula_control, 'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace', 'Verbose', 0);

    % --- C. TEST MODELS (Predict on the unseen test set) ---
    
    % Predict probabilities on the test set
    pred_prob_complex = predict(glme_complex_k, T_test);
    pred_prob_control = predict(glme_control_k, T_test);

    % Convert probabilities to binary predictions (0 or 1) using P=0.5 threshold
    pred_outcome_complex = (pred_prob_complex >= P_THRESHOLD);
    pred_outcome_control = (pred_prob_control >= P_THRESHOLD);
    
    % Actual outcome of the test set
    actual_outcome_test = double(T_test.SubjectiveOutcome);

    % --- D. Calculate Accuracy and Store ---
    accuracy_complex(k) = sum(pred_outcome_complex == actual_outcome_test) / length(actual_outcome_test);
    accuracy_control(k) = sum(pred_outcome_control == actual_outcome_test) / length(actual_outcome_test);
    
end % End of K-Fold Loop

toc
disp('Cross-Validation complete.');

% --- 3. FINAL RESULTS ---

Avg_Accuracy_Complex = mean(accuracy_complex);
Avg_Accuracy_Control = mean(accuracy_control);

disp('--- CROSS-VALIDATION SUMMARY ---');
disp(['Complex Model (Alpha*Int) Avg Accuracy: ' num2str(Avg_Accuracy_Complex * 100, 4) '%']);
disp(['Control Model (Int Only) Avg Accuracy:  ' num2str(Avg_Accuracy_Control * 100, 4) '%']);

% Final Statistical Test: Paired T-test on the K-Fold accuracies
[h, p_val_cv, ~, stats] = ttest(accuracy_complex, accuracy_control);

if h == 1 && Avg_Accuracy_Complex > Avg_Accuracy_Control
    disp(['CONCLUSION: Complex model is SIGNIFICANTLY better at predicting UNSEEN data (p = ' num2str(p_val_cv, 4) ').']);
else
    disp('CONCLUSION: No significant predictive difference found between models.');
end




%% F. CROSS-GENERALIZATION: Train on Threshold, Test on All Bins

% --- Action: This demonstrates the specialized nature of the Alpha predictor ---

% 1. Define the Threshold Trials (The "Sweet Spot" where Simple Model peaked)
% Assuming your threshold is intensity levels 3, 4, and 5 (the middle contrast levels)
THRESHOLD_LEVELS = [4 5 6]; 

% Find all trials belonging to the threshold levels
idx_threshold_trials = ismember(MasterTable.StimIntensityRaw, THRESHOLD_LEVELS);

% --- TRAIN MODEL ONLY ON THRESHOLD DATA ---
T_train_threshold = MasterTable(idx_threshold_trials, :);

% Fit the Simple Model (Alpha only) on the threshold data
glme_simple_threshold = fitglme(T_train_threshold, ...
    'SubjectiveOutcome ~ AlphaPower_Avg_100ms + (1|SubjectID)', ...
    'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace');
disp('Model trained exclusively on Threshold Trials.');




%% Baseline Accuracy for train_threshold

% 1. Count the number of 'Seen' trials (1) and 'Unseen' trials (0)
N_Total = length(T_train_threshold.SubjectiveOutcome);
N_seen = sum(T_train_threshold.SubjectiveOutcome == 1);
N_unseen = sum(T_train_threshold.SubjectiveOutcome == 0);

% 2. Calculate the proportion of the largest class (the maximum chance performance)
max_chance_accuracy = max(N_seen, N_unseen) / N_Total;

if N_seen < N_unseen
    threshold = 1-max_chance_accuracy;  
else 
    threshold = max_chance_accuracy;
end

disp(['Baseline Performance (Max Chance Guess): ' num2str(max_chance_accuracy * 100, 3) '%']);

% --- Now update the plotting line to use this new variable ---

% Original Chance Line: line(xlim, [0.5 0.5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
%line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);



%% F.1 PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 4 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_100ms);
alpha_std = std(MasterTable.AlphaPower_Avg_100ms);
alpha_plot_range = linspace(alpha_mean - 4*alpha_std, alpha_mean + 4*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = mean(MasterTable.StimIntensity); 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme_simple_threshold.Coefficients;
beta_0 = T.Estimate(strcmp(T.Name, '(Intercept)'));
beta_alpha = T.Estimate(strcmp(T.Name, 'AlphaPower_Avg_100ms'));


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
plot(alpha_plot_range, probability_seen>=threshold, 'r', 'LineWidth', 3);
hold on;


% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_100ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;

%% F.2 PREDICTION ACCURACY AND CLASSIFICATION

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(glme_simple_threshold, MasterTable);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= threshold); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(MasterTable.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy1 = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy1 * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy1, max_chance_accuracy]; % Compare Accuracy to Chance (0.5)

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;

confusion_matrix(actual_outcome, predicted_outcome);


%% F.3 --- TEST ACCURACY ACROSS ALL 7 RAW INTENSITY BINS ---
% (This reuses the logic from your previous accuracy bar plot)

raw_intensity_levels = unique(MasterTable.StimIntensityRaw);
N_BINS = length(raw_intensity_levels);
accuracy_per_bin_threshold_trained = zeros(N_BINS, 1);

for i = 1:N_BINS
    current_level = raw_intensity_levels(i);
    
    % Find all trials belonging to this level (the full test set)
    idx_test_level = (MasterTable.StimIntensityRaw == current_level);
    T_test_level = MasterTable(idx_test_level, :);
    
    if ~isempty(T_test_level)
        % Predict using the model trained only on the threshold data
        predicted_prob = predict(glme_simple_threshold, T_test_level);
        predicted_outcome = (predicted_prob >= threshold);
        
        actual_outcome = double(T_test_level.SubjectiveOutcome);
        
        accuracy_per_bin_threshold_trained(i) = sum(predicted_outcome == actual_outcome) / size(T_test_level, 1);
    end

    %if current_level == 2
        %histogram(probab)
end


%% 8. VISUALIZATION OF SPECIALIZATION (Bar Plot)

figure('Units', 'normalized', 'Position', [0.2 0.2 0.5 0.5]);
bar(accuracy_per_bin_threshold_trained);

% Aesthetics
labels = arrayfun(@(x) ['Level ' num2str(x)], raw_intensity_levels, 'UniformOutput', false);
set(gca, 'XTickLabel', labels);
ylim([0 1]);
line(xlim, [max_chance_accuracy max_chance_accuracy], 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
title('Accuracy of Model Trained ONLY on Threshold Data', 'FontSize', 14);
xlabel('Stimulus Intensity Level (Test Set)');
ylabel('Accuracy');