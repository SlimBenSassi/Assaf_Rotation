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



%% 2. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) SIMPLE

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_200ms + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower = double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_simple = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');


%% 2.1 

%% 5. PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 2 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_200ms);
alpha_std = std(MasterTable.AlphaPower_Avg_200ms);
alpha_plot_range = linspace(alpha_mean - 2*alpha_std, alpha_mean + 2*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = 4; 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme.Coefficients;
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

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;







%% 3. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX

% Define the full model formula (The Core Scientific Test)
% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes the main effects AND the critical interaction term.
model_formula = 'SubjectiveOutcome ~ AlphaPower_Avg_200ms * StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower = double(MasterTable.AlphaPower_Avg_200ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_simple = fitglme(MasterTable, model_formula, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit');

disp('Model fitting complete.');


%% 5. PLOT LOGISTIC PREDICTION CURVE (Seen Probability vs. Alpha Power)

% --- CONFIGURATION: Define the range of Alpha Power to plot ---
% We use 2 standard deviations around the mean power for the X-axis range.
alpha_mean = mean(MasterTable.AlphaPower_Avg_200ms);
alpha_std = std(MasterTable.AlphaPower_Avg_200ms);
alpha_plot_range = linspace(alpha_mean - 2*alpha_std, alpha_mean + 2*alpha_std, 100);

% --- CRUCIAL: Define the Stimulus Intensity for the plot ---
% To plot a single 2D line, we must fix the StimIntensity (X2). 
% We typically set it to the mean intensity of the dataset.
fixed_intensity = 4; 


% 1. Extract Fixed Effect Coefficients (Assuming the structure is correct for your system)
T = glme.Coefficients;
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

% Add points for the raw data groups (for visual context)
scatter(MasterTable.AlphaPower_Avg_200ms, MasterTable.SubjectiveOutcome, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.1); 

% Aesthetics
title('Logistic Prediction: Alpha Power vs. Awareness (Inverse S-Curve)', 'FontSize', 14);
xlabel('Pre-Stimulus Alpha Power (Predictor X1)');
ylabel('Predicted Probability of "Seen" (Y)');
ylim([0 1]); % Probability scale
grid on;
hold off;