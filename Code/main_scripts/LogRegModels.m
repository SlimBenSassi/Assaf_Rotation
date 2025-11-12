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

%% Global Variables

MasterTable = MasterTable(MasterTable.StimIntensityRaw <= 3 , :); 
%MasterTable = MasterTable(MasterTable.StimIntensityRaw >= 3, :); 


% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms= double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

n_permutations = 50;

% Baseline Accuracy on Dataset

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


%%  A.1 DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODELS (GLMM)

% 1. Model with only alpha as fixed effect

model_formula1 = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms + (1 |SubjectID)';

disp(['Fitting GLMM model: ' model_formula1]);


% --- Fit the GLMM ---
% Family: 'Binomial' (because the outcome is binary: Seen vs. Unseen)
% Distribution: 'logit' (Standard link function for binary logistic regression)

glme_alpha = fitglme(MasterTable, model_formula1, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_alpha.Coefficients);

%% A.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome1, predicted_outcome1, accuracy1] = predict_outcome(MasterTable, glme_alpha, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome1, predicted_outcome1);

accuracy_per_bin1 = accuracy_per_bin(MasterTable, actual_outcome1, predicted_outcome1, threshold);

%% A.3 PERMUTATION TESTS

[null_accuracies, p_value_permutation] = permutation_test(MasterTable, model_formula1, threshold, accuracy1, n_permutations);


%% A.4 LIKELIHOOD RATIO TEST: STIM ONLY MODEL VS STIM + ALPHA

model_formula2 = 'SubjectiveOutcome ~ StimIntensity + (1 |SubjectID)';

disp(['Fitting GLMM model: ' model_formula2]);

glme_stim = fitglme(MasterTable, model_formula2, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_stim.Coefficients);

model_formula3 = 'SubjectiveOutcome ~ StimIntensity + AlphaPower_Avg_100ms + (1 |SubjectID)';

disp(['Fitting GLMM model: ' model_formula3]);

glme_alpha_stim = fitglme(MasterTable, model_formula3, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_alpha_stim.Coefficients);

disp('--- Running Likelihood Ratio Test (LRT) ---');

comparison_table = compare(glme_stim, glme_alpha_stim);

disp('LRT Results:');
disp(comparison_table);


%% B. DEFINE AND FIT THE GENERALIZED LINEAR MIXED MODEL (GLMM) COMPLEX


% Model Formula: SubjectiveOutcome (Y) ~ AlphaPower * StimIntensity + (1|SubjectID)
% The * includes both main effects AND the critical interaction term.
model_formula4 = 'SubjectiveOutcome ~ AlphaPower_Avg_100ms * StimIntensity + (1|SubjectID)';

disp(['Fitting GLMM model: ' model_formula4]);

% --- Add this fix immediately before the fitglme call ---

% Force predictors to double precision (the standard for GLMM)
MasterTable.AlphaPower_Avg_100ms = double(MasterTable.AlphaPower_Avg_100ms);
MasterTable.StimIntensity = double(MasterTable.StimIntensity);

% --- Now run the model ---
glme_alpha_stim_inter = fitglme(MasterTable, model_formula4, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

disp('Model fitting complete.');

disp(glme_alpha_stim_inter.Coefficients);


%% B.2 PREDICTION ACCURACY AND CLASSIFICATION

% Predicting Outcome and Accuracy
[actual_outcome4, predicted_outcome4, accuracy4] = predict_outcome(MasterTable, glme_alpha_stim_inter, threshold);

% Confusion Matrix
confusion_matrix(actual_outcome4, predicted_outcome4);

accuracy_per_bin4 = accuracy_per_bin(MasterTable, actual_outcome4, predicted_outcome4, threshold);


%% Interaction LRT

disp('--- Running Likelihood Ratio Test (LRT) ---');

comparison_table = compare(glme_alpha_stim, glme_alpha_stim_inter);

disp('LRT Results:');
disp(comparison_table);
