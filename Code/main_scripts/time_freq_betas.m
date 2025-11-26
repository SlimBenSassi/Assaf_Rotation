%% TIME-FREQUENCY LOGISTIC REGRESSION ANALYSIS


%% 1. INITIALIZATION AND LOAD MASTER TABLE

%clear; close all; clc
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

tic

% Load the MasterTable (contains SubjectID, AlphaPower, StimIntensity, SubjectiveOutcome)
load(master_table_file, 'MasterTable');
toc
disp('Master Table loaded successfully.');

% Ensure data types are correct for the GLMM function
MasterTable.SubjectID = categorical(MasterTable.SubjectID);
MasterTable.SubjectiveOutcome = logical(MasterTable.SubjectiveOutcome); 

% Display total N and the first few rows
disp(['Total trials in model: ' num2str(size(MasterTable, 1))]);
head(MasterTable);

%copy in case I need it quickly instead of reloading dataset file
MasterTable_copy = MasterTable;



%% GLOBAL VARIABLES

DO_BASELINE_CORRECTION = true;
Fs = 1024; % Assuming Fs is 1024 Hz
n_channels = 71; %change if needed
PRE_EVENT_SEC = 0.5; % Assumed pre-stimulus window
N_TIME_BINS = 10;
N_FREQ_BINS = 5; % Code not ready for other values yet
alpha_freq_range = [8 12];
time_window_sec = [-0.350 0];
baseline_start_sec = -0.490; % to avoid rounding problems not taking -500ms
baseline_end_sec = -0.400; 



% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 28, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};


%% BASELINE_CORRECTION

tic
if DO_BASELINE_CORRECTION
    for i = 1:height(MasterTable)
        MasterTable.AlphaAmplitude{i} = baseline_correction(MasterTable.AlphaAmplitude{i}, Fs, PRE_EVENT_SEC, baseline_start_sec, baseline_end_sec);
    end
    disp('All single-trial Alpha features are now baseline-normalized (dB).');
end
toc

%% AVERAGE DATA OF ROI (only run this if data file loaded isn't already ROI-averaged)

% 1. Average across the Channel Dimension (Dimension 3)
% Slices the power envelope to the current ROI channels and averages their values.
tic
if length(size(MasterTable.AlphaAmplitude{1})) > 2
    for i = 1:height(MasterTable)
        MasterTable.AlphaAmplitude{i} = squeeze(mean(MasterTable.AlphaAmplitude{i}(:, :, currentROI), 3));
    end
end
toc
disp('All single-trial Alpha features are now averaged over ROI channels.');

%% Z-SCORING (ASSUMES CHANNELS ALREADY AVERAGED)

tic
MasterTable = zscore_data(MasterTable, Fs, PRE_EVENT_SEC, baseline_start_sec, baseline_end_sec);
toc

disp('All single-trial Alpha Amplitude and StimIntensity data is now Z-scored within condition.');

%% DIFFERENT DATASETS    

% Convert the categorical column to its underlying numeric codes (1, 2, 3)
condition_codes = double(MasterTable.Condition);

% Now use the numerical codes for the comparison logic

% 1. MasterTable_Predictive (Condition < 3: Keep Rhythm and Interval)
MasterTable_Predictive = MasterTable(condition_codes < 3, :);

% 1.a Rhythm
MasterTable_Rhythm = MasterTable(condition_codes == 1, :);

% 1.b Interval
MasterTable_Interval = MasterTable(condition_codes == 2, :);

% 2. MasterTable_Irregular (Condition == 3: Keep Irregular)
MasterTable_Irregular = MasterTable(condition_codes == 3, :);

%% TIME-FREQUENCY REGRESSION MAP

outcome_type = 'Objective';

tic
[BetaMap, PValueMap, TimeBins, FreqBins] = tf_regression_map(MasterTable_Rhythm, Fs, [-0.490;0], alpha_freq_range, N_TIME_BINS, N_FREQ_BINS, outcome_type, true);
toc

%%
% Generate a 3x2 figure for different outcome types and datasets
outcome_types = {'Objective', 'Subjective'};
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};

% outer script
f = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
for row = 1:length(outcome_types)
    for col = 1:length(datasets)
        ax = subplot(length(outcome_types), length(datasets), (row-1)*length(datasets) + col);
        [BetaMap, PValueMap, TimeBins, FreqBins] = ...
            tf_regression_map(datasets{col}{1}, Fs, time_window_sec, alpha_freq_range, ...
                              N_TIME_BINS, N_FREQ_BINS, outcome_types{row}, false);
        % plot into the current axes
        imagesc(TimeBins, FreqBins, BetaMap'); %axis xy;
        clim([-0.05, 0.05]); 
        colormap('jet'); % Use a high-contrast diverging colormap (jet or parula)
        hold on; 
        title(sprintf('%s - %s', outcome_types{row},datasets{col}{2}));
        axis xy; % CRITICAL: Flips the Y-axis so low frequencies are at the bottom.
        % Reverse X-axis direction
        %set(gca, 'YDir', 'reverse');
        % --- 1. Create a logical mask for significance (p < 0.05) ---
        %significant_mask = (PValueMap' < 0.1); % 0.025 because our hypothesis is one-tailed 
        % If you already used transposed maps in plotting, keep same orientation:
        sig_neg_mask = (PValueMap' < 0.1) & (BetaMap' < 0);
        % 2. Plot the contour lines using the mask
        [~, h_contour] = contour(TimeBins, FreqBins, sig_neg_mask, [0.5 0.5], 'LineWidth', 3, 'LineColor', 'w', 'LineStyle', ':');
        colorbar;

    end
end

%%
% Generate a 3x2 figure for different outcome types and datasets
outcome_types = {'Objective', 'Subjective'};
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};

% Define colors for clarity
colors = {[0, 0.447, 0.741], [0.85, 0.325, 0.098]}; % Blue for 0, Red for 1
outcome_labels = {'Subjective Outcome = 0 (Failure/No-Percept)', 'Subjective Outcome = 1 (Success/Percept)'};

% Create a 3x2 figure for different outcome types and datasets
f = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);

for row = 1:length(outcome_types)            % subjective / objective
    for col = 1:length(datasets)             % dataset 1..N
        
        ax = subplot(length(outcome_types), length(datasets), ...
                     (row-1)*length(datasets) + col);
        hold on;

        % Pick correct outcome column
        if strcmp(outcome_types{row}, 'Objective')
            outcome_col = datasets{col}{1}.ObjectiveOutcome;
        else
            outcome_col = datasets{col}{1}.SubjectiveOutcome;
        end

        % Two lines: outcome = 0 and outcome = 1
        for o = 0:1

            % Trials matching outcome
            trial_indices = (outcome_col == o);

            % Extract α time series for those trials
            time_series_cells = datasets{col}{1}.AlphaAmplitude(trial_indices);

            % Extract the 3rd column (10 Hz band), row vector per trial
            time_course_data = cellfun(@(x) x(:,3).', ...
                time_series_cells, 'UniformOutput', false);

            % Stack into matrix: nTrials × nTime
            M = cell2mat(time_course_data);

            % Average time course
            avg_tc = mean(M, 1);

            % Plot
            plot(avg_tc, 'LineWidth', 1.8, 'DisplayName', sprintf('Outcome = %d', o));
        end
        
        title(sprintf('%s – %s', outcome_types{row}, datasets{col}{2}));
        xlabel('Time');
        line([512, 512], [4 5], 'Color', 'k', 'LineStyle', ':', 'LineWidth', 1.5, 'DisplayName', 'Target');
        ylabel('Alpha Amplitude (10 Hz)');
        ylim([4 5])
        legend('show', 'Location', 'Best' );
        grid on;
        hold off;
    end
end

hold off;
%% INTERACTION ANALYSIS

% Here we will take one alpha predictor and not each time-frequency
% combination, choice is informed visually by the heatmap results.

time_pred_bin = [-0.300; -0.200];
freq_pred_bin = [10; 11];

time_zero_sample = round(PRE_EVENT_SEC * Fs); 
pred_start_sample = time_zero_sample + round(time_pred_bin(1) * Fs); 
pred_end_sample = time_zero_sample + round(time_pred_bin(2) * Fs);

all_freqs = [alpha_freq_range(1):alpha_freq_range(2)];
freq_start_sample = find(all_freqs==freq_pred_bin(1));
freq_end_sample = find(all_freqs==freq_pred_bin(2));


% assumes no channel dimension
for i = 1:height(MasterTable)
    MasterTable.AlphaAmplitudeAvg(i) = ...
    squeeze(mean(mean(MasterTable.AlphaAmplitude{i}(pred_start_sample:pred_end_sample, freq_start_sample:freq_end_sample), 1), 2));
end


%% DIFFERENT DATASETS    

% Convert the categorical column to its underlying numeric codes (1, 2, 3)
condition_codes = double(MasterTable.Condition);

% Now use the numerical codes for the comparison logic

% 1. MasterTable_Predictive (Condition < 3: Keep Rhythm and Interval)
MasterTable_Predictive = MasterTable(condition_codes < 3, :);

% 1.a Rhythm
MasterTable_Rhythm = MasterTable(condition_codes == 1, :);

% 1.b Interval
MasterTable_Interval = MasterTable(condition_codes == 2, :);

% 2. MasterTable_Irregular (Condition == 3: Keep Irregular)
MasterTable_Irregular = MasterTable(condition_codes == 3, :);

%%

% Define the datasets and outcome types
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};
outcome_types = {'Subjective', 'Objective'};

% Prepare data for bar plot
subjective_betas = [];
objective_betas = [];
subjective_CI = [];
objective_CI = [];

labels = cell(1, length(datasets));

for col = 1:length(datasets)
    data_condition = datasets{col}{1};
    labels{col} = datasets{col}{2};
    
    % Fit GLME for Subjective
    model_formula_subjective = 'SubjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)';
    glme_subjective = fitglme(data_condition, model_formula_subjective, ...
                               'Distribution', 'Binomial', ...
                               'Link', 'logit', 'FitMethod', 'Laplace');
    subjective_betas(col) = glme_subjective.Coefficients.Estimate(2);
    all_CIs = coefCI(glme_subjective); 
    subjective_CI(:, col) = all_CIs(2,:);
    
    % Fit GLME for Objective
    model_formula_objective = 'ObjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)';
    glme_objective = fitglme(data_condition, model_formula_objective, ...
                              'Distribution', 'Binomial', ...
                              'Link', 'logit', 'FitMethod', 'Laplace');
    objective_betas(col) = glme_objective.Coefficients.Estimate(2);
    all_CIs = coefCI(glme_objective); 
    objective_CI(:, col) = all_CIs(2,:);
end

% Create bar plot
figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.4]);
bar_data = [subjective_betas; objective_betas]';
bar_handle = bar(bar_data, 'grouped');
hold on;

% Set colors for bars
bar_handle(1).FaceColor = [0.2 0.6 0.8]; % Subjective color
bar_handle(2).FaceColor = [0.8 0.2 0.2]; % Objective color


% --- Correct x-positions (this is the IMPORTANT part) ---
x_subjective = bar_handle(1).XEndPoints;
x_objective  = bar_handle(2).XEndPoints;

% Use the extracted labels (Assuming 'labels' is defined correctly)
set(gca, 'XTickLabel', labels, 'XTickLabelRotation', 45);
ylabel('Coefficient Estimate');
ylim([-0.06 0.06]);
title('Coefficient Estimates and CIs for Subjective and Objective Outcomes');


% Add legend before error bars so the bars are in the legend
legend({'Subjective', 'Objective'}, 'Location', 'Best', 'TextColor', 'k', ...
       'Color', 'w', 'EdgeColor', 'k', 'Box', 'on');

% Add error bars for confidence intervals (Use the calculated x_subjective/x_objective)
for col = 1:length(datasets)
    % Subjective CI
    errorbar(x_subjective(col), subjective_betas(col), ...
             subjective_betas(col) - subjective_CI(1, col), ...
             subjective_CI(2, col) - subjective_betas(col), ...
             'k', 'linestyle', 'none', 'LineWidth', 1.5, 'HandleVisibility','off');
    
    % Objective CI
    errorbar(x_objective(col), objective_betas(col), ...
             objective_betas(col) - objective_CI(1, col), ...
             objective_CI(2, col) - objective_betas(col), ...
             'k', 'linestyle', 'none', 'LineWidth', 1.5, 'HandleVisibility','off');
end

grid on;


%% A.1 LINEAR ALPHA SLOPE PER INTENSITY (NO INTERACTION TERM)

data_condition = MasterTable_Rhythm;
outcome_type = 'Subjective';

if strcmp(outcome_type, 'Subjective')
   model_formula1 = 'SubjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)';
else
   model_formula1 = 'ObjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)';
end

intensity_levels = unique(data_condition.StimIntensity);
betas_per_intensity = [];
CI_per_intensity = [];


for i = 1:length(intensity_levels)
    
    temp_table = data_condition(data_condition.StimIntensity == intensity_levels(i) , :); 

    glme_alpha_temp = fitglme(temp_table, model_formula1, ...
               'Distribution', 'Binomial', ...
               'Link', 'logit', 'FitMethod','Laplace');

    beta_temp = glme_alpha_temp.Coefficients.Estimate(2);
    CI_temp = coefCI(glme_alpha_temp);  % gives 95% confidence intervals

    
    betas_per_intensity = [betas_per_intensity; beta_temp];
    CI_per_intensity = [CI_per_intensity; CI_temp];

    %disp('Model fitting complete.');
    %disp(glme_alpha_temp.Coefficients);

end

figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

%scatter(intensity_levels, betas_per_intensity);
bar_plot = bar(betas_per_intensity);
hold on;

% Create labels for the X-axis (e.g., 'Bin 1', 'Bin 2', etc.)
labels = arrayfun(@(x) ['Constrast ' num2str(x)], min(intensity_levels):max(intensity_levels), 'UniformOutput', false);

set(gca, 'XTickLabel', labels);
max_abs_beta = max(abs(betas_per_intensity));
ylim([-max_abs_beta-0.01 max_abs_beta]);
%line(xlim, [baseline_accuracy baseline_accuracy], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5); % Chance line

title('Alpha Coefficient Estimate for models trained on each Stimulus Intensity', 'FontSize', 14);
xlabel('Difficulty Level');
ylabel('Alpha Coefficient Estimate');
grid on;

%% A.2 LINEAR ALPHA SLOPE PER INTENSITY (ONE INTERACTION TERM)


% Choose model formula
if strcmp(outcome_type, 'Subjective')
    model_formula = 'SubjectiveOutcome ~ AlphaAmplitudeAvg * StimIntensityZ + (1|SubjectID)';
else
    model_formula = 'ObjectiveOutcome ~ AlphaAmplitudeAvg * StimIntensityZ + (1|SubjectID)';
end

% --- 1. Create 10 bins of StimIntensityZ
num_bins = 10;

% Bin StimIntensityZ using quantiles so bins have equal number of samples
data_condition.StimZ_bin = discretize(data_condition.StimIntensityZ, ...
     linspace(min(data_condition.StimIntensityZ), max(data_condition.StimIntensityZ),num_bins+1));

% Compute mean StimIntensityZ per bin
meanZ_per_bin = splitapply(@mean, data_condition.StimIntensityZ, data_condition.StimZ_bin);

%---- 2. Fit one GLME to all data (with interaction!) ----
glme_alpha = fitglme(data_condition, model_formula, ...
           'Distribution','Binomial', ...
           'Link','logit', ...
           'FitMethod','Laplace');

beta0 = glme_alpha.Coefficients.Estimate(1); % intercept
beta1 = glme_alpha.Coefficients.Estimate(3); % Alpha
beta2 = glme_alpha.Coefficients.Estimate(2); % StimZ
beta3 = glme_alpha.Coefficients.Estimate(4); % Alpha × StimZ interaction

% ---- 3. Compute simple slopes: beta_alpha(bin) = beta1 + beta3 * meanStimZ(bin) ----
beta_alpha_per_bin = beta1 + beta3 * meanZ_per_bin;

% ---- 4. Plot ----
figure('Units','normalized','Position',[0.1 0.1 0.5 0.5]);
bar(beta_alpha_per_bin)
hold on

% X labels
labels = arrayfun(@(x) sprintf('Bin %d', x), 1:num_bins, 'UniformOutput', false);
set(gca,'XTickLabel', labels)

max_abs_beta = max(abs(beta_alpha_per_bin));
ylim([-max_abs_beta-0.01, max_abs_beta+0.01])

title('Alpha Slope (β + β_{interaction} × StimZ) across 10 bins')
xlabel('StimIntensityZ Bins')
ylabel('Alpha Slope')
grid on


%% NON-LINEAR INTERACTION MODEL


% Choose model formula
if strcmp(outcome_type, 'Subjective')
    model_formula = ['SubjectiveOutcome ~ AlphaAmplitudeAvg * StimIntensityZ + ' ...
                 'AlphaAmplitudeAvg * StimIntensityZ^2 + (1|SubjectID)'];
else
    model_formula = ['ObjectiveOutcome ~ AlphaAmplitudeAvg * StimIntensityZ + ' ...
                 'AlphaAmplitudeAvg * StimIntensityZ^2 + (1|SubjectID)'];
end
% --- 1. Create 10 bins of StimIntensityZ
num_bins = 10;

% Bin StimIntensityZ using quantiles so bins have equal number of samples
data_condition.StimZ_bin = discretize(data_condition.StimIntensityZ, ...
     linspace(min(data_condition.StimIntensityZ), max(data_condition.StimIntensityZ),num_bins+1));

% Compute mean StimIntensityZ per bin
meanZ_per_bin = splitapply(@mean, data_condition.StimIntensityZ, data_condition.StimZ_bin);

%---- 2. Fit one GLME to all data (with interaction!) ----
glme_alpha = fitglme(data_condition, model_formula, ...
           'Distribution','Binomial', ...
           'Link','logit', ...
           'FitMethod','Laplace');
disp(glme_alpha.Coefficients);

beta0 = glme_alpha.Coefficients.Estimate(1); % intercept
beta1 = glme_alpha.Coefficients.Estimate(3); % Alpha
beta2 = glme_alpha.Coefficients.Estimate(2); % StimZ
beta3 = glme_alpha.Coefficients.Estimate(4); % Alpha × StimZ interaction
beta4 = glme_alpha.Coefficients.Estimate(6); % Alpha × StimZ^2 interaction


% ---- 3. Compute simple slopes: beta_alpha(bin) = beta1 + beta3 * meanStimZ(bin) ----
beta_alpha_per_bin2 = beta1 + beta3 * meanZ_per_bin + beta4 * meanZ_per_bin * 2;

% ---- 4. Plot ----
figure('Units','normalized','Position',[0.1 0.1 0.5 0.5]);
bar(beta_alpha_per_bin2);
hold on

% X labels
labels = arrayfun(@(x) sprintf('Bin %d', x), 1:num_bins, 'UniformOutput', false);
set(gca,'XTickLabel', labels)

max_abs_beta = max(abs(beta_alpha_per_bin2));
ylim([-max_abs_beta-0.01, max_abs_beta+0.01])

title('Alpha Slope (Quadratic model) across 10 bins')
xlabel('StimIntensityZ Bins')
ylabel('Alpha Slope')
grid on




%% A.4 Comparing Linear Interaction to true interaction

%PLOT GROUPED BETA COEFFICIENTS (Comparing Model Effects)

% --- 1. Create the Grouped Matrix ---
% The matrix must be [N_Bins x N_Models]
beta_comparison_matrix = [betas_per_intensity, beta_alpha_per_bin, beta_alpha_per_bin2];

% --- 2. Create the Figure and Bar Plot ---
figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.6]);

% Plot the matrix. MATLAB automatically groups the bars by row (intensity level).
hBar = bar(beta_comparison_matrix);
hold on;

% --- 3. Add Zero Line (Crucial Reference) ---
line(xlim, [0 0], 'Color', 'k', 'LineStyle', ':', 'LineWidth', 1.5);

% --- 4. Aesthetics and Labels ---
% Use the raw intensity levels (e.g., 1 to 7) for X-axis labels
intensity_labels = arrayfun(@(x) ['Level ' num2str(x)], intensity_levels, 'UniformOutput', false);

set(gca, 'XTickLabel', intensity_labels);
title('Predictive Power (Beta) by Intensity: Simple vs. Complex Model', 'FontSize', 14);
xlabel('Stimulus Intensity Level (Bins)');
ylabel('Alpha Power Beta Coefficient (Log-Odds)');
legend({'No Interaction Model', 'Linear Interaction Model', 'Quadratic Interaction Model'}, 'Location', 'NorthWest');
grid on;
hold off;



%% SHORT EXPLORATIION OF CALIBRATION

% 1.  EMPIRICAL CONFUSION MATRIX (Observed vs. Reality)
data_condition = MasterTable_Irregular;
% --- Configuration ---
% Note: We assume that TRUE (1) means "Correct Tilt Identified" (Objective Positive) 
% and TRUE (1) means "Reported Seen" (Subjective Positive).

objective_outcome = double(data_condition.ObjectiveOutcome); % Actual result (0 or 1)
subjective_outcome = double(data_condition.SubjectiveOutcome); % Subject's report (0 or 1)

% --- 1. Calculate the Four Outcomes ---

% True Positives (TP): Subjective SEEN (1) AND Objective CORRECT (1)
TP = sum(objective_outcome == 1 & subjective_outcome == 1);

% True Negatives (TN): Subjective UNSEEN (0) AND Objective INCORRECT (0)
TN = sum(objective_outcome == 0 & subjective_outcome == 0);

% False Positives (FP): Subjective SEEN (1) BUT Objective INCORRECT (0)
% Subject reported seeing it, but made an incorrect objective judgment. (Overconfidence)
FP = sum(objective_outcome == 0 & subjective_outcome == 1);

% False Negatives (FN): Subjective UNSEEN (0) BUT Objective CORRECT (1)
% Subject missed seeing it, but was objectively correct. (Unconscious processing)
FN = sum(objective_outcome == 1 & subjective_outcome == 0);


% --- 2. Display the Results (Averaged Across Participants) ---
N_total = TP + TN + FP + FN;

disp(' ');
disp('--- EMPIRICAL CONFUSION MATRIX (Participant Report vs. Reality) ---');
disp(['Total Trials: ' num2str(N_total)]);
disp(' ');
disp('RESULTING CATEGORIES:');
disp(['TP (Seen & Correct): ' num2str(TP/N_total * 100, 3) '%']);
disp(['TN (Unseen and Incorrect Rejected): ' num2str(TN/N_total * 100, 3) '%']);
disp(['FP (False Alarm/Overconfidence): ' num2str(FP/N_total * 100, 3) '%']);
disp(['FN (Unconscious Processing or Guessing): ' num2str(FN/N_total * 100, 3) '%']);
disp('------------------------------------------');



%% --- Choose participant
participant_id = "102";   % <-- change your subject ID here

T = MasterTable_Irregular(MasterTable_Irregular.SubjectID == participant_id, :);

% Extract variables
intensity = T.StimIntensityZ;       % or StimIntensity, whichever you're using
alpha    = T.AlphaAmplitudeAvg;
response = T.SubjectiveOutcome;     % OR ObjectiveOutcome

% --- Split alpha into LOW / MID / HIGH bins (quantile-based)
edges = quantile(alpha, [0 1/3 2/3 1]);
alpha_bin = discretize(alpha, edges);

% 1 = low, 2 = mid, 3 = high

% --- Prepare intensity levels
uI = unique(intensity);
uI = sort(uI);

% --- Fit logistic models for each alpha range
curve_low  = nan(size(uI));
curve_mid  = nan(size(uI));
curve_high = nan(size(uI));

for i = 1:length(uI)
    % indices of current intensity
    idx = intensity == uI(i);

    % LOW ALPHA
    resp_low = response(idx & alpha_bin == 1);
    if ~isempty(resp_low)
        p = mean(resp_low);
    else
        p = NaN;
    end
    curve_low(i) = p;

    % MID ALPHA
    resp_mid = response(idx & alpha_bin == 2);
    if ~isempty(resp_mid)
        p = mean(resp_mid);
    else
        p = NaN;
    end
    curve_mid(i) = p;

    % HIGH ALPHA
    resp_high = response(idx & alpha_bin == 3);
    if ~isempty(resp_high)
        p = mean(resp_high);
    else
        p = NaN;
    end
    curve_high(i) = p;
end

% --- Plot
figure; hold on;
plot(uI, curve_low,  '-o', 'LineWidth', 2);
plot(uI, curve_mid,  '-o', 'LineWidth', 2);
plot(uI, curve_high, '-o', 'LineWidth', 2);

xlabel('Stimulus Intensity (Z)');
ylabel('P(Response = 1)');
title(['Psychometric Curves for Participant ' participant_id]);

legend({'Low Alpha','Mid Alpha','High Alpha'}, 'Location','best');
grid on;

%%
% extract coefficients and covariance
coefTable = glme_alpha.Coefficients;
beta_names = coefTable.Name;
beta_vals  = coefTable.Estimate;
covMat = glme_alpha.CoefficientCovariance;  % full covariance matrix

% find indices (adjust names to your table)
i_alpha = find(strcmp(beta_names,'AlphaAmplitudeAvg'));
i_int   = find(strcmp(beta_names,'StimIntensityZ'));
i_intx  = find(strcmp(beta_names,'StimIntensityZ:AlphaAmplitudeAvg'));
% Extract the coefficient for the quadratic term (if applicable)
i_intx2 = find(strcmp(beta_names, 'StimIntensityZ^2:AlphaAmplitudeAvg'));

% Update the covariance submatrix to include the quadratic term
cov_sub = covMat([i_alpha, i_intx, i_intx2], [i_alpha, i_intx, i_intx2]);

b1 = beta_vals(i_alpha);
b3 = beta_vals(i_intx);
b5 = beta_vals(i_intx2);

% covariance submatrix for [beta1; beta3]
%cov_sub = covMat([i_alpha,i_intx],[i_alpha,i_intx]);

% grid of stim values (use meaningful range, e.g. min..max or bin means)
%s_grid = linspace(min(data_condition.StimIntensityZ), max(data_condition.StimIntensityZ), 10);
% instead of 10 values starting from min to max, can i get a s_grid that's made of the mean value
% of each bin if range of StiIntensityZ is split into N bins
% data_condition.StimIntensityZ is a vector
N = 10;                                 % number of bins wanted
x = data_condition.StimIntensityZ(:);   % ensure column vector

% create bin edges spanning min..max
edges = linspace(min(x), max(x), N+1);

% assign bin indices (1..N), last edge included in last bin by default
binIdx = discretize(x, edges);

% preallocate s_grid and compute mean per bin
s_grid = NaN(N,1);
for k = 1:N
    members = x(binIdx == k);
    if ~isempty(members)
        s_grid(k) = mean(members);
    end
end

% Optionally remove empty bins
s_grid = s_grid(~isnan(s_grid));


% cov_sub must be [beta1; beta3; beta5]
cov_sub = covMat([i_alpha, i_intx, i_intx2], [i_alpha, i_intx, i_intx2]);

se_slope = arrayfun(@(s) ...
    sqrt([1 s s^2] * cov_sub * [1; s; s^2]), ...
    s_grid);


slope = b1 + b3.*s_grid + b5.*(s_grid.^2);
%se_slope = arrayfun(@(s) sqrt([1 s]*cov_sub*[1; s]), s_grid);  % delta method

z = 1.96;
CI_lower = slope - z*se_slope;
CI_upper = slope + z*se_slope;

% plot
figure; hold on;
plot(s_grid, slope, 'k-', 'LineWidth', 2);
plot(s_grid, CI_lower, 'r--');
plot(s_grid, CI_upper, 'r--');
xlabel('StimIntensityZ'); ylabel('Alpha slope (β1 + β3*s)');
title('Simple slope of Alpha across StimIntensityZ with 95% CI');
grid on;

%% POLYNOMIAL (DEGREE 3) DIDNT ADD MUCH

% prepare
T = MasterTable_Irregular;  % rename for convenience
T.StimZ = T.StimIntensityZ;
T.StimZc = T.StimZ;
% create polynomial terms (center & scale first)
%T.StimZc = (T.StimZ - mean(T.StimZ)) ./ std(T.StimZ);
T.StimZ2 = T.StimZc.^2;
T.StimZ3 = T.StimZc.^3; % optional

% create interaction columns
T.Alpha_StimZ  = T.AlphaAmplitudeAvg .* T.StimZc;
T.Alpha_StimZ2 = T.AlphaAmplitudeAvg .* T.StimZ2;
T.Alpha_StimZ3 = T.AlphaAmplitudeAvg .* T.StimZ3;  % optional

% build formula (with quadratic interaction)
formula = ['SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + ' ...
           'Alpha_StimZ + Alpha_StimZ2 + Alpha_StimZ3 + (1|SubjectID)'];

% fit GLME
glme_poly = fitglme(T, formula, 'Distribution','Binomial','Link','logit','FitMethod','Laplace');

% Example for polynomial case:
% indices
tbl = glme_poly.Coefficients;
i_alpha = find(strcmp(tbl.Name,'AlphaAmplitudeAvg'));
i_a_s   = find(strcmp(tbl.Name,'Alpha_StimZ'));
i_a_s2  = find(strcmp(tbl.Name,'Alpha_StimZ2'));
i_a_s3  = find(strcmp(tbl.Name,'Alpha_StimZ3'));

b1 = tbl.Estimate(i_alpha);
b3 = tbl.Estimate(i_a_s);
b5 = tbl.Estimate(i_a_s2);
b7 = tbl.Estimate(i_a_s3);

% grid (use centered StimZc)
s_grid = linspace(min(T.StimZc), max(T.StimZc), 200)';
slope = b1 + b3.*s_grid + b5.*(s_grid.^2) + b7.*(s_grid.^3);

% delta-method CI (cov_sub 3x3)
covMat = glme_poly.CoefficientCovariance;
cov_sub = covMat([i_alpha, i_a_s, i_a_s2, i_a_s3],[i_alpha, i_a_s, i_a_s2, i_a_s3]);

se_slope = arrayfun(@(s) sqrt([1 s s^2 s^3]*cov_sub*[1; s; s^2; s^3]), s_grid);
CI_lower = slope - 1.96*se_slope; CI_upper = slope + 1.96*se_slope;


% plot
figure; hold on;
plot(s_grid, slope, 'k-', 'LineWidth', 2);
plot(s_grid, CI_lower, 'r--');
plot(s_grid, CI_upper, 'r--');
xlabel('StimIntensityZ'); ylabel('Alpha slope (β1 + β3*s)');
title('Simple slope of Alpha across StimIntensityZ with 95% CI');
grid on;

%% DEGREE 4

T.StimZ4 = T.StimZc.^4;  % create fourth degree polynomial term
T.Alpha_StimZ4 = T.AlphaAmplitudeAvg .* T.StimZ4;  % create interaction for fourth degree
formula = ['ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + StimZ4 + ' ...
           'Alpha_StimZ + Alpha_StimZ2 + Alpha_StimZ3 + Alpha_StimZ4 + (1|SubjectID)'];
% fit GLME with updated formula
glme_poly = fitglme(T, formula, 'Distribution','Binomial','Link','logit','FitMethod','Laplace');

% Example for polynomial case:
% indices
tbl = glme_poly.Coefficients;
i_alpha = find(strcmp(tbl.Name,'AlphaAmplitudeAvg'));
i_a_s   = find(strcmp(tbl.Name,'Alpha_StimZ'));
i_a_s2  = find(strcmp(tbl.Name,'Alpha_StimZ2'));
i_a_s3  = find(strcmp(tbl.Name,'Alpha_StimZ3'));
i_a_s4  = find(strcmp(tbl.Name,'Alpha_StimZ4'));


b1 = tbl.Estimate(i_alpha);
b3 = tbl.Estimate(i_a_s);
b5 = tbl.Estimate(i_a_s2);
b7 = tbl.Estimate(i_a_s3);
b9 = tbl.Estimate(i_a_s4);  % extract coefficient for the fourth degree interaction

% grid (use centered StimZc)
s_grid = linspace(min(T.StimZc), max(T.StimZc), 200)';
slope = b1 + b3.*s_grid + b5.*(s_grid.^2) + b7.*(s_grid.^3);
slope = slope + b9.*(s_grid.^4); 

covMat = glme_poly.CoefficientCovariance;

% delta-method CI (cov_sub 4x4)
cov_sub = covMat([i_alpha, i_a_s, i_a_s2, i_a_s3, i_a_s4], [i_alpha, i_a_s, i_a_s2, i_a_s3, i_a_s4]);

se_slope = arrayfun(@(s) sqrt([1 s s^2 s^3 s^4]*cov_sub*[1; s; s^2; s^3; s^4]), s_grid);
CI_lower = slope - 1.96*se_slope;
CI_upper = slope + 1.96*se_slope;


% plot
figure; hold on;
plot(s_grid, slope, 'k-', 'LineWidth', 2);
plot(s_grid, CI_lower, 'r--');
plot(s_grid, CI_upper, 'r--');
xlabel('StimIntensityZ'); ylabel('Alpha slope (β1 + β3*s)');
title('Simple slope of Alpha across StimIntensityZ with 95% CI');
grid on;
%%
% Create a figure for subplots
figure;

% Define the datasets and outcome types
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};
outcome_types = {'Subjective', 'Objective'};

% Loop through each combination of dataset and outcome type
for d = 1:length(datasets)
    for o = 1:length(outcome_types)
        subplot(length(datasets), length(outcome_types), (d-1)*length(outcome_types) + o); 
        % Load the dataset based on the current combination
        DATA = datasets{d}{1};  % Assuming the datasets are in the workspace
        
        % Prepare the data for plotting
        a = DATA.AlphaAmplitudeAvg;
        edges = quantile(a, [0 1/2 1]);
        DATA.alphaBin = discretize(a, edges);

        DATA.StimZc = DATA.StimIntensityZ; %TODO ADD Z
        
        % Define the grid for StimIntensityZ
        s_grid = linspace(min(DATA.StimIntensityZ), max(DATA.StimIntensityZ), 200)';
        s_grid_plot = linspace(min(DATA.StimIntensity), max(DATA.StimIntensity), 200)';

        
        % Initialize colors for the models
        colors = lines(4);
        
        % Define the models and their corresponding formulas
        models = {'linear interaction', 'quadratic', 'degree 3', 'degree 4'};
        if  strcmp(outcome_types{o}, 'Subjective')
            formulas = {
            'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + (1|SubjectID)', ...
            'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + (1|SubjectID)', ...
            'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + (1|SubjectID)', ...
            'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + StimZ4 + (1|SubjectID)'
            };
        else
            formulas = {
            'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + (1|SubjectID)', ...
            'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + (1|SubjectID)', ...
            'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + (1|SubjectID)', ...
            'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimZ2 + StimZ3 + StimZ4 + (1|SubjectID)'
            };
        end
        % Loop through each model
        for m = 1:length(models)
            
            % Create interaction terms for the model
            DATA.StimAlpha = DATA.AlphaAmplitudeAvg .* DATA.StimZc;
            if m > 1
                DATA.StimZ2 = DATA.StimZc.^2;
                DATA.StimAlpha2 = DATA.AlphaAmplitudeAvg .* DATA.StimZ2;
            end
            if m > 2
                DATA.StimZ3 = DATA.StimZc.^3;
                DATA.StimAlpha3 = DATA.AlphaAmplitudeAvg .* DATA.StimZ3;
            end
            if m > 3
                DATA.StimZ4 = DATA.StimZc.^4;
                DATA.StimAlpha4 = DATA.AlphaAmplitudeAvg .* DATA.StimZ4;
            end
            
            % Update the formulas to include interaction terms
            formulas{m} = strrep(formulas{m}, 'StimZc', 'StimZc + StimAlpha');
            if m > 1
            formulas{m} = strrep(formulas{m}, 'StimZ2', 'StimZ2 + StimAlpha2');
            end
            if m > 2
            formulas{m} = strrep(formulas{m}, 'StimZ3', 'StimZ3 + StimAlpha3');
            end
            if m > 3
            formulas{m} = strrep(formulas{m}, 'StimZ4', 'StimZ4 + StimAlpha4');
            end
            
            % Fit the GLME model
            glme_poly = fitglme(DATA, formulas{m}, 'Distribution', 'Binomial', 'Link', 'logit', 'FitMethod', 'Laplace');

            % Extract coefficients
            tbl = glme_poly.Coefficients;
            b1 = tbl.Estimate(find(strcmp(tbl.Name, 'AlphaAmplitudeAvg')));
            b2 = tbl.Estimate(find(strcmp(tbl.Name, 'StimAlpha')));
            if m > 1
                b3 = tbl.Estimate(find(strcmp(tbl.Name, 'StimAlpha2')));
            end
            if m > 2
                b4 = tbl.Estimate(find(strcmp(tbl.Name, 'StimAlpha3')));
            end
            if m > 3
                b5 = tbl.Estimate(find(strcmp(tbl.Name, 'StimAlpha4')));
            end
            
            % Calculate slope based on the model
            slope = b1 + b2 .* s_grid;
            if m > 1
                slope = slope + b3 .* (s_grid.^2);
            end
            if m > 2
                slope = slope + b4 .* (s_grid.^3);
            end
            if m > 3
                slope = slope + b5 .* (s_grid.^4);
            end
            
            % Calculate confidence intervals using delta method
            covMat = glme_poly.CoefficientCovariance;
            cov_sub = covMat(1:m+1, 1:m+1); % Adjust for the number of coefficients
            %temp_cov =  [1 s s^2 s^3 s^4];
            %se_slope = arrayfun(@(s) sqrt(temp_cov(1:m) * cov_sub * temp_cov(1:m)'), s_grid);
            %se_slope = se_slope(:);  % Ensure se_slope is a column vector
            
            % s_grid: column vector of s values
            % cov_sub: m-by-m covariance matrix
            % m ≤ 5 (since you use powers 0:4)
            
            S = s_grid(:).^(0:4);        % nS-by-5 matrix, row i = [1 s s^2 s^3 s^4]
            Ssub = S(:,1:m+1);             % nS-by-m
            % compute quadratic form row-wise: se_i = sqrt( Ssub(i,:) * cov_sub * Ssub(i,:)' )
            tmp = Ssub * cov_sub;        % nS-by-m
            q = sum(tmp .* Ssub, 2);     % nS-by-1, each row is Ssub(i,:)*cov_sub*Ssub(i,:)'
            se_slope = sqrt(q);

            %powers = 0:(m-1);
            %Ssub = s_grid(:).^powers;     % nS-by-m
            
            %tmp = Ssub * cov_sub;         % nS-by-m
            %q   = sum(tmp .* Ssub, 2);    % nS-by-1
            %se_slope = sqrt(q);


            %CI_lower = slope - 1.96 * se_slope;
            %CI_upper = slope + 1.96 * se_slope;
            hold on;   % ← TURN ON BEFORE ANYTHING

            % Plot the slope and confidence intervals
            %fill([s_grid_plot; flipud(s_grid_plot)], [CI_lower; flipud(CI_upper)], colors(m,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            %hold on;
            plot(s_grid_plot, slope, 'LineWidth', 2, 'Color', colors(m,:), 'DisplayName', models{m});
            %hold off;
            
        end
        
        % Set x-axis ticks based on the range of StimIntensityZ
        %xticks(linspace(min(s_grid_plot), max(s_grid_plot), 10));
        % Set plot labels and title
        xlabel('StimIntensity');
        ylabel('Alpha slope');
        xlim([0 11]);
        ylim([-0.4, 0.1]);
        title(sprintf('%s - %s', datasets{d}{2}, outcome_types{o}));
        legend(models, 'Location', 'Best');
        grid on;
    end
end
%%

a = DATA.AlphaAmplitudeAvg;
edges = quantile(a, [0 1/2 1]);
DATA.alphaBin = discretize(a, edges);



%%

% --- Get representative alpha levels ---
a_low  = quantile(T.AlphaAmplitudeAvg, 0.1);
a_mid  = quantile(T.AlphaAmplitudeAvg, 0.5);
a_high = quantile(T.AlphaAmplitudeAvg, 0.9);

alpha_levels = [a_low; a_mid; a_high];
labels = {'Low α','Mid α','High α'};

% --- Stimulus grid ---
s_grid = linspace(min(T.StimIntensityZ), max(T.StimIntensityZ), 300)';

% --- Create prediction table for GLMM ---
figure; hold on;
colors = lines(3);

for k = 1:3
    
    A = alpha_levels(k);
    
    newT = T(:, {'SubjectID','ObjectiveOutcome'});  % keep these
    newT.StimZc    = s_grid;
    newT.AlphaAmplitudeAvg = repmat(A, length(s_grid), 1);
    newT.ObjectiveOutcome = nan(size(s_grid));
    
    % Recreate all required variables exactly like in T:
    newT.StimZ2 = newT.StimZc.^2;
    newT.StimZ3 = newT.StimZc.^3;
    newT.StimZ4 = newT.StimZc.^4;
    
    newT.Alpha_StimZ  = newT.AlphaAmplitudeAvg .* newT.StimZc;
    newT.Alpha_StimZ2 = newT.AlphaAmplitudeAvg .* newT.StimZ2;
    newT.Alpha_StimZ3 = newT.AlphaAmplitudeAvg .* newT.StimZ3;
    newT.Alpha_StimZ4 = newT.AlphaAmplitudeAvg .* newT.StimZ4;


    % other model columns (random effects ignored)
    % Fit conditional predictions
    p_hat = predict(glme_poly, newT, 'Conditional', false);   % logistic P(response=1)
    
    plot(s_grid, p_hat, 'LineWidth', 2, 'Color', colors(k,:));
end

xlabel('Stimulus intensity (z-centered)');
ylabel('P(report = seen)');
title('GLMM predicted psychometric curves for low/mid/high alpha');
legend(labels);
grid on;


%% FIT (NOT PSIGNIFIT)

DATA = MasterTable_Interval;

% Assuming your table is called DATA
a = DATA.AlphaAmplitudeAvg;

% tertile edges
edges = quantile(a, [0 1/2 1]);

% assign bin (1 = low, 2 = mid, 3 = high)
DATA.alphaBin = discretize(a, edges);


intensities = unique(DATA.StimIntensity);

for b = 1:2
    for i = 1:numel(intensities)
        idx = DATA.alphaBin == b & DATA.StimIntensity == intensities(i);

        nTotal = sum(idx);
        nSeen  = sum(DATA.ObjectiveOutcome(idx)==1);

        prop(b,i) = nSeen / nTotal;

        % Binomial 95% CI
        [phat, pci] = binofit(nSeen, nTotal);
        ci_low(b,i)  = pci(1);
        ci_high(b,i) = pci(2);

        %data_ps = [ intensities(:)  nSeen(:)  nTotal(:) ];

    end
end

figure; hold on;

%labels = {'Low alpha','Mid alpha','High alpha'};
labels = {'Low alpha','High alpha'};

%colors = lines(3);
colors = lines(2);


%for b = 1:3
for b = 1:2

    errorbar(intensities, prop(b,:), ...
        prop(b,:) - ci_low(b,:), ci_high(b,:) - prop(b,:), ...
        'o-','LineWidth',1.5,'Color',colors(b,:));
end

xlabel('Stimulus intensity');
ylabel('Proportion seen');
legend(labels);
title('Empirical psychometric curves (alpha tertiles)');
grid on;


%% FIT (WITH PSIGNIFIT)

participant_id = "138";   % <-- change your subject ID here

%T = MasterTable_Irregular(MasterTable_Irregular.SubjectID == participant_id, :);

DATA = MasterTable_Rhythm;

a = DATA.AlphaAmplitudeAvg;
edges = quantile(a, [0 1/2 1]);
DATA.alphaBin = discretize(a, edges);


intensities = unique(DATA.StimIntensity);
nInt = numel(intensities);

prop = nan(2, nInt);
ci_low = nan(2, nInt);
ci_high = nan(2, nInt);

nSeen_all = zeros(2, nInt);
nTotal_all = zeros(2, nInt);

for b = 1:2
    for i = 1:nInt
        
        idx = (DATA.alphaBin == b) & (DATA.StimIntensity == intensities(i));

        nTotal = sum(idx);
        nSeen  = sum(DATA.SubjectiveOutcome(idx)==1);

        nSeen_all(b,i) = nSeen;
        nTotal_all(b,i) = nTotal;

        if nTotal == 0
            prop(b,i) = NaN;
            ci_low(b,i) = NaN;
            ci_high(b,i) = NaN;
        else
            prop(b,i) = nSeen / nTotal;

            [~, pci] = binofit(nSeen, nTotal);
            ci_low(b,i)  = pci(1);
            ci_high(b,i) = pci(2);
        end
    end
            % Display the largest and lowest AlphaAmplitudeAvg value of each bin
    max_alpha = max(DATA.AlphaAmplitudeAvg(DATA.alphaBin == b));
    min_alpha = min(DATA.AlphaAmplitudeAvg(DATA.alphaBin == b));
    fprintf('Bin %d: Max AlphaAmplitudeAvg = %.2f, Min AlphaAmplitudeAvg = %.2f\n', b, max_alpha, min_alpha);
end


%%
figure;

datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};
outcome_types = {'Subjective', 'Objective'};

% Loop through each combination of dataset and outcome type
for d = 1:length(datasets)
    for o = 1:length(outcome_types)

 
        DATA = datasets{d}{1};
        
        a = DATA.AlphaAmplitudeAvg;
        edges = quantile(a, [0 1/2 1]);
        DATA.alphaBin = discretize(a, edges);
        
        
        intensities = unique(DATA.StimIntensity);
        options = struct;
        options.sigmoidName = 'logistic';
        %options.fixedPars   = [0.5 NaN NaN NaN NaN]; 
        
        colors = lines(3);
        labels = {'Low alpha','High alpha'};
        
        % Create a dense x-axis for smooth curves
        x_fit = linspace(min(intensities), max(intensities), 300);
        
        %figure; hold on;
        
        for b = 1:2
            data_ps = [ intensities(:), nSeen_all(b,:)', nTotal_all(b,:)' ];
        
            % remove intensities with 0 trials
            data_ps = data_ps(data_ps(:,2)>0, :);
            
            result = psignifit(data_ps, options);
            %nexttile;
            %plotPsych(result);
        
            %title(['Alpha bin ' num2str(b)]);
            params = result.Fit;
        
            x_fit = linspace(min(intensities), max(intensities), 300);
        
        % --- Inside the 'for' loop: ---
        % Original (Failing):
        % y_fit = result.options.core(params, x_fit);           % evaluate psychometric function
        
        % --- FIXED LINE: Use the correct function handle ---
            y_fit = result.psiHandle([x_fit(:), repmat(params', length(x_fit), 1)]);
        % ----------------------------------------------------
        
            
            % Plot curve
            plot(x_fit, y_fit(:,1), 'LineWidth', 2, 'Color', colors(b,:));
            hold on;
        
                % confidence interval bounds (95% CI → index 2)
            params_lo = result.conf_Intervals(:,1,1);
            params_hi = result.conf_Intervals(:,2,1);
            
            % evaluate lower & upper psychometric curves
            y_lo = result.psiHandle([x_fit(:), repmat(params_lo', length(x_fit), 1)]);
            y_hi = result.psiHandle([x_fit(:), repmat(params_hi', length(x_fit), 1)]);
            
            % plot confidence interval as dashed lines
            %plot(x_fit, y_lo(:,1), '--', 'Color', colors(b,:), 'LineWidth', 3);
            hold on;
            %plot(x_fit, y_hi(:,1), '--', 'Color', colors(b,:), 'LineWidth', 3);
            hold on;
            
            % Plot empirical points
            idx = data_ps(:,2) > 0;
            scatter(data_ps(idx,1), data_ps(idx,2) ./ data_ps(idx,3), 40, ...
                    'MarkerFaceColor', colors(b,:), ...
                    'MarkerEdgeColor', colors(b,:), 'HandleVisibility','off');
        end
        
        xlabel('Stimulus intensity');
        if o==1
            ylabel('P(seen)');
        else
            ylabel('P(correct)');
        end

        title('Fitted psychometric curves for alpha levels');
        legend(labels, 'Location','best');
        grid on;
    end
end

%%
% Create a 2x3 figure for subplots
figure;
numRows = 2;
numCols = 3;

% Loop through each combination of outcome type and dataset for subplots
for o = 1:length(outcome_types)
    for d = 1:length(datasets)
        subplot(numRows, numCols, (o-1)*numCols + d);
        
        DATA = datasets{d}{1};
        
        a = DATA.AlphaAmplitudeAvg;
        edges = quantile(a, [0 .25 .75 1]); % quartiles
        idx_low  = a <= edges(2);
        idx_high = a >= edges(3);
        
        DATA.alphaBin = zeros(size(a));
        DATA.alphaBin(idx_low)  = 1;
        DATA.alphaBin(idx_high) = 2;

        % and ignore trials with alphaBin==0

        
        options = struct;
        options.sigmoidName = 'logistic';

        nSeen_all = zeros(2, nInt);
        nTotal_all = zeros(2, nInt);
        
        colors = lines(3);
        labels = {'Low alpha','High alpha'};
        
        % Create a dense x-axis for smooth curves
        x_fit = linspace(min(intensities), max(intensities), 300);
        
        for b = 1:2

        
            idx = DATA.alphaBin == b;
        
            for ii = 1:nInt
                curr = DATA.StimIntensity == intensities(ii) & idx;
        
                % Count trials at this intensity
                nTotal_all(b,ii) = sum(curr);
        
                % Count "seen" responses
                if o==1
                    nSeen_all(b,ii) = sum(DATA.SubjectiveOutcome(curr) == 1);
                else
                    nSeen_all(b,ii) = sum(DATA.ObjectiveOutcome(curr) == 1);
                end

            end


            data_ps = [ intensities(:), nSeen_all(b,:)', nTotal_all(b,:)' ];
            data_ps = data_ps(data_ps(:,2)>0, :);
            
            result = psignifit(data_ps, options);
            params = result.Fit;
            
            y_fit = result.psiHandle([x_fit(:), repmat(params', length(x_fit), 1)]);
            
            % Plot curve
            plot(x_fit, y_fit(:,1), 'LineWidth', 2, 'Color', colors(b,:));
            hold on;
            
            params_lo = result.conf_Intervals(:,1,1);
            params_hi = result.conf_Intervals(:,2,1);
            
            y_lo = result.psiHandle([x_fit(:), repmat(params_lo', length(x_fit), 1)]);
            y_hi = result.psiHandle([x_fit(:), repmat(params_hi', length(x_fit), 1)]);
            
            idx = data_ps(:,2) > 0;
            scatter(data_ps(idx,1), data_ps(idx,2) ./ data_ps(idx,3), 40, ...
                    'MarkerFaceColor', colors(b,:), ...
                    'MarkerEdgeColor', colors(b,:), 'HandleVisibility','off');
        end
        
        xlabel('Stimulus intensity');
        if o==1
            ylabel('P(seen)');
        else
            ylabel('P(correct)');
        end

        title(['Dataset: ' datasets{d}{2} ', Outcome: ' outcome_types{o}]);
        legend(labels, 'Location','best');
        grid on;
    end
end
%%

% Split data
low  = obj_subj(DATA(DATA.alphaBin==1, :));
high = obj_subj(DATA(DATA.alphaBin==2, :));

p_low  = polyfit(low.ObjectiveOutcome,  low.SubjectiveOutcome, 1);
p_high = polyfit(high.ObjectiveOutcome, high.SubjectiveOutcome, 1);

xfit = linspace(0,1,100);
yfit_low  = polyval(p_low,  xfit);
yfit_high = polyval(p_high, xfit);

figure; hold on;

% Scatter points
scatter(low.ObjectiveOutcome,  low.SubjectiveOutcome, 50, "blue", "filled");
scatter(high.ObjectiveOutcome, high.SubjectiveOutcome,50, "red",  "filled");

% Fitted lines
plot(xfit, yfit_low,  "blue", "LineWidth",2);
plot(xfit, yfit_high, "red",  "LineWidth",2);

xlabel("Correctness (mean per stimulus)");
ylabel("Subjective report (mean per stimulus)");
legend("Low α","High α","Low α fit","High α fit","Location","best");
title("Subjective report vs correctness (per stimulus)");

grid on;
ylim([0 1]);
xlim([0 1]);


%%

%% ------------------------------
% 1. Load data + alpha bins
% ------------------------------
DATA = MasterTable_Irregular;
a = DATA.AlphaAmplitudeAvg;
edges = quantile(a, [0 1/2 1]);
DATA.alphaBin = discretize(a, edges);   % 1 = low alpha, 2 = high alpha

%% ------------------------------
% 2. Prediction grid
% ------------------------------
s_grid = linspace(min(DATA.StimIntensityZ), max(DATA.StimIntensityZ), 200)';

%% ------------------------------
% 3. Compute mean alpha in each bin
% ------------------------------
alpha_low  = mean(DATA.AlphaAmplitudeAvg(DATA.alphaBin == 1));
alpha_high = mean(DATA.AlphaAmplitudeAvg(DATA.alphaBin == 2));

%% ------------------------------
% 4. Build table for LOW alpha
% ------------------------------
SubjectDummy = DATA.SubjectID(1);  % use any existing subject ID

newT_low = table();

newT_low.SubjectID          = repmat(SubjectDummy, length(s_grid), 1);
newT_low.ObjectiveOutcome   = zeros(length(s_grid),1);   % dummy
newT_low.StimZc             = s_grid;
newT_low.StimZ2             = s_grid.^2;
newT_low.StimZ3             = s_grid.^3;
newT_low.StimZ4             = s_grid.^4;

newT_low.AlphaAmplitudeAvg  = repmat(alpha_low, length(s_grid),1);

newT_low.Alpha_StimZ        = newT_low.AlphaAmplitudeAvg .* newT_low.StimZc;
newT_low.Alpha_StimZ2       = newT_low.AlphaAmplitudeAvg .* newT_low.StimZ2;
newT_low.Alpha_StimZ3       = newT_low.AlphaAmplitudeAvg .* newT_low.StimZ3;
newT_low.Alpha_StimZ4       = newT_low.AlphaAmplitudeAvg .* newT_low.StimZ4;


%% ------------------------------
% 5. Build table for HIGH alpha
% ------------------------------
newT_high = table();

newT_high.SubjectID          = repmat(SubjectDummy, length(s_grid), 1);
newT_high.ObjectiveOutcome   = zeros(length(s_grid),1);   % dummy
newT_high.StimZc             = s_grid;
newT_high.StimZ2             = s_grid.^2;
newT_high.StimZ3             = s_grid.^3;
newT_high.StimZ4             = s_grid.^4;

newT_high.AlphaAmplitudeAvg  = repmat(alpha_high, length(s_grid),1);

newT_high.Alpha_StimZ        = newT_high.AlphaAmplitudeAvg .* newT_high.StimZc;
newT_high.Alpha_StimZ2       = newT_high.AlphaAmplitudeAvg .* newT_high.StimZ2;
newT_high.Alpha_StimZ3       = newT_high.AlphaAmplitudeAvg .* newT_high.StimZ3;
newT_high.Alpha_StimZ4       = newT_high.AlphaAmplitudeAvg .* newT_high.StimZ4;


%% ------------------------------
% 6. Predict GLME probabilities
% ------------------------------
p_low  = predict(glme_poly, newT_low,  'Conditional', false);
p_high = predict(glme_poly, newT_high, 'Conditional', false);


%% ------------------------------
% 7. Plot
% ------------------------------
figure; hold on;
plot(s_grid, p_low,  'LineWidth', 2);
plot(s_grid, p_high, 'LineWidth', 2);

xlabel('StimZc');
ylabel('Predicted P(response=1)');
legend({'Low alpha','High alpha'});
title('GLME Predicted Psychometric Curves (Two Alpha Bins)');
grid on;


%% 
% Load datasets
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};
beta_diff = struct();

for d = 1:length(datasets)
    dataset_name = datasets{d}{2};
    DATA = datasets{d}{1}; % Assume a function to load the dataset

    % Fit GLME for objective outcome
    glme_obj = fitglme(DATA, 'ObjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)');
    beta_obj = fixedEffects(glme_obj);

    % Fit GLME for subjective outcome
    glme_subj = fitglme(DATA, 'SubjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)');
    beta_subj = fixedEffects(glme_subj);

    % Calculate the difference
    beta_diff.(dataset_name) = beta_subj(2) - beta_obj(2); % Assuming the second coefficient is for AlphaAmplitudeAvg
    fprintf('Difference in estimates for %s: %.4f\n', dataset_name, beta_diff.(dataset_name));

end


%%

