%% DATA ANALYSIS

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

tic

% Load the MasterTable (contains SubjectID, AlphaPower, StimIntensity, SubjectiveOutcome)
%load(master_table_file, 'MasterTable_Mouihbi');
load(master_table_file, 'MasterTable');
toc
disp('Master Table loaded successfully.');

%MasterTable = MasterTable_Mouihbi;

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
PRE_SAMPLES = PRE_EVENT_SEC * Fs; % 500 samples
dims = size(MasterTable.AlphaAmplitude{1});
N_samples = dims(1);
time_axis_ms = ((1:N_samples) - PRE_SAMPLES - 1) * (1000 / Fs);
POST_EVENT_SEC = 0.1;
time_samples = size(MasterTable.AlphaAmplitude{1});
time_samples = time_samples(1);
N_TIME_BINS = 10;
N_FREQ_BINS = 5;
alpha_freq_range = [8 12];
time_window_sec = [-0.495 0.005];


% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 28, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};

% Raw or baseline-corrected, starts true
RAW=true;

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


tic
if length(size(MasterTable.Baseline{1})) > 2
    for i = 1:height(MasterTable)
        MasterTable.Baseline{i} = squeeze(mean(MasterTable.Baseline{i}(:, :, currentROI), 3));
    end
end
toc
disp('All single-trial Baselines are now averaged over ROI channels.');



%% BASELINE PER TRIAL AND Z-SCORE PER PARTICIPANT

MasterTable= baseline_and_subject_zscore(MasterTable, currentROI, false);
RAW=false;
disp('All single-trial baseline-corrected and z-scored per participant.');



%% CREATE ONE AVERAGE ALPHA AMPLITUDE

% Here we will take one alpha predictor and not each time-frequency
% combination, choice is informed visually by the heatmap results.

time_pred_bin = [-0.495; -0.050];
freq_pred_bin = [8; 12];

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

% StimIntensity fix for convenience
MasterTable.StimIntensityZ = MasterTable.StimIntensity;

% Now use the numerical codes for the comparison logic

% 1. MasterTable_Predictive (Condition < 3: Keep Rhythm and Interval)
MasterTable_Predictive = MasterTable(condition_codes < 3, :);

% 1.a Rhythm
MasterTable_Rhythm = MasterTable(condition_codes == 1, :);

% 1.b Interval
MasterTable_Interval = MasterTable(condition_codes == 2, :);

% 2. MasterTable_Irregular (Condition == 3: Keep Irregular)
MasterTable_Irregular = MasterTable(condition_codes == 3, :);

%%  Make Irregular comparable with others

% Create a logical index that is TRUE for every row that meets BOTH criteria
logical_index = (MasterTable_Irregular.IrregularTargetTime >= 2) & (MasterTable_Irregular.IrregularTargetTime <= 4);

% Use the logical index to select the matching rows into a new table
MasterTable_Irregular = MasterTable_Irregular(logical_index, :);


%% CONDITIONS AND OUTCOMES


% Generate a 3x2 figure for different outcome types and datasets
outcome_types = {'Objective', 'Subjective'};
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};

%% TIME COURSES

figure;
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
            %time_course_data = cellfun(@(x) x(:,10).', ...
                %time_series_cells, 'UniformOutput', false);

                        % Define the start and end indices of the frequency band you want to average over
            %start_index = 8;  % e.g., for 8 Hz
            %end_index = 12;   % e.g., for 12 Hz
            colors = {'r', 'b'};
            
            % Average across the specified column range (start_index to end_index)
            time_course_data = cellfun(@(x) mean(x(:, freq_start_sample:freq_end_sample), 2).', ...
                time_series_cells, 'UniformOutput', false);

            % Stack into matrix: nTrials × nTime
            M = cell2mat(time_course_data);

            % Average time course
            avg_tc = mean(M, 1);

            % --- CALCULATIONS FOR SHADOW ---
            std_tc = std(M, 0, 1); % Compute standard deviation across trials (dimension 1)
            
            upper_bound = avg_tc + std_tc;
            lower_bound = avg_tc - std_tc;

            % --- PLOTTING THE SHADOW (FILL) ---
            % x-coordinates for the fill: [time_axis, reversed time_axis]
            X_fill = [time_axis_ms, fliplr(time_axis_ms)];
            % y-coordinates for the fill: [lower_bound, reversed upper_bound]
            Y_fill = [lower_bound, fliplr(upper_bound)];
            
            % Use the 'fill' function to draw the shaded area
            fill(X_fill, Y_fill, colors{o+1}, ... % Use color corresponding to 'o'
                 'FaceAlpha', 0.15, ... % Transparency of the shadow
                 'EdgeColor', 'none', ... % No border around the shadow
                 'HandleVisibility', 'off'); % Prevents shadow from cluttering legend

            % Plot
            plot(time_axis_ms, avg_tc, 'LineWidth', 1.8, 'DisplayName', sprintf('Outcome = %d', o));
        end
        
        title(sprintf('%s – %s', outcome_types{row}, datasets{col}{2}));
        xlabel('Time');
        %ylim([0. 1.3])
        ylabel('Alpha Amplitude (8-12 Hz)');
        xline(0, '--k', 'DisplayName', 'Target');
        legend('show', 'Location','best');
        grid on;
        hold off;
    end
end

%% OVERALL WEIGHTS 

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
    model_formula_subjective = ['SubjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)'];
    glme_subjective = fitglme(data_condition, model_formula_subjective, ...
                               'Distribution', 'Binomial', ...
                               'Link', 'logit', 'FitMethod', 'Laplace');
    subjective_betas(col) = glme_subjective.Coefficients.Estimate(2);
    all_CIs = coefCI(glme_subjective, 'Alpha', 0.1); 
    subjective_CI(:, col) = all_CIs(2,:);
    
    % Fit GLME for Objective
    model_formula_objective = ['ObjectiveOutcome ~ AlphaAmplitudeAvg + (1|SubjectID)'];

    glme_objective = fitglme(data_condition, model_formula_objective, ...
                              'Distribution', 'Binomial', ...
                              'Link', 'logit', 'FitMethod', 'Laplace');
    objective_betas(col) = glme_objective.Coefficients.Estimate(2);
    all_CIs = coefCI(glme_objective, 'Alpha', 0.1); 
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
ylim([-0.1 0.1]);
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


%% TIME-FREQUENCY

tic
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
        clim([-0.1, 0.1]); 
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
        %sig_neg_mask = (PValueMap' < 0.1);
        % 2. Plot the contour lines using the mask
        [~, h_contour] = contour(TimeBins, FreqBins, sig_neg_mask, [0.5 0.5], 'LineWidth', 3, 'LineColor', 'w', 'LineStyle', ':');
        colorbar;

    end
end
toc
%%

% --- PARAMETERS ---
time_bins_edges = -0.495:0.050:-0.395;   % in seconds, edges of bins
freq_pred_bin = [8 12]; 
all_freqs = alpha_freq_range(1):alpha_freq_range(2); 
freq_start_idx = find(all_freqs==freq_pred_bin(1));
freq_end_idx = find(all_freqs==freq_pred_bin(2));% Hz, frequency range of interest
degree = 3;                              % Polynomial degree
time_bins_edges_ms = time_bins_edges*1000;
f = figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
N_rows = length(outcome_types); % 2
N_cols = length(datasets);      % 3

% --- LOOP OVER DATASETS AND OUTCOMES ---
for d = 1:length(datasets) % This iterates over the COLUMNS
    DATA = datasets{d}{1};
    condition_name = datasets{d}{2};
    
    for o = 1:length(outcome_types) % This iterates over the ROWS
        
        % Calculate the subplot index
        % Index = (Column Index - 1) * N_rows + Row Index
        plot_index = (o - 1) * N_cols + d;
        
        ax = subplot(N_rows, N_cols, plot_index); % Use the calculated index
        hold on;

        outcome_name = outcome_types{o};
        
        slopes_time = nan(length(time_bins_edges)-1, 100); % rows=time bins, cols=stim intensity grid
        s_grid = linspace(min(DATA.StimIntensityZ), max(DATA.StimIntensityZ), 100)';
        s_grid_plot = linspace(min(DATA.StimIntensity), max(DATA.StimIntensity), 100)';
        
        % --- LOOP OVER TIME BINS ---
        for tb = 1:length(time_bins_edges)-1
            t_start_sec = time_bins_edges(tb);
            t_end_sec   = time_bins_edges(tb+1);
            
            % Convert to sample indices
            t_start_samp = round((t_start_sec + PRE_EVENT_SEC)*Fs);
            t_end_samp   = round((t_end_sec + PRE_EVENT_SEC)*Fs);
            
            % --- 1. Average alpha over this time bin and frequency range ---
            AlphaAmplitudeAvg_bin = nan(height(DATA),1);
            for i = 1:height(DATA)
                alpha_cell = DATA.AlphaAmplitude{i};
                AlphaAmplitudeAvg_bin(i) = squeeze(mean(mean(alpha_cell(t_start_samp:t_end_samp, freq_start_idx:freq_end_idx),1),2));
            end
            DATA.AlphaAmplitudeAvg = AlphaAmplitudeAvg_bin;
            DATA.StimZc = DATA.StimIntensityZ;
            
            % --- 2. Polynomial interaction terms (degree 3) ---
            DATA.StimAlpha  = DATA.AlphaAmplitudeAvg .* DATA.StimZc;
            DATA.StimZ2     = DATA.StimZc.^2; DATA.StimAlpha2 = DATA.AlphaAmplitudeAvg .* DATA.StimZ2;
            DATA.StimZ3     = DATA.StimZc.^3; DATA.StimAlpha3 = DATA.AlphaAmplitudeAvg .* DATA.StimZ3;
            
            % --- 3. GLME formula ---
            if strcmp(outcome_name,'Subjective')
                formula = 'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimAlpha + StimZ2 + StimAlpha2 + StimZ3 + StimAlpha3 + (1|SubjectID)';
            else
                formula = 'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimZc + StimAlpha + StimZ2 + StimAlpha2 + StimZ3 + StimAlpha3 + (1|SubjectID)';
            end

            % for numerical stability for cubic interaction
            DATA = DATA(DATA.StimIntensity < 10, :);

            % --- 4. Fit GLME ---
            glme = fitglme(DATA, formula, 'Distribution','Binomial','Link','logit','FitMethod','Laplace');
            
            % --- 5. Extract slope: AlphaAmplitude effect across stim intensity grid ---
            tbl = glme.Coefficients;
            b1 = tbl.Estimate(strcmp(tbl.Name,'AlphaAmplitudeAvg'));
            b2 = tbl.Estimate(strcmp(tbl.Name,'StimAlpha'));
            b3 = tbl.Estimate(strcmp(tbl.Name,'StimAlpha2'));
            b4 = tbl.Estimate(strcmp(tbl.Name,'StimAlpha3'));
            
            slopes_time(tb,:) = b1 + b2.*s_grid' + b3.*(s_grid'.^2) + b4.*(s_grid'.^3);            
        end
        
        % --- 6. Plot heatmap ---
        time_centers = (time_bins_edges_ms(1:end-1) + time_bins_edges_ms(2:end)) / 2;
        
        imagesc(time_centers, s_grid_plot, slopes_time'); 
        clim([-0.1 0.1]);
        set(gca,'YDir','reverse');
        xlim([min(time_centers) max(time_centers)]);
        ylim([min(s_grid) max(s_grid)]);
        axis tight;
        colorbar;
        xlabel('Time relative to event (ms)');
        ylabel('Stimulus intensity');
        title(sprintf('%s – %s: time-resolved alpha slope', condition_name, outcome_name));
    end
end

%% SUBJECTIVE INFLATION

% --- Settings ---
[Xvar, Yvar] = deal("ObjectiveOutcome", "SubjectiveOutcome");

datasets = {
    {MasterTable_Rhythm,   'Rhythm'}, 
    {MasterTable_Interval, 'Interval'}, 
    {MasterTable_Irregular,'Irregular'}
};

stim_levels = 1:10; % or unique(DATA.StimIntensity)

figure;
set(gcf, 'Position', [100 100 1000 400]);  % wider and shorter


for d = 1:length(datasets)

    DATA = datasets{d}{1};
    dataset_name = datasets{d}{2};

    % --- Compute terciles of alpha ---
    alpha = DATA.AlphaAmplitudeAvg;
    alpha = alpha(:); 
    edges = quantile(alpha, [0 1/3 2/3 1]);
    DATA.alphaTercile = discretize(alpha, edges);

    % --- Prepare subplot ---
    subplot(1,3,d); hold on;

    % -------- LOW TERCILE --------
    D_low = DATA(DATA.alphaTercile == 1, :);

    mean_obj_low  = zeros(numel(stim_levels),1);
    mean_sub_low  = zeros(numel(stim_levels),1);

    for s = 1:numel(stim_levels)
        subset = D_low(D_low.StimIntensity == stim_levels(s), :);
        mean_obj_low(s) = mean(double(subset.(Xvar)), 'omitnan');
        mean_sub_low(s) = mean(double(subset.(Yvar)), 'omitnan');
    end

    % -------- HIGH TERCILE --------
    D_high = DATA(DATA.alphaTercile == 3, :);

    mean_obj_high = zeros(numel(stim_levels),1);
    mean_sub_high = zeros(numel(stim_levels),1);

    for s = 1:numel(stim_levels)
        subset = D_high(D_high.StimIntensity == stim_levels(s), :);
        mean_obj_high(s) = mean(double(subset.(Xvar)), 'omitnan');
        mean_sub_high(s) = mean(double(subset.(Yvar)), 'omitnan');
    end

    % -------- SCATTER POINTS --------
    scatter(mean_obj_low,  mean_sub_low,  60, "blue", "filled");
    scatter(mean_obj_high, mean_sub_high, 60, "red",  "filled");

    % -------- FIT LINES --------
    p_low  = polyfit(mean_obj_low,  mean_sub_low,  1);
    p_high = polyfit(mean_obj_high, mean_sub_high, 1);

    xf = linspace(0,1,100);
    plot(xf, polyval(p_low,  xf), "blue", "LineWidth", 2);
    plot(xf, polyval(p_high, xf), "red",  "LineWidth", 2);

    % -------- Formatting --------
    title(dataset_name);
    xlabel("Objective performance (mean per stimulus)");
    ylabel("Subjective performance (mean per stimulus)");
    xlim([0.5 1]); ylim([0 1]);
    grid on;

    if d == 1
        legend("Low α","High α","Low α fit","High α fit","Location","best");
    end
end

%%

% --- Settings ---
[Xvar, Yvar] = deal("ObjectiveOutcome", "SubjectiveOutcome");

datasets = {
    {MasterTable_Rhythm,   'Rhythm'}, 
    {MasterTable_Interval, 'Interval'}, 
    {MasterTable_Irregular,'Irregular'}
};

stim_levels = 1:10;

figure;
set(gcf, 'Position', [100 100 1000 400]);

for d = 1:length(datasets)

    DATA = datasets{d}{1};
    dataset_name = datasets{d}{2};

    % --- Compute terciles of alpha ---
    alpha = DATA.AlphaAmplitudeAvg(:);
    edges = quantile(alpha, [0 1/3 2/3 1]);
    DATA.alphaTercile = discretize(alpha, edges);

    subjects = unique(DATA.SubjectID);

    % Storage for GROUP regression
    all_obj_low  = [];  % each row = a subject-level point
    all_sub_low  = [];
    all_obj_high = [];
    all_sub_high = [];

    subplot(1,3,d); hold on;

    % =====================================================================
    % LOOP SUBJECTS
    % =====================================================================
    for si = 1:numel(subjects)
        subj = subjects(si);
        Dsub = DATA(DATA.SubjectID == subj, :);
        Dsub = Dsub(Dsub.StimIntensity <8, :);

        % -------- LOW tercile --------
        D_low = Dsub(Dsub.alphaTercile == 1, :);

        for s = 1:numel(stim_levels)
            sel = D_low(D_low.StimIntensity == stim_levels(s), :);
            if height(sel) > 0
                all_obj_low(end+1,1) = mean(double(sel.(Xvar)), 'omitnan');
                all_sub_low(end+1,1) = mean(double(sel.(Yvar)), 'omitnan');
            end
        end

        % -------- HIGH tercile --------
        D_high = Dsub(Dsub.alphaTercile == 3, :);

        for s = 1:numel(stim_levels)
            sel = D_high(D_high.StimIntensity == stim_levels(s), :);
            if height(sel) > 0
                all_obj_high(end+1,1) = mean(double(sel.(Xvar)), 'omitnan');
                all_sub_high(end+1,1) = mean(double(sel.(Yvar)), 'omitnan');
            end
        end
    end

    % =====================================================================
    % SCATTER ALL SUBJECT-LEVEL POINTS
    % =====================================================================
    scatter(all_obj_low,  all_sub_low,  40, "blue", "filled", "MarkerFaceAlpha",0.6);
    scatter(all_obj_high, all_sub_high, 40, "red",  "filled", "MarkerFaceAlpha",0.6);

    % =====================================================================
    % FIT GROUP LINES ACROSS SUBJECT-LEVEL POINTS
    % =====================================================================
    if numel(all_obj_low) > 2
        p_low  = polyfit(all_obj_low,  all_sub_low, 1);
        xf = linspace(0,1,100);
        plot(xf, polyval(p_low, xf), "blue", "LineWidth", 2);
    end

    if numel(all_obj_high) > 2
        p_high = polyfit(all_obj_high, all_sub_high, 1);
        xf = linspace(0,1,100);
        plot(xf, polyval(p_high, xf), "red", "LineWidth", 2);
    end

    % --- Formatting ---
    title(dataset_name);
    xlabel("Objective performance (per-subject mean)");
    ylabel("Subjective performance (per-subject mean)");
    xlim([0.5 1]); ylim([0 1]);
    grid on;

    if d == 1
        legend("Low α (subj points)", "High α (subj points)", ...
               "Low α fit", "High α fit", "Location","best");
    end
end





%% PSYCHOMETRIC FUNCTIONS WITH PSIGNIFIT
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
figure;
numRows = 2;
numCols = 3;

for o = 1:length(outcome_types)
    for d = 1:length(datasets)
        
        subplot(numRows, numCols, (o-1)*numCols + d);
        hold on;
        
        DATA = datasets{d}{1};
        intensities = unique(DATA.StimIntensity);
        nInt = length(intensities);

        % ----------------------------------------------------
        % 1. BIN ALPHA INTO LOW/HIGH (PER-PARTICIPANT TERCILES)
        % ----------------------------------------------------
        
        % Initialize the bin column: 0 means middle tercile (to be removed)
        DATA.alphaBin = zeros(height(DATA), 1);
        
        % Get unique Subject IDs
        subjectIDs = unique(DATA.SubjectID);
        
         % Loop through each participant to calculate individual tercile edges
        for subj = 1:length(subjectIDs)
            currentID = subjectIDs(subj);
            
            % Create a mask for the current participant's data
            subjMask = (DATA.SubjectID == currentID);
            
            % Extract the alpha values for this subject
            alpha_subj = DATA.AlphaAmplitudeAvg(subjMask);
            
            % Ensure there is enough data to calculate terciles
            if length(alpha_subj) < 3
                % Skip subject if there are too few data points
                continue;
            end
            
            % Calculate individual tercile edges (33rd and 67th percentiles)
            % [0 .33 .67 1] are the quantiles.
            edges_subj = quantile(alpha_subj, [0 .33 .67 1]);
            
            % --- REVISED BIN ASSIGNMENT ---
            % We will use the 'find' function on the subjMask to get the original row
            % indices, and then use the *subject-specific* logical mask (lowMaskSubj)
            % to sub-index those original row numbers.
            
            % Get the list of row indices for the current subject
            original_rows = find(subjMask);
            
            % Identify rows within this subject's data that fall into the bins
            lowMaskSubj  = alpha_subj <= edges_subj(2);
            highMaskSubj = alpha_subj >= edges_subj(3);
            
            % Assign bin values back to the main DATA table using the revised indexing
            
            % Bin 1: Low Alpha (below 33rd percentile for this participant)
            % Use the original row indices, filtered by the low alpha mask
            lowAlphaRows = original_rows(lowMaskSubj);
            DATA.alphaBin(lowAlphaRows) = 1;
            
            % Bin 2: High Alpha (above 67th percentile for this participant)
            % Use the original row indices, filtered by the high alpha mask
            highAlphaRows = original_rows(highMaskSubj);
            DATA.alphaBin(highAlphaRows) = 2;
        end
        % Remove alphaBin == 0 (middle quartiles)
        validMask = DATA.alphaBin ~= 0;
        
        DATA = DATA(validMask, :);
        
        % -------------------------
        % 2. PSIGNIFIT SETTINGS
        % -------------------------
        options = struct;
        options.sigmoidName = 'logistic';

        nSeen_all  = zeros(2, nInt);
        nTotal_all = zeros(2, nInt);
        
        colors = lines(3);
        labels = {'Low alpha','High alpha'};
        x_fit = linspace(min(intensities), max(intensities), 300);

        % -------------------------
        % 3. LOOP LOW/HIGH ALPHA
        % -------------------------
        for b = 1:2
            
            binMask = (DATA.alphaBin == b);
            
            for ii = 1:nInt
                intensityMask = (DATA.StimIntensity == intensities(ii));
                useMask = binMask & intensityMask;
                
                nTotal_all(b,ii) = sum(useMask);
                
                if o == 1
                    nSeen_all(b,ii) = sum(DATA.ObjectiveOutcome(useMask) == 1);
                else
                    nSeen_all(b,ii) = sum(DATA.SubjectiveOutcome(useMask) == 1);
                end
            end
            
            % -----------------------------------
            % Build data matrix for psignifit
            % Only keep rows with total > 0
            % -----------------------------------
            data_ps = [ intensities(:), nSeen_all(b,:)', nTotal_all(b,:)' ];
            data_ps = data_ps(data_ps(:,3) > 0, :);

            if isempty(data_ps)
                warning('No valid data for bin %d in dataset %s', b, datasets{d}{2});
                continue;
            end
            
            % -------------------------
            % 4. FIT THE PSYCHOMETRIC CURVE
            % -------------------------
            result = psignifit(data_ps, options);
            params = result.Fit;   % fitted parameters

            y_fit = result.psiHandle([x_fit(:), repmat(params', length(x_fit), 1)]);

            plot(x_fit, y_fit(:,1), 'LineWidth', 2, 'Color', colors(b,:));

            % -------------------------
            % 5. CONFIDENCE INTERVALS
            % -------------------------
            %params_lo = result.conf_Intervals(:,1,1);
            %params_hi = result.conf_Intervals(:,2,1);
            
            %y_lo = result.psiHandle([x_fit(:), repmat(params_lo', length(x_fit), 1)]);
            %y_hi = result.psiHandle([x_fit(:), repmat(params_hi', length(x_fit), 1)]);
            
            % (Optional) Fill CI shading
            fill([x_fit fliplr(x_fit)], ...
                 [y_lo(:,1)' fliplr(y_hi(:,1)')], ...
                 colors(b,:), ...
                 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
                 'HandleVisibility','off');

            % -------------------------
            % 6. SCATTER REAL DATA
            % -------------------------
            scatter(data_ps(:,1), data_ps(:,2) ./ data_ps(:,3), 40,...
                'MarkerFaceColor', colors(b,:), ...
                'MarkerEdgeColor', colors(b,:), ...
                'HandleVisibility','off');
        end
        
        xlabel('Stimulus intensity');

        if o == 1
            ylabel('P(correct)');
        else
            ylabel('P(seen)');
        end

        title(sprintf('%s — %s', datasets{d}{2}, outcome_types{o}));

        legend(labels, 'Location','best');
        grid on;
        hold off;
    end
end

%%
%%

% function sigMask = fdr_mask(pMap, q)
%     if nargin < 2
%         q = 0.05; 
%     end
% 
%     % flatten
%     p = pMap(:);
%     m = numel(p);
% 
%     % sort
%     [p_sorted, idx] = sort(p);
% 
%     % BH criterion
%     thresh = (1:m)'/m * q;
% 
%     % find largest k where p(k) <= thresh(k)
%     k = find(p_sorted <= thresh, 1, 'last');
% 
%     sigMask = false(size(p));
% 
%     if ~isempty(k)
%         sigMask(idx(1:k)) = true;
%     end
% 
%     % reshape back
%     sigMask = reshape(sigMask, size(pMap));
% end




%%



%% 0) SETTINGS (edit as needed)
Fs = 1024;                                 % sampling rate
PRE_EVENT_SEC = 0.500;                     % alignment (target at PRE_EVENT_SEC)
time_window_sec = [-0.495, 0.005];               % window to analyze (sec relative to target)
freq_range_hz = [8 12];                    % alpha band to collapse
N_TIME_BINS = 20;                          % number of time bins across window
criterion = 0.1;  %because directional                         % cluster forming p-value (inside function)
nPerms = 10000;    % permutations for cluster test (adjust)



figure('Units','normalized','Position',[0.05 0.05 0.9 0.8]); % adjust figure size

for r = 1:length(datasets)        % loop over datasets → rows
    T = datasets{r}{1};
    subjects = cellstr(unique(T.SubjectID));
    for c = 1:length(outcome_types)  % loop over outcomes → columns
        outcome = outcome_types{c};
        
        ax = subplot(length(datasets), length(outcome_types), (r-1)*length(outcome_types) + c);
        hold(ax,'on');

        % 1) Unpack data & basic indices  
        % Ensure SubjectID is categorical or string
       % if ~iscategorical(T.SubjectID)
        %    T.SubjectID = categorical(T.SubjectID);
        %end
        %subjects = categories(T.SubjectID);
        nSubj = numel(subjects);
        nTrialsTotal = height(T);
        
        % Unpack AlphaAmplitude cell to a 3D array [time x freqs x trials]
        alpha_data_cube = cat(3, T.AlphaAmplitude{:});  % time x freqs x trials
        [totalSamples, nFreqs, ~] = size(alpha_data_cube);
        
        % Map freq_range_hz to frequency indices (assumes all_freqs are integers starting freq_range_hz(1):freq_range_hz(2))
        all_freqs = freq_range_hz(1):freq_range_hz(2);
        % If your alpha_data_cube second dim corresponds to frequencies in `all_freqs`, use:
        freq_idx = 1:nFreqs; % if second dim already matches the small range
        % If you have a full freq vector, replace above with proper mapping.
        
        % 2) Time bin indices
        time_zero_sample = round(PRE_EVENT_SEC * Fs);
        start_sample = time_zero_sample + round(time_window_sec(1) * Fs);
        end_sample   = time_zero_sample + round(time_window_sec(2) * Fs);
        total_window_samples = end_sample - start_sample + 1;
        samples_per_time_bin = floor(total_window_samples / N_TIME_BINS);
        
        % Precompute time bin centers in ms
        time_bin_centers = zeros(N_TIME_BINS,1);
        for tb = 1:N_TIME_BINS
            s = start_sample + (tb-1)*samples_per_time_bin;
            e = s + samples_per_time_bin - 1;
            time_bin_centers(tb) = ((s+e)/2 - time_zero_sample) * 1000 / Fs; % ms relative to target
        end
        
        % 3) Build per-subject beta matrix (subjects x timeBins)
        betaMat = nan(nSubj, N_TIME_BINS);
        warning('off','all'); % suppress warnings inside loop; remove if you want them


        for si = 1:nSubj
            subjLabel = subjects{si};             % categorical ID as char
            subj_idx = (T.SubjectID == subjLabel);       % logical indexing
            Tsub = T(subj_idx, :);
            subjTrials = find(subj_idx); % indices into alpha_data_cube third dim

            %fprintf('Subject %d: %d trials\n', subjLabel, sum(subj_idx));

            % require at least some trials
            if numel(subjTrials) < 1
                % too few trials to fit reliable logistic; leave NaNs
                continue;
            end
        
            % construct subject-level predictors/outcome per trial for speed
            % pre-extract outcome and stim for this subject
            if strcmpi(outcome, 'Subjective')
                y_vec = double(Tsub.SubjectiveOutcome);
            else
                y_vec = double(Tsub.ObjectiveOutcome);
            end
            stim_vec = double(Tsub.StimIntensity); % you can zscore within subject if needed
        
            % For each time bin compute alpha predictor per trial and fit glm (logistic)
            for tb = 1:N_TIME_BINS
                s = start_sample + (tb-1)*samples_per_time_bin;
                e = s + samples_per_time_bin - 1;
                % extract [time x freq x nTrials_subj] then average time & freq -> vector length nTrials_subj
                % NOTE: alpha_data_cube indexes by global trial order; use subjTrials to select
                sub_cube = alpha_data_cube(s:e, freq_idx, subjTrials);   % small 3D block
                % average across time and freq -> 1 x 1 x nTrials_subj
                alpha_trial_vals = squeeze(mean(mean(sub_cube,1,'omitnan'),2,'omitnan'))'; % column vector nTrials_subj x 1
        
                % check numeric shape
                if isempty(alpha_trial_vals) || numel(alpha_trial_vals) ~= numel(y_vec)
                    betaMat(si,tb) = NaN;
                    continue;
                end
        
                % Optional: center/scale stimulus within subject (helps fit stability)
                stim_z = (stim_vec - mean(stim_vec, 'omitnan')) ./ std(stim_vec, 'omitnan');
                stim_z(isnan(stim_z)) = 0;
        
                % Fit logistic regression: outcome ~ alpha + stimulus
                % Design: [intercept, alpha, stim_z]
                %X = [alpha_trial_vals, stim_z];
                X = [alpha_trial_vals];
                % Remove rows with NaN or constant outcome
                mane = ~isnan(y_vec);
                sadio = ~any(isnan(X),2);
                valid = ~any(isnan(X),2) & ~isnan(y_vec);
               % Xv = X(valid,:);
               % Xv = X(valid);
               % yv = y_vec(valid);
        
                %if numel(unique(yv)) < 2 || size(Xv,1) < 6
                %    betaMat(si,tb) = NaN;
                %    continue;
                %end
        
              %  try
                %    B = glmfit(Xv', yv, 'binomial', 'link', 'logit'); % glmfit with grouped format [y n] but here n=1
                    % B: [intercept; coef_alpha; coef_stim]
                  %  betaMat(si,tb) = B(2); % coefficient for alpha predictor
               % catch
                 %   betaMat(si,tb) = NaN;
               % end
        
                Xv = X(valid);
                Xv = Xv(:);    % ensure column
                yv = y_vec(valid);
                yv = yv(:);    % ensure column
                
                
                
                % Skip if not enough variability
                if numel(unique(yv)) < 2 || numel(yv) < 6
                    betaMat(si,tb) = NaN;
                    continue;
                end
                
                try
                    B = glmfit(Xv, yv, 'binomial', 'link', 'logit'); % simple binomial, no grouped format
                    betaMat(si,tb) = B(2);  % coefficient for alpha
                catch
                    betaMat(si,tb) = NaN;
                end
        
            end
        end
        warning('on','all');
        
        % 4) Quick sanity: remove subjects with all-NaN
        valid_subj = any(~isnan(betaMat),2);
        tempbm = betaMat;
        betaMat = betaMat(valid_subj, :);
        subjects_valid = subjects(valid_subj);
        nSubj_valid = size(betaMat,1);
        if nSubj_valid < 3
            warning('Only %d valid subjects with beta time-series - cluster test may be unreliable.', nSubj_valid);
        end
        
        % 5) Run cluster-based permutation test (your lab function)
        data1 = betaMat;          % subjects x time
        data2 = zeros(size(data1)); % null (compare to zero)
        testType = 1; % within-subject
        [clustersTrue, trueT_P, maxSumPermDistribution] = clusterBasedPermTest(data1, data2, testType, criterion, nPerms);
        
        % clustersTrue: rows = clusters, cols = [startIdx, nSamples, tSum, pVal]
        disp('Clusters (startIdx, length, tSum, pVal):');
        disp(clustersTrue);
        
        % 6) Plot mean beta timecourse with cluster overlays
        meanBeta = nanmean(betaMat,1);
        seBeta = nanstd(betaMat,[],1) ./ sqrt(size(betaMat,1));
        
        %figure('Units','normalized','Position',[0.1 0.2 0.7 0.4]);
        plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); hold on;
        fill([time_bin_centers; flipud(time_bin_centers)], [meanBeta'+1.96*seBeta'; flipud(meanBeta'-1.96*seBeta')], ...
             [0.9 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.6);
        xlabel('Time relative to target (ms)');
        ylabel('Beta (alpha predictor)');
        title(sprintf('%s - %s: Mean subject beta', datasets{r}{2}, outcome));    
        xlim([time_bin_centers(1) time_bin_centers(end)]);
        
        if ~isempty(clustersTrue)
            for c = 1:size(clustersTrue,1)
                startIdx = clustersTrue(c,1);
                nSamp = clustersTrue(c,2);
                pCl = clustersTrue(c,4);
     
                if pCl < 0.1
                    xs = time_bin_centers(startIdx:startIdx+nSamp-1);
                    plot(xs, meanBeta(startIdx:startIdx+nSamp-1), 'r', 'LineWidth', 10); % highlight
                     y_text = max(meanBeta(startIdx:startIdx+nSamp-1)) + 0.02; % adjust vertical position
                    text(mean(xs), y_text, sprintf('p=%.3f', pCl), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',10, 'Color','k');
                end
            end
        end
        plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); % redraw line on top
       
        grid on;
    end
end



%% 0) SETTINGS (edit as needed)
Fs = 1024;                                 % sampling rate
PRE_EVENT_SEC = 0.500;                     % alignment (target at PRE_EVENT_SEC)
time_window_sec = [-0.495, 0.005];               % window to analyze (sec relative to target)
freq_range_hz = [8 12];                    % alpha band to collapse
N_TIME_BINS = 20;                          % number of time bins across window
outcome = 'Subjective';                    % 'Subjective' or 'Objective'
criterion = 0.05;  %because directional                         % cluster forming p-value (inside function)
nPerms = 10000;    % permutations for cluster test (adjust)



figure('Units','normalized','Position',[0.05 0.05 0.9 0.8]); % adjust figure size

for r = 1:length(datasets)        % loop over datasets → rows
    T = datasets{r}{1};
    subjects = cellstr(unique(T.SubjectID));
    for c = 1:length(outcome_types)  % loop over outcomes → columns
        outcome = outcome_types{c};
        
        ax = subplot(length(datasets), length(outcome_types), (r-1)*length(outcome_types) + c);
        hold(ax,'on');

        % 1) Unpack data & basic indices  
        % Ensure SubjectID is categorical or string
       % if ~iscategorical(T.SubjectID)
        %    T.SubjectID = categorical(T.SubjectID);
        %end
        %subjects = categories(T.SubjectID);
        nSubj = numel(subjects);
        nTrialsTotal = height(T);
        
        % Unpack AlphaAmplitude cell to a 3D array [time x freqs x trials]
        alpha_data_cube = cat(3, T.AlphaAmplitude{:});  % time x freqs x trials
        [totalSamples, nFreqs, ~] = size(alpha_data_cube);
        
        % Map freq_range_hz to frequency indices (assumes all_freqs are integers starting freq_range_hz(1):freq_range_hz(2))
        all_freqs = freq_range_hz(1):freq_range_hz(2);
        % If your alpha_data_cube second dim corresponds to frequencies in `all_freqs`, use:
        freq_idx = 1:nFreqs; % if second dim already matches the small range
        % If you have a full freq vector, replace above with proper mapping.
        
        % 2) Time bin indices
        time_zero_sample = round(PRE_EVENT_SEC * Fs);
        start_sample = time_zero_sample + round(time_window_sec(1) * Fs);
        end_sample   = time_zero_sample + round(time_window_sec(2) * Fs);
        total_window_samples = end_sample - start_sample + 1;
        samples_per_time_bin = floor(total_window_samples / N_TIME_BINS);
        
        % Precompute time bin centers in ms
        time_bin_centers = zeros(N_TIME_BINS,1);
        for tb = 1:N_TIME_BINS
            s = start_sample + (tb-1)*samples_per_time_bin;
            e = s + samples_per_time_bin - 1;
            time_bin_centers(tb) = ((s+e)/2 - time_zero_sample) * 1000 / Fs; % ms relative to target
        end
        
        % 3) Build per-subject beta matrix (subjects x timeBins)
        betaMat = nan(nSubj, N_TIME_BINS);
        warning('off','all'); % suppress warnings inside loop; remove if you want them


        for si = 1:nSubj
            subjLabel = subjects{si};             % categorical ID as char
            subj_idx = (T.SubjectID == subjLabel);       % logical indexing
            Tsub = T(subj_idx, :);
            subjTrials = find(subj_idx); % indices into alpha_data_cube third dim

            %fprintf('Subject %d: %d trials\n', subjLabel, sum(subj_idx));

            % require at least some trials
            if numel(subjTrials) < 1
                % too few trials to fit reliable logistic; leave NaNs
                continue;
            end
        
            % construct subject-level predictors/outcome per trial for speed
            % pre-extract outcome and stim for this subject
            if strcmpi(outcome, 'Subjective')
                y_vec = double(Tsub.SubjectiveOutcome);
            else
                y_vec = double(Tsub.ObjectiveOutcome);
            end
            stim_vec = double(Tsub.StimIntensity); % you can zscore within subject if needed
        
            % For each time bin compute alpha predictor per trial and fit glm (logistic)
            for tb = 1:N_TIME_BINS
                s = start_sample + (tb-1)*samples_per_time_bin;
                e = s + samples_per_time_bin - 1;
                % extract [time x freq x nTrials_subj] then average time & freq -> vector length nTrials_subj
                % NOTE: alpha_data_cube indexes by global trial order; use subjTrials to select
                sub_cube = alpha_data_cube(s:e, freq_idx, subjTrials);   % small 3D block
                % average across time and freq -> 1 x 1 x nTrials_subj
                alpha_trial_vals = squeeze(mean(mean(sub_cube,1,'omitnan'),2,'omitnan'))'; % column vector nTrials_subj x 1
        
                % check numeric shape
                if isempty(alpha_trial_vals) || numel(alpha_trial_vals) ~= numel(y_vec)
                    betaMat(si,tb) = NaN;
                    continue;
                end
        
                % Optional: center/scale stimulus within subject (helps fit stability)
                stimZ = (stim_vec - mean(stim_vec, 'omitnan'));
                %./ std(stim_vec, 'omitnan');
                stimZ(isnan(stimZ)) = 0;
        
                % --- Build design matrix ---
                X = [ ...
                    alpha_trial_vals(:), ...             % main effect of alpha
                    stimZ(:), ...                        % main effect of stimulus
                    alpha_trial_vals(:) .* stimZ(:) ...  % interaction term
                ];
                
                valid = ~any(isnan(X),2) & ~isnan(y_vec);
                
                Xv = X(valid,:);
                yv = y_vec(valid);
                
                % Skip if no trials or no variation
                if numel(unique(yv)) < 2 || size(Xv,1) < 6
                    betaAlpha(si,tb) = NaN;
                    betaStim(si,tb)  = NaN;
                    betaInteract(si,tb) = NaN;
                    continue
                end
                
                % --- Fit GLM ---
                try
                    B = glmfit(Xv, yv, 'binomial', 'link', 'logit');  
                    % B = [intercept; beta_alpha; beta_stim; beta_interaction]
                
                    betaAlpha(si,tb)    = B(2);
                    betaStim(si,tb)     = B(3);
                    betaInteract(si,tb) = B(4);
                
                catch
                    betaAlpha(si,tb) = NaN;
                    betaStim(si,tb)  = NaN;
                    betaInteract(si,tb) = NaN;
                end
            end
        end
        warning('on','all');
        
        % 4) Quick sanity: remove subjects with all-NaN
        betaMat = betaInteract;
        valid_subj = any(~isnan(betaMat),2);
        tempbm = betaMat;
        betaMat = betaMat(valid_subj, :);
        subjects_valid = subjects(valid_subj);
        nSubj_valid = size(betaMat,1);
        if nSubj_valid < 3
            warning('Only %d valid subjects with beta time-series - cluster test may be unreliable.', nSubj_valid);
        end
        
        % 5) Run cluster-based permutation test (your lab function)
        data1 = betaMat;          % subjects x time
        data2 = zeros(size(data1)); % null (compare to zero)
        testType = 1; % within-subject
        [clustersTrue, trueT_P, maxSumPermDistribution] = clusterBasedPermTest(data1, data2, testType, criterion, nPerms);
        
        % clustersTrue: rows = clusters, cols = [startIdx, nSamples, tSum, pVal]
        disp('Clusters (startIdx, length, tSum, pVal):');
        disp(clustersTrue);
        
        % 6) Plot mean beta timecourse with cluster overlays
        meanBeta = nanmean(betaMat,1);
        seBeta = nanstd(betaMat,[],1) ./ sqrt(size(betaMat,1));
        
        %figure('Units','normalized','Position',[0.1 0.2 0.7 0.4]);
        plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); hold on;
        fill([time_bin_centers; flipud(time_bin_centers)], [meanBeta'+1.96*seBeta'; flipud(meanBeta'-1.96*seBeta')], ...
             [0.9 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.6);
        xlabel('Time relative to target (ms)');
        ylabel('Interaction Effect');
        ylim([-0.04 0.08])
        title(sprintf('%s - %s: Interaction effect between Stimulus Intensity and Alpha', datasets{r}{2}, outcome));    
        xlim([time_bin_centers(1) time_bin_centers(end)]);
        
        if ~isempty(clustersTrue)
            for c = 1:size(clustersTrue,1)
                startIdx = clustersTrue(c,1);
                nSamp = clustersTrue(c,2);
                pCl = clustersTrue(c,4);
     
                if pCl < 0.05
                    xs = time_bin_centers(startIdx:startIdx+nSamp-1);
                    plot(xs, meanBeta(startIdx:startIdx+nSamp-1), 'r', 'LineWidth', 10); % highlight
                     y_text = max(meanBeta(startIdx:startIdx+nSamp-1)) + 0.02; % adjust vertical position
                    text(mean(xs), y_text, sprintf('p=%.3f', pCl), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize',10, 'Color','k');
                end
            end
        end
        plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); % redraw line on top
       
        grid on;
    end
end



%%
% === Settings ===
stimLevels = 1:10;
Xvar = "ObjectiveOutcome";   % performance, used for PF
DATA.master = MasterTable_Rhythm; % example dataset
dataset_name = 'Rhythm';

% --- Compute alpha terciles ---
alpha = DATA.master.AlphaAmplitudeAvg(:);
edges = quantile(alpha,[0 1/3 2/3 1]);
DATA.master.alphaTerc = discretize(alpha,edges);

subjects = unique(DATA.master.SubjectID);

% Storage
results = struct();
res_idx = 1;

for s = 1:numel(subjects)
    SID = subjects(s);

    % extract data of this subject
    Dsub = DATA.master(DATA.master.SubjectID == SID, :);

    for terc = 1:3
        Dst = Dsub(Dsub.alphaTerc == terc, :);
        if height(Dst) < 30
            continue; % skip too small subsets
        end

        % Build psignifit data structure
        data_ps = [];
        for lv = stimLevels
            trials = Dst(Dst.StimIntensity == lv, :);
            if isempty(trials); continue; end
            nCorrect = sum(trials.(Xvar)==1);
            nTotal   = height(trials);
            data_ps(end+1,:) = [lv, nCorrect, nTotal]; %#ok<AGROW>
        end

        if size(data_ps,1) < 4
            continue; % too few levels
        end

        % === Fit psychometric ===
        options = struct;
        %options.expType = 'yesno';
        options.sigmoidName = 'logistic';
        options.threshPC = 0.5;
        options.estimateType = 'MAP';

        fit = psignifit(data_ps, options);
        argus = fit.Fit;


        % --- Extract thresholds & slopes ---
        threshold = argus(1);
        slope = fit.slope;

        % Store results
        results(res_idx).subject = SID;
        results(res_idx).tercile = terc;
        results(res_idx).threshold = threshold;
        results(res_idx).slope = slope;
        res_idx = res_idx + 1;
    end
end

%%

figure; hold on;

colors = [0 0 1; 1 0 0; 0 0.6 0]; % blue low, red high, green mid

for terc = 1:3
    idx = [results.tercile] == terc;
    thr = [results(idx).threshold];
    slp = [results(idx).slope];

    scatter(thr, slp, 60, colors(terc,:), 'filled');

    % regression line
    p = polyfit(thr, slp, 1);
    xf = linspace(min(thr), max(thr), 200);
    plot(xf, polyval(p, xf), 'Color', colors(terc,:), 'LineWidth', 2);
end

xlabel("Threshold");
ylabel("Slope");
title("Psychometric parameters per subject");
legend("Low α","Mid α","High α");
grid on;


%%
% ---------------------------
% SETTINGS
% ---------------------------
Xvar = "ObjectiveOutcome";   % use "SubjectiveOutcome" for subjective
stim_levels = 1:10;
x_fit = linspace(min(stim_levels), max(stim_levels), 200)';
options = struct;
options.sigmoidName = 'logistic';
%options.expType = 'Yesno';

% choose dataset
DATA = MasterTable_Rhythm;   % example
alpha = DATA.AlphaAmplitudeAvg(:);
edges = quantile(alpha, [0 1/3 2/3 1]);
DATA.alphaTerc = discretize(alpha, edges);

subjects = categories(DATA.SubjectID);   % categorical subjects

% containers for group average
group_pred_low  = zeros(numel(subjects), numel(x_fit));
group_pred_high = zeros(numel(subjects), numel(x_fit));
n_subj_low = 0; n_subj_high = 0;

figure; hold on;
colors = lines(2); % low=blue, high=red

for si = 1:numel(subjects)
    subj = subjects{si};
    subjCat = categorical(cellstr(subj));
    Dsub = DATA(DATA.SubjectID == subjCat, :);
    if isempty(Dsub), continue; end

    % compute subject-level empirical points for low & high tercile
    for terc = [1 3]    % 1 = low, 3 = high
        Dterc = Dsub(Dsub.alphaTerc == terc, :);
        if height(Dterc) < 6
            continue; % skip if too few trials
        end

        % build psignifit data: rows [intensity, nCorrect, nTotal]
        data_ps = [];
        subj_points_x = [];
        subj_points_y = [];
        for lv = stim_levels
            sel = Dterc(Dterc.StimIntensity == lv, :);
            if isempty(sel), continue; end
            nTot = height(sel);
            nCorr = sum(double(sel.(Xvar)) == 1);
            data_ps(end+1,:) = [lv, nCorr, nTot]; %#ok<AGROW>

            % store per-subject empirical point (for scatter)
            subj_points_x(end+1,1) = lv; %#ok<AGROW>
            subj_points_y(end+1,1) = nCorr / nTot; %#ok<AGROW>
        end

        if size(data_ps,1) < 4
            continue; % not enough intensity levels
        end

        % Fit psignifit for this subject-tercile
        result = psignifit(data_ps, options);

                % ensure x_fit is a column
        x_fit = x_fit(:);
        
        % params from psignifit
        params = result.Fit(:);
        
        % initialize
        y_fit_sub = nan(length(x_fit),1);
        
        % --- TRY MODERN CALL STYLE: psiHandle(x, params) ---
        did = false;
        try
            tmp = result.psiHandle(x_fit, params);
            if ismatrix(tmp) && size(tmp,1) == numel(x_fit)
                y_fit_sub = tmp(:,1);  % first column = predicted probability
                did = true;
            end
        catch
            % ignore and try fallback
        end
        
        % --- FALLBACK FOR OLD PSIGNIFIT VERSIONS ---
        if ~did
            try
                tmp = result.psiHandle([x_fit, repmat(params', numel(x_fit), 1)]);
                
                if size(tmp,1) == numel(x_fit)
                    y_fit_sub = tmp(:,1);
                
                elseif size(tmp,2) == numel(x_fit)
                    tmp2 = tmp';
                    y_fit_sub = tmp2(:,1);
                
                else
                    % last resort: take first N entries of flattened vector
                    tmpv = tmp(:);
                    y_fit_sub = tmpv(1:numel(x_fit));
                end
        
            catch ME
                rethrow(ME); % show real error if BOTH calling styles fail
            end
        end
        
        % ensure column vector shape
        y_fit_sub = y_fit_sub(:);

        % Plot subject empirical points and faint subject fit
        if terc == 1
            n_subj_low = n_subj_low + 1;
            group_pred_low(n_subj_low, :) = y_fit_sub(:)';
            scatter(subj_points_x, subj_points_y, 28, 'MarkerFaceColor', colors(1,:), ...
                    'MarkerEdgeColor','k', 'MarkerFaceAlpha', 0.6);
            plot(x_fit, y_fit_sub(:,1), '-', 'Color', [colors(1,:) 0.15], 'LineWidth', 1); % faint
            % optionally fill CI:
            % fill([x_fit; flipud(x_fit)], [y_lo(:,1); flipud(y_hi(:,1))], colors(1,:), 'FaceAlpha', 0.05, 'EdgeColor','none');
        else
            n_subj_high = n_subj_high + 1;
            group_pred_high(n_subj_high, :) = y_fit_sub(:)';
            %scatter(subj_points_x, subj_points_y, 28, 'MarkerFaceColor', colors(2,:), ...
             %       'MarkerEdgeColor','k', 'MarkerFaceAlpha', 0.6);
            plot(x_fit, y_fit_sub(:,1), '-', 'Color', [colors(2,:) 0.15], 'LineWidth', 1); % faint
        end
    end
end

% Trim unused rows and compute group mean predicted curve
group_pred_low = group_pred_low(1:n_subj_low, :);
group_pred_high = group_pred_high(1:n_subj_high, :);

mean_pred_low = mean(group_pred_low, 1, 'omitnan');
mean_pred_high = mean(group_pred_high, 1, 'omitnan');

% Plot bold group-mean fits
plot(x_fit, mean_pred_low, '-', 'Color', colors(1,:), 'LineWidth', 3, 'DisplayName','Low α (group)');
plot(x_fit, mean_pred_high, '-', 'Color', colors(2,:), 'LineWidth', 3, 'DisplayName','High α (group)');

xlabel('Stimulus intensity'); ylabel('P(correct / seen)');
legend('Location','best');
grid on;

%%


% === Define number of bins ===
N_BINS = 10;

% your existing info
time_zero_sample = round(PRE_EVENT_SEC * Fs);
T = size(MasterTable.AlphaAmplitude{1},1);   % #time samples in TF matrix
time_vector = ((1:T) - time_zero_sample) / Fs;   % create the time axis in seconds

% compute bin edges on your chosen time range:
TIME_RANGE = [-0.495 0];   % adjust if needed
edges = linspace(TIME_RANGE(1), TIME_RANGE(2), N_BINS+1);

% Pre-allocate
chi2_vals = zeros(N_BINS,1);
p_vals = zeros(N_BINS,1);
deltaAIC = zeros(N_BINS,1);


for bi = 1:N_BINS
    
    % current time bin boundaries:
    t_start = edges(bi);
    t_end   = edges(bi+1);

    % convert boundaries to sample indices
    pred_start_sample = time_zero_sample + round(t_start * Fs);
    pred_end_sample   = time_zero_sample + round(t_end   * Fs);

    % RESTRICT TO ALPHA FREQUENCIES (same as your code)
    all_freqs = alpha_freq_range(1):alpha_freq_range(2);
    freq_start_sample = find(all_freqs == freq_pred_bin(1));
    freq_end_sample   = find(all_freqs == freq_pred_bin(2));

    % --- COMPUTE ALPHA AVERAGE FOR THIS BIN ---
    AlphaBin = zeros(height(MasterTable),1);

    for i = 1:height(MasterTable)
        TF = MasterTable.AlphaAmplitude{i};

        % safety clipping
        s1 = max(1, pred_start_sample);
        s2 = min(size(TF,1), pred_end_sample);

        f1 = freq_start_sample;
        f2 = freq_end_sample;

        AlphaBin(i) = mean(mean(TF(s1:s2, f1:f2),1),2);
    end

    % attach temporary column for this bin
    MasterTable.AlphaTemp = AlphaBin;

    % ===========================
    %        FIT MODELS
    % ===========================

    % --- MODEL 0: NO INTERACTION ---
    M0 = fitglme(MasterTable, ...
        'ObjectiveOutcome ~ StimIntensity + AlphaTemp + (1|SubjectID)', ...
        'Distribution','Binomial','Link','logit', 'FitMethod', 'Laplace');

    % --- MODEL 1: LINEAR INTERACTION ---
    M1 = fitglme(MasterTable, ...
        'ObjectiveOutcome ~ StimIntensity * AlphaTemp + (1|SubjectID)', ...
        'Distribution','Binomial','Link','logit', 'FitMethod', 'Laplace');

    % ===========================
    %         LRT TEST
    % ===========================
    stats = compare(M0, M1);

    chi2_vals(bi) = stats.LRStat(2);       % log-likelihood χ²
    p_vals(bi)    = stats.pValue(2);       % p-value
    deltaAIC(bi)  = stats.AIC(2) - stats.AIC(1);   % AIC drop (neg = improvement)
end

% cleanup
MasterTable.AlphaTemp = [];


figure;
yyaxis left
bar(chi2_vals,'FaceColor',[0.2 0.4 0.8], 'EdgeColor','none');
ylabel('\chi^2 improvement (M1 vs M0)');
xlabel('Time bins');

yyaxis right
plot(1:N_BINS, p_vals,'ko','MarkerFaceColor','k');
ylabel('p-value');

title('Stim × Alpha interaction strength over time');
grid on;

% Optional significance line
hold on;
yline(0.05,'r--','p=0.05');

