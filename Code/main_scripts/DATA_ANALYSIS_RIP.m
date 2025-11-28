%% DATA ANALYSIS

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
time_window_sec = [-0.495 0];




% --- Channels --- %
single_channel_idx = 48; % Cz=48, Oz=29
ROI.Central = {[11, 12, 13, 46, 47, 48 ,49], "Central Cluster"};
ROI.Occipital = {[25, 26, 27, 28, 29, 30, 62, 63, 64], "Occipital Cluster"};
ROI.All = {[1:n_channels], "All Channels"};
ROI.Single = {[single_channel_idx], num2str(single_channel_idx)}; %put electrode of interest idx in here
current_ROI_cell = ROI.Occipital;
currentROI = current_ROI_cell{1};
currentROI_name = current_ROI_cell{2};

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

MasterTable= baseline_and_subject_zscore(MasterTable, currentROI);
RAW=false;
disp('All single-trial baseline-corrected and z-scored per participant.');



%% CREATE ONE AVERAGE ALPHA AMPLITUDE

% Here we will take one alpha predictor and not each time-frequency
% combination, choice is informed visually by the heatmap results.

time_pred_bin = [-0.300; -0.200];
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



%% CONDITIONS AND OUTCOMES


% Generate a 3x2 figure for different outcome types and datasets
outcome_types = {'Objective', 'Subjective'};
datasets = {{MasterTable_Rhythm, 'Rhythm'}, {MasterTable_Interval, 'Interval'}, {MasterTable_Irregular, 'Irregular'}};


%% TIME COURSES

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
        ylabel('Alpha Amplitude (10 Hz)');
        legend('show');
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
        %sig_neg_mask = (PValueMap' < 0.1) & (BetaMap' < 0);
        sig_neg_mask = (PValueMap' < 0.05);
        % 2. Plot the contour lines using the mask
        [~, h_contour] = contour(TimeBins, FreqBins, sig_neg_mask, [0.5 0.5], 'LineWidth', 3, 'LineColor', 'w', 'LineStyle', ':');
        colorbar;

    end
end

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
