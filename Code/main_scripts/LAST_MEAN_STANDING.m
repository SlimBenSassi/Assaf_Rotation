% LAST MEAN STANDING

%% LOADING DATASET
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


%% BASELINE PER TRIAL

if DO_BASELINE_CORRECTION
    MasterTable= baseline_and_subject_zscore(MasterTable, currentROI, false);
end

RAW=false;

disp('All single-trial baseline-corrected and z-scored per participant.');



%% CREATE ONE AVERAGE ALPHA AMPLITUDE

% Here we will take one alpha predictor and not each time-frequency
% combination, choice is informed visually by the heatmap results.

time_pred_bin = [-0.100; 0];
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
%datasets = {{MasterTable_Predictive, 'Predictive'}, {MasterTable_Irregular, 'Irregular'}};

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
    model_formula_subjective = ['SubjectiveOutcome ~ AlphaAmplitudeAvg + StimIntensity + (1|SubjectID)'];
    glme_subjective = fitglme(data_condition, model_formula_subjective, ...
                               'Distribution', 'Binomial', ...
                               'Link', 'logit', 'FitMethod', 'Laplace');
    subjective_betas(col) = glme_subjective.Coefficients.Estimate(3);
    all_CIs = coefCI(glme_subjective, 'Alpha', 0.1); 
    subjective_CI(:, col) = all_CIs(3,:);
    
    % Fit GLME for Objective
    model_formula_objective = ['ObjectiveOutcome ~ AlphaAmplitudeAvg + StimIntensity + (1|SubjectID)'];

    glme_objective = fitglme(data_condition, model_formula_objective, ...
                              'Distribution', 'Binomial', ...
                              'Link', 'logit', 'FitMethod', 'Laplace');
    objective_betas(col) = glme_objective.Coefficients.Estimate(3);
    all_CIs = coefCI(glme_objective, 'Alpha', 0.1); 
    objective_CI(:, col) = all_CIs(3,:);
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


%% Time-resolved + Cluster-based permutation

% Settings

time_window_sec = [-0.495, 0];               % window to analyze (sec relative to target)
freq_range_hz = [8 12];                    % alpha band to collapse
N_TIME_BINS = 5;                          % number of time bins across window
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

        % Compute time-bin edges in ms
        time_bin_edges = zeros(N_TIME_BINS+1,1);
        for tb = 1:N_TIME_BINS
            s = start_sample + (tb-1)*samples_per_time_bin;
            e = s + samples_per_time_bin - 1;
        
            time_bin_edges(tb) = ((s) - time_zero_sample) * 1000 / Fs;  % bin start
        end
        time_bin_edges(end) = ((end_sample) - time_zero_sample) * 1000 / Fs; % final end

        
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
                %stimZ = (stim_vec - mean(stim_vec, 'omitnan'));
                %./ std(stim_vec, 'omitnan');
                %stimZ(isnan(stimZ)) = 0;
                stimZ = stim_vec;
        
                % --- Build design matrix ---
                X = [ ...
                    alpha_trial_vals(:), ...             % main effect of alpha
                    stimZ(:), ...                        % main effect of stimulus
                    %alpha_trial_vals(:) .* stimZ(:) ...  % interaction term
                ];
                
                valid = ~any(isnan(X),2) & ~isnan(y_vec);
                
                Xv = X(valid,:);
                yv = y_vec(valid);
                
                % Skip if no trials or no variation
                if numel(unique(yv)) < 2 || size(Xv,1) < 6
                    betaAlpha(si,tb) = NaN;
                    betaStim(si,tb)  = NaN;
                    %betaInteract(si,tb) = NaN;
                    continue
                end
                
                % --- Fit GLM ---
                try
                    B = glmfit(Xv, yv, 'binomial', 'link', 'logit');  
                    % B = [intercept; beta_alpha; beta_stim; beta_interaction]
                
                    betaAlpha(si,tb)    = B(2);
                    betaStim(si,tb)     = B(3);
                    %betaInteract(si,tb) = B(4);
                
                catch
                    betaAlpha(si,tb) = NaN;
                    betaStim(si,tb)  = NaN;
                    %betaInteract(si,tb) = NaN;
                end
            end
        end
        warning('on','all');
        
        % 4) Quick sanity: remove subjects with all-NaN
        betaMat = betaAlpha;
        disp(size(betaMat));
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

        % Define full x-range from first to last actual samples
        x_full = linspace( ...
            (start_sample - time_zero_sample)*1000/Fs, ...
            (end_sample   - time_zero_sample)*1000/Fs, ...
            500);  % high resolution
        
        % Interpolate betas and SE across the whole window
        meanBeta_full = interp1(time_bin_centers, meanBeta, x_full, 'pchip');
        seBeta_full   = interp1(time_bin_centers, seBeta,   x_full, 'pchip');
        hold on;
        
        % Shaded SE region
        fill([x_full fliplr(x_full)], ...
             [meanBeta_full + 1.96*seBeta_full, fliplr(meanBeta_full - 1.96*seBeta_full)], ...
             [0.8 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.5);
        
        % Smooth line
        plot(x_full, meanBeta_full, 'k','LineWidth',1.8);

        
        %figure('Units','normalized','Position',[0.1 0.2 0.7 0.4]);
        %plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); hold on;
        %fill([time_bin_centers; flipud(time_bin_centers)], [meanBeta'+1.96*seBeta'; flipud(meanBeta'-1.96*seBeta')], ...
            % [0.9 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.6);
        xlabel('Time relative to target (ms)');
        ylabel('Alpha Effect');
        ylim([-0.2 0.2])
        title(sprintf('%s - %s: Interaction effect between Stimulus Intensity and Alpha', datasets{r}{2}, outcome));    
        %xlim([time_bin_centers(1) time_bin_centers(end)]);
        
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

        x_plot = zeros(2*N_TIME_BINS,1);
        y_plot = zeros(2*N_TIME_BINS,1);
        
        k = 1;
        for tb = 1:N_TIME_BINS
            x_plot(k)   = time_bin_edges(tb);
            y_plot(k)   = meanBeta(tb);
            x_plot(k+1) = time_bin_edges(tb+1);
            y_plot(k+1) = meanBeta(tb);
            k = k + 2;
        end
        %plot(x_plot, y_plot, 'k', 'LineWidth', 1.8); hold on;

        %plot(time_bin_centers, meanBeta, '-k','LineWidth',1.8); % redraw line on top
       
        grid on;
    end
end






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


%% PARAMETERS (Assuming these are defined in your original script)
% freq_start_sample and freq_end_sample are indices for your alpha band (e.g., 8-12 Hz)
% time_axis_ms is the time vector in milliseconds
% datasets is a cell array of cells, where datasets{col}{1} is the main data table,
% and datasets{col} has one entry per participant (e.g., datasets{col}{1}.AlphaAmplitude is the table for participant 1).
% We assume the structure is: datasets{dataset_index}{participant_index}.TableVariable

%% TIME COURSES - DIFFERENCE PLOT
figure;

for row = 1:length(outcome_types)      % subjective/objective
    for col = 1:length(datasets)       % datasets/conditions
        
        % Pull this condition's table
        T_cond = datasets{col}{1};     % FIX: datasets{col}{1} is the *table*, not a participant
        all_subj = unique(T_cond.SubjectID);
        num_participants = length(all_subj);

        % Storage
        participant_diff_time_courses = cell(num_participants,1);

        % LOOP participants
        for p = 1:num_participants
            
            subjLabel = all_subj(p);
            P_data = T_cond(T_cond.SubjectID == subjLabel, :);

            % Pick outcome
            if strcmp(outcome_types{row}, 'Objective')
                outcome_col = P_data.ObjectiveOutcome;
            else
                outcome_col = P_data.SubjectiveOutcome;
            end

            % Time-courses for outcome == 1
            idx1 = (outcome_col == 1);
            TC1 = P_data.AlphaAmplitude(idx1);
            M1 = calculate_avg_time_course(TC1, freq_start_sample, freq_end_sample);

            % Time-courses for outcome == 0
            idx0 = (outcome_col == 0);
            TC0 = P_data.AlphaAmplitude(idx0);
            M0 = calculate_avg_time_course(TC0, freq_start_sample, freq_end_sample);

            % Participant-level difference curve
            participant_diff_time_courses{p} = M1 - M0;
        end

        % --- AGGREGATE ---
        Diff_Matrix = cell2mat(participant_diff_time_courses);
        avg_diff_tc = mean(Diff_Matrix,1);
        sem_diff_tc = std(Diff_Matrix,0,1) / sqrt(num_participants);

        upper = avg_diff_tc + sem_diff_tc;
        lower = avg_diff_tc - sem_diff_tc;

        % Compute time axis (SAME for all participants)
        time_axis_ms = ((1:length(avg_diff_tc)) - time_zero_sample) / Fs * 1000;

        % --- PLOT ---
        ax = subplot(length(outcome_types), length(datasets), ...
                     (row-1)*length(datasets) + col);
        hold on;

        % Fill
        fill([time_axis_ms fliplr(time_axis_ms)], ...
             [lower fliplr(upper)], ...
             [0.4 0.4 0.4], 'FaceAlpha',0.15,'EdgeColor','none');

        % Main line
        plot(time_axis_ms, avg_diff_tc, 'k','LineWidth',1.8);

        % Reference lines
        xline(0,'--r');
        yline(0,'--r');
        %yline(0,':','Color','--r',[0.5 0.5 0.5]);
        ylim([-0.6 0.3]);

        title(sprintf('Diff: %s – %s', outcome_types{row}, datasets{col}{2}));
        xlabel('Time (ms)');
        ylabel('Alpha difference (Outcome 1–0)');
        grid on;
        hold off;
    end
end


%% INTERACTION ANALYSIS

% Setup
intensity_ranges = {
    [1 2],  '1–3';
    [3 6],  '4–6';
    [7 10], '7–10'
};
nRanges = size(intensity_ranges,1);
nDatasets = length(datasets);
nCols = nDatasets * 2;   % columns per group (subj,obj) x datasets

% Preallocate storage
subjective_betas = nan(nRanges, nDatasets);
objective_betas = nan(nRanges, nDatasets);
subjective_CI_low  = nan(nRanges, nDatasets);
subjective_CI_high = nan(nRanges, nDatasets);
objective_CI_low   = nan(nRanges, nDatasets);
objective_CI_high  = nan(nRanges, nDatasets);

% Get dataset labels
labels = cell(1,nDatasets);
for d = 1:nDatasets
    labels{d} = datasets{d}{2};
end

% Fit GLMEs per intensity range and dataset
for r = 1:nRanges
    Imin = intensity_ranges{r,1}(1);
    Imax = intensity_ranges{r,1}(2);

    for d = 1:nDatasets
        data_condition = datasets{d}{1};

        % filter trials by intensity
        idx = (data_condition.StimIntensity >= Imin & data_condition.StimIntensity <= Imax);
        data_range = data_condition(idx, :);

        % Skip if no trials in range
        if isempty(data_range)
            warning('No trials for dataset %s range %s', datasets{d}{2}, intensity_ranges{r,2});
            continue;
        end

        % Subjective model
        fm_subj = 'SubjectiveOutcome ~ AlphaAmplitudeAvg + StimIntensity + (1|SubjectID)';
        glme_subj = fitglme(data_range, fm_subj, 'Distribution','Binomial','Link','logit','FitMethod','Laplace');

        % get coefficient index for AlphaAmplitudeAvg (name might vary, find it)
        cn = glme_subj.CoefficientNames;
        idx_alpha = find(strcmp(cn,'AlphaAmplitudeAvg'));
        if isempty(idx_alpha), error('AlphaAmplitudeAvg not found in coeff names for subjective model'); end

        subjective_betas(r,d) = glme_subj.Coefficients.Estimate(idx_alpha);
        CI = coefCI(glme_subj);            % default 95% CI
        subjective_CI_low(r,d)  = CI(idx_alpha,1);
        subjective_CI_high(r,d) = CI(idx_alpha,2);

        % Objective model
        fm_obj = 'ObjectiveOutcome ~ AlphaAmplitudeAvg + StimIntensity + (1|SubjectID)';
        glme_obj = fitglme(data_range, fm_obj, 'Distribution','Binomial','Link','logit','FitMethod','Laplace');

        cn2 = glme_obj.CoefficientNames;
        idx_alpha2 = find(strcmp(cn2,'AlphaAmplitudeAvg'));
        if isempty(idx_alpha2), error('AlphaAmplitudeAvg not found in coeff names for objective model'); end

        objective_betas(r,d) = glme_obj.Coefficients.Estimate(idx_alpha2);
        CI2 = coefCI(glme_obj);
        objective_CI_low(r,d)  = CI2(idx_alpha2,1);
        objective_CI_high(r,d) = CI2(idx_alpha2,2);
    end
end

% Build bar_data matrix with columns ordered as:
% [ dataset1_subj, dataset1_obj, dataset2_subj, dataset2_obj, ... ]
bar_data = zeros(nRanges, nCols);
ci_low  = zeros(nRanges, nCols);
ci_high = zeros(nRanges, nCols);

for d = 1:nDatasets
    col_subj = 2*(d-1) + 1;
    col_obj  = col_subj + 1;
    bar_data(:, col_subj) = subjective_betas(:, d);
    bar_data(:, col_obj)  = objective_betas(:, d);

    ci_low(:, col_subj)   = subjective_betas(:,d) - subjective_CI_low(:,d);
    ci_high(:, col_subj)  = subjective_CI_high(:,d) - subjective_betas(:,d);

    ci_low(:, col_obj)    = objective_betas(:,d) - objective_CI_low(:,d);
    ci_high(:, col_obj)   = objective_CI_high(:,d) - objective_betas(:,d);
end

% Plot grouped bars: rows = ranges (groups), cols = nCols bars per group
figure('Units','normalized','Position',[0.1 0.1 0.75 0.45]);
bh = bar(bar_data, 'grouped');  % bh is array of bar objects, length = nCols
hold on;

% --- COLOR SCHEME FOR DATASETS ---
% One color per dataset
dataset_colors = [
    0.2 0.4 0.8;  % dataset 1 hue (blueish)
    0.85 0.3 0.3; % dataset 2 hue (reddish)
    0.3 0.7 0.3;  % dataset 3 hue (greenish)
    % extend if more datasets
];

% Assign colors to bars:
% odd columns = subjective, even columns = objective
for d = 1:nDatasets
    base_color = dataset_colors(d, :);

    % darker version for SUBJECTIVE
    subj_color = base_color * 0.7;

    % lighter version for OBJECTIVE
    obj_color  = base_color * 1.3;
    obj_color(obj_color > 1) = 1;  % clip to [0,1]

    col_subj = 2*(d-1) + 1;
    col_obj  = col_subj + 1;

    bh(col_subj).FaceColor = subj_color;
    bh(col_obj).FaceColor  = obj_color;
end


% Get x positions for each column across groups
xpos = nan(nRanges, nCols);
for c = 1:numel(bh)
    xpos(:,c) = bh(c).XEndPoints;  % returns vector length nRanges
end

% Add error bars for each column and each group
for c = 1:nCols
    err_low = ci_low(:,c);
    err_high = ci_high(:,c);

    % plot errorbars for each group's bar for this column
    errorbar(xpos(:,c), bar_data(:,c), err_low, err_high, 'k', 'LineStyle','none', 'LineWidth', 1.2);
end

% Ticks: put xticks at group centers and label with intensity range labels
group_centers = mean(xpos,2);
set(gca, 'XTick', group_centers, 'XTickLabel', intensity_ranges(:,2));
xlabel('Stimulus intensity range');
ylabel('AlphaAmplitudeAvg coefficient (GLME)');
title('Alpha coefficient by StimRange × Condition × Outcome');

% Define colors for each bar in the plot (dataset × outcome type)
colors = [
    dataset_colors(1,:) * 0.7;  % Dataset1 Subjective (dark)
    min(dataset_colors(1,:) * 1.3, 1); % Dataset1 Objective (light)
    dataset_colors(2,:) * 0.7;  % Dataset2 Subjective
    min(dataset_colors(2,:) * 1.3, 1); % Dataset2 Objective
    dataset_colors(3,:) * 0.7;  % Dataset3 Subjective
    min(dataset_colors(3,:) * 1.3, 1); % Dataset3 Objective
];

% Create dummy handles for legend
dummy = gobjects(6,1);
labels_legend = {...
    'Rhythm - Objective', 'Rhythm - Subjective', ...
    'Interval - Objective','Interval - Subjective', ...
    'Irregular - Objective','Irregular - Subjective'};

hold on;
for i = 1:6
    dummy(i) = plot(nan, nan, 's', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor','k');
end

legend(dummy, labels_legend, 'Location','northeastoutside');


grid on; box on;
hold off;
