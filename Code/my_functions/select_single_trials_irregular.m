function [latencies, codes, intensities, outcomes, n_trials] = select_single_trials_irregular(SDATA, warning_codes_of_interest, all_warning_codes, subj_id)
% SELECT_SINGLE_TRIALS_IRREGULAR Specialized function to link targets to subjective reports 
% when target codes are faulty. Relies on external behavioral files for Stimulus Intensity and Outcome.
%
% INPUTS: SDATA, target_codes, warning_code (073), external_data_path (to find the .mat files), report codes.
% NOTE: report_unseen_code and report_seen_codes are now redundant inputs, but kept for function compatibility.

% --- 1. Internal Setup & Trigger Extraction ---
Fs = SDATA.info.sampling_rate;
status_vector = SDATA.events.triggerChannel;
search_window_sec = 10.0; 
search_window_samples = round(search_window_sec * Fs); 
trigger_indices = find(status_vector ~= 0); 
trigger_codes = status_vector(trigger_indices);

% --- 2. Load External Behavioral Data ---
% --- The full path to the external behavioral data file ---
full_external_path = fullfile('\\wks3\pr_breska\el-Christina\SxA\SxA_Data\Behaviour Preprocessed', ['SxA_ResultsSubject' subj_id '_Session2.mat']);
try
    external_data = load(full_external_path);
    beh_table = external_data.subresults.data; 
    disp('External behavioral data table loaded successfully.');
catch ME
    disp(['FATAL ERROR: Could not load external behavioral file. Error: ' ME.message]);
    return;
end

% --- 3. Link Targets, Extract Intensity, and Match ---
final_target_latencies = [];
final_target_codes = [];
y_subjective_outcome = [];
stim_intensity_vector = [];

eeg_trial_counter = 0;

for i = 1:length(trigger_codes)
    current_code = trigger_codes(i);
    if ismember(current_code, all_warning_codes)
        eeg_trial_counter = eeg_trial_counter + 1; % Increment EEG counter
    end

    % --- CHECK 1: Found Warning Signal of Interest ---
    if ismember(current_code, warning_codes_of_interest)

        
        % Check if the next event is a target (we assume it is)
        target_latency = trigger_indices(i + 1); 
        
        % --- Cross-Reference with Behavioral Table ---
        % Find the row in the behavioral table that matches the current EEG trial number.
        beh_row_idx = eeg_trial_counter;

        if ~isempty(beh_row_idx)

            % --- Extract Stimulus Intensity & Outcome from Behavioral File ---
            raw_contrast_level = beh_table.("Contrast Level")(beh_row_idx);
            binary_visibility = beh_table.("Binary Visibility")(beh_row_idx); 
            
            % CRITICAL FILTER: Only proceed if Contrast Level is NOT 0 (Catch Trial)
            if raw_contrast_level > 0
                
                final_target_latencies = [final_target_latencies; target_latency];
                final_target_codes = [final_target_codes; 100+raw_contrast_level-1]; % Ignoring time before target, coding only for stim intensity
                stim_intensity_vector = [stim_intensity_vector; raw_contrast_level]; 
                
                % Store the binary outcome (0 or 1) directly from the table
                y_subjective_outcome = [y_subjective_outcome; binary_visibility]; 
            end 
        end
    end 
end


% --- 4. Finalize Outputs ---
latencies = final_target_latencies;
codes = final_target_codes;
outcomes = y_subjective_outcome;
intensities = stim_intensity_vector; 
n_trials = length(latencies);

% --- Display Summary ---
disp('--- Target-Response Linking Summary (IRREGULAR) ---');
disp(['Total successfully linked and non-catch trials: ' num2str(n_trials)]);
disp(['Total trials marked SEEN (1): ' num2str(sum(outcomes == 1))]);
disp(['Total trials marked UNSEEN (0): ' num2str(sum(outcomes == 0))]);

end