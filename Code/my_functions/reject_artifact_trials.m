function [filtered_latencies, filtered_codes, filtered_intensities, filtered_obj_outcomes, filtered_subj_outcomes, filtered_conditions, filtered_isis, n_trials_final] = reject_artifact_trials(SDATA, epoch_latencies, epoch_codes, intensities, y_objective_outcome, y_subjective_outcome, conditions, isis, total_epoch_samples, pre_samples)
% REJECT_ARTIFACT_TRIALS Filters trials that overlap with the continuous artifact mask.
%
% INPUTS:
%   SDATA: Main data structure containing the continuous artifact mask.
%   epoch_latencies: Sample indices of the target events (N x 1).
%   epoch_codes: Event codes corresponding to the latencies (N x 1).
%   y_objective_outcome: Binary objective outcome (0/1) (N x 1).
%   y_subjective_outcome: Binary subjective outcome (0/1) (N x 1).
%   condition
%   isi
%   total_epoch_samples: Total length of the epoch (in samples).
%   pre_samples: Number of samples in the pre-stimulus baseline.
%
% OUTPUTS:
%   filtered_latencies, filtered_codes, filtered_outcomes: The sliced lists (only good trials).
%   n_trials_final: The final count of good trials.

disp('Applying trial-level artifact rejection...');

% --- 1. Prepare Artifact Mask ---
% Get the continuous artifact mask (N_time_points x 1 vector)
artifact_mask_cont = SDATA.metadata.artifacts; 

% 2. Initialize a logical vector to mark good trials
n_trials = length(epoch_latencies);
is_good_trial = true(n_trials, 1); 

% 3. The Rejection Loop: Check every trial segment
for trial_idx = 1:n_trials
    
    center_idx = epoch_latencies(trial_idx); 
    start_idx = center_idx - pre_samples;
    end_idx = start_idx + total_epoch_samples - 1;
    
    % Check the continuous artifact mask in the time window of the current trial
    % IMPORTANT: Add safety check for array bounds, as the mask might be shorter than continuous data.
    if start_idx >= 1 && end_idx <= length(artifact_mask_cont)
        trial_artifact_segment = artifact_mask_cont(start_idx:end_idx);
        
        % If ANY sample in this segment is marked '1' (artifact), the whole trial is BAD.
        if any(trial_artifact_segment) 
            is_good_trial(trial_idx) = false;
            % disp(center_idx); % Displaying center_idx for debugging purposes only
        end
    else
        % If the epoch slice goes out of bounds of the artifact mask, reject the trial
        is_good_trial(trial_idx) = false;
    end
end

% --- 4. Apply Rejection to Data and Labels (The Slicing) ---
n_rejected = sum(~is_good_trial);
disp(['Rejected ' num2str(n_rejected) ' trials due to temporal artifacts.']);

% Filter all trial-dependent variables to keep only good trials:
filtered_latencies = epoch_latencies(is_good_trial);
filtered_codes = epoch_codes(is_good_trial);
filtered_intensities = intensities(is_good_trial);
filtered_obj_outcomes = y_objective_outcome(is_good_trial);
filtered_subj_outcomes = y_subjective_outcome(is_good_trial);
filtered_conditions = conditions(is_good_trial);
filtered_isis = isis(is_good_trial);
n_trials_final = sum(is_good_trial); % Update the total trial count

disp(['Final N after temporal rejection: ' num2str(n_trials_final)]);
end