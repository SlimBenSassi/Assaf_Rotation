function [latencies, codes, intensities, outcomes, n_trials] = select_single_trials(SDATA, target_codes, report_unseen_code, report_seen_codes)
% SELECT_SINGLE_TRIALS Filters continuous data triggers to link targets to subjective reports.
%
% INPUTS:
%   SDATA: Main data structure containing events and Fs.
%   target_codes: Vector of target codes (e.g., [112, 122]).
%   report_unseen_code: Code for 'Miss' (e.g., 231).
%   report_seen_codes: Vector of codes for 'Seen' (e.g., [232, 233, 234]).
%
% OUTPUTS:
%   latencies: Sample indices of the selected target events (N x 1).
%   codes: Event codes corresponding to the latencies (N x 1).
%   outcomes: Binary subjective outcome (0=Unseen, 1=Seen) (N x 1).
%   n_trials: Total number of successfully linked trials.

% --- 1. Internal Setup ---
Fs = SDATA.info.sampling_rate;
status_vector = SDATA.events.triggerChannel;
search_window_sec = 10; % in seconds, max time to look for a subjective report
search_window_samples = round(search_window_sec * Fs); % Search based on time (samples)

% --- 2. Extract Raw Triggers ---
trigger_indices = find(status_vector ~= 0); 
trigger_codes = status_vector(trigger_indices);

% --- 3. Link Targets to Subjective Outcomes (CRITICAL STEP) ---
final_target_latencies = [];
final_target_codes = [];
y_subjective_outcome = [];

for i = 1:length(trigger_codes)
    current_code = trigger_codes(i);
    
    if ismember(current_code, target_codes)
        target_latency = trigger_indices(i);
        
        % Search for the next subjective report event (Max 4 events look-ahead because subj responses come 4 events after target, for safere use 10)
        for j = i + 1 : min(i + 4, length(trigger_codes)) 
            report_code = trigger_codes(j);
            report_latency = trigger_indices(j);

            % Check 1: Is the event a subjective report?
            if ismember(report_code, [report_unseen_code, report_seen_codes])
                
                % Check 2: Did the report occur within the valid time window?
                if report_latency - target_latency <= search_window_samples
                    
                    final_target_latencies = [final_target_latencies; target_latency];
                    final_target_codes = [final_target_codes; current_code];
                    
                    % Determine binary outcome: 1 if Seen, 0 if Unseen
                    if ismember(report_code, report_unseen_code)
                        y_subjective_outcome = [y_subjective_outcome; 0];
                    else
                        y_subjective_outcome = [y_subjective_outcome; 1];
                    end
                    break; % Found the response, move to the next target
                end
            end
        end
    end
end

% --- 4. Finalize Outputs ---
latencies = final_target_latencies;
codes = final_target_codes;
intensities = mod(codes, 10) + 1;
outcomes = y_subjective_outcome;
n_trials = length(latencies);

% --- Display Summary ---
disp('--- Target-Response Linking Summary ---');
disp(['Total successfully linked trials: ' num2str(n_trials)]);

for code = target_codes
    count = sum(final_target_codes == code);
    %disp(final_target_codes);
    % We use an if check to display 0s cleanly (the TODO you mentioned)
    if count > 0 
        disp(['  > Code ' num2str(code) ' found: ' num2str(count) ' trials.']); 
    end
end
end