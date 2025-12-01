function DATA = baseline_and_subject_zscore(DATA, roi_channels, DO_ZSCORE)
% BASELINE_AND_SUBJECT_ZSCORE
% Z-scores AlphaAmplitude relative to trial baseline, then within-subject
%
% INPUTS:
%   DATA         : table with columns:
%                  - AlphaAmplitude (cell, each cell [time x freq] or [time x freq x channels])
%                  - Baseline       (cell, same size as AlphaAmplitude)
%                  - SubjectID      (categorical or string)
%   roi_channels : numeric vector of channel indices to average across if 3D
%
% OUTPUT:
%   DATA         : same table, AlphaAmplitude updated

if nargin < 2
    roi_channels = []; % if empty, no channel averaging
end

n_trials = height(DATA);
alpha_bc = cell(n_trials,1); % temporary storage

% Step 1: Baseline z-score per trial
for t = 1:n_trials
    alpha_trial = DATA.AlphaAmplitude{t};
    baseline_trial = DATA.Baseline{t};
    
    % If 3D: [time x freq x channels]
    if ndims(alpha_trial) == 3
        if ~isempty(roi_channels)
            alpha_trial = mean(alpha_trial(:,:,roi_channels), 3);
            baseline_trial = mean(baseline_trial(:,:,roi_channels), 3);
        end
    end
    
    % Compute baseline mean and std
    mu_base = mean(baseline_trial(:));
    sigma_base = std(baseline_trial(:));
    if sigma_base == 0, sigma_base = 1; end
    
    % Baseline z-score
    alpha_bc{t} = (alpha_trial - mu_base) / sigma_base;
end

if DO_ZSCORE

% Step 2: Subject-level z-score
    subjects = cellstr(unique(DATA.SubjectID));
    
    for s = 1:length(subjects)
        subj = subjects{s};
        idx_subj = DATA.SubjectID == subj;
        
        % Mean per trial across time x freq
        trial_means = cellfun(@(x) mean(x(:)), alpha_bc(idx_subj));
        
        mu_subj = mean(trial_means);
        sigma_subj = std(trial_means);
        if sigma_subj == 0, sigma_subj = 1; end
        sigma_subj = 1; % because maybe dividing by baseline sigma isn't meaningful
        
        % Apply z-score to each trial
        for t_idx = find(idx_subj)'
            alpha_bc{t_idx} = (alpha_bc{t_idx} - mu_subj) / sigma_subj;
        end
    end
end

% Step 3: Update table
DATA.AlphaAmplitude = alpha_bc;

end



