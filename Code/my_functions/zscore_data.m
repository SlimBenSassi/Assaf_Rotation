function data = zscore_data(data, Fs, PRE_EVENT_SEC, baseline_start_sec, baseline_end_sec)
%z-scoring data based on baseline period's mean and std


time_zero_sample = round(PRE_EVENT_SEC * Fs); 
baseline_start_sample = time_zero_sample + round(baseline_start_sec * Fs); 
baseline_end_sample = time_zero_sample + round(baseline_end_sec * Fs);


% --- B. Get Unique Subjects and Conditions ---
subject_list = unique(data.SubjectID);
condition_list = categories(data.Condition); 


% --- C. Loop Through Subjects and Conditions for Normalization ---
for i = 1:length(subject_list)
    current_subj = subject_list(i);
    
    for c = 1:length(condition_list)
        current_cond = condition_list(c);
        
        % 1. Create a filter index for the current subject AND condition
        idx = (data.SubjectID == current_subj) & (data.Condition == current_cond);
        if sum(idx) < 2, continue; end % Skip if not enough trials

        % 2. Extract Data for Normalization: The raw power cubes [Time x Freq x Channel x Trial]
        alpha_data_cell = data.AlphaAmplitude(idx);
        alpha_data_3D = cat(3, alpha_data_cell{:}); 

        % --- 3. Calculate Baseline Mean (Mu) and SD (Sigma) ---
        % Slice the baseline power from the 3D cube: [Time_base x Freqs x Trials]
        baseline_power_3D = alpha_data_3D(baseline_start_sample:baseline_end_sample, :, :);
        
        % Calculate Mu: Average across Time (1) and Trials (3)
        mu_baseline = mean(mean(baseline_power_3D, 1), 4); % Result: [1 x Freqs x 1]
        
        % Calculate Sigma: SD across Time (1) and Trials (3)
        sigma_baseline = std(std(baseline_power_3D, 0, 1), 0, 4); % Result: [1 x Freqs x 1]
        
        % Ensure sigma is not zero (safety shield)
        sigma_baseline(sigma_baseline == 0) = 1; 

        % --- 4. Apply Baseline-Referenced Z-Score to ALL Data Points ---
        % Data is [Time x Freqs x Channels x Trial]. We subtract Mu and divide by Sigma.
        alpha_power_zscored = bsxfun(@minus, alpha_data_3D, mu_baseline);
        alpha_power_zscored = bsxfun(@rdivide, alpha_power_zscored, sigma_baseline);
        
        % --- 5. Update the MasterTable with the Z-Scored Cubes ---
        alpha_data_cell_z = num2cell(alpha_power_zscored, [1 2]);
        data.AlphaAmplitude(idx) = alpha_data_cell_z;
        
        % --- 6. Z-Score the Stimulus Intensity (Vector Standardization) ---
        raw_intensity_subj_cond = data.StimIntensity(idx);
        data.StimIntensityZ(idx) = zscore_vector(raw_intensity_subj_cond);
        
    end % End Condition Loop
end % End Subject Loop


end