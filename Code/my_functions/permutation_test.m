function [null_accuracies, p_value_permutation] = permutation_test(data, model_formula, threshold, real_accuracy, n_permutations)

null_accuracies = zeros(n_permutations, 1);
MasterTable_original = data; % Keep a copy of the original data

disp(['Running Permutation Test (N = ' num2str(n_permutations) ')...']);

for i = 1:n_permutations
    
    % 1. SHUFFLE THE OUTCOME (The clean fix for complex models)
    % Create a randomly permuted index list based on the total number of trials.
    shuffled_indices = randperm(size(MasterTable_original, 1));
    
    % 2. Create the Null Table by injecting the shuffled outcome (Y)
    MasterTable_null = MasterTable_original;
    % We shuffle the SubjectiveOutcome column (Y) while keeping X1 and X2 in place.
    %MasterTable_null.SubjectiveOutcome = MasterTable_null.SubjectiveOutcome(shuffled_indices);
    MasterTable_null.AlphaPower_Avg_100ms = MasterTable_null.AlphaPower_Avg_100ms(shuffled_indices); % change to
    %this for harsher test

    % 3. RUN THE NULL MODEL (Recalculate the GLMM on shuffled data)!
    glme_null = fitglme(MasterTable_null, model_formula, ...
                        'Distribution', 'Binomial', 'Link', 'logit', 'Verbose', 0, ...
                        'FitMethod','Laplace');
    
    % 4. GET NULL ACCURACY
    predicted_outcome_null = (predict(glme_null, MasterTable_null) >= threshold); 
    actual_outcome_null = double(MasterTable_null.SubjectiveOutcome);
    
    null_accuracies(i) = sum(predicted_outcome_null == actual_outcome_null) / length(actual_outcome_null);
end

p_value_permutation = sum(null_accuracies >= real_accuracy) / n_permutations;

disp('--- PERMUTATION TEST RESULT ---');
disp(['Real Accuracy: ' num2str(real_accuracy * 100, 3) '%']);
disp(['95th Percentile of Null (Chance Threshold): ' num2str(prctile(null_accuracies, 95) * 100, 3) '%']);
disp(['Permutation P-value (P_perm): ' num2str(p_value_permutation, 4)]);


figure('Units', 'normalized', 'Position', [0.1 0.1 0.5 0.5]);

% --- 1. Plot the Null Distribution (Histogram) ---
h = histogram(null_accuracies, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); 
hold on;

% --- 2. Define Critical Thresholds ---
% Calculate the 95th percentile (the significance cut-off)
chance_threshold_95 = prctile(null_accuracies, 95); 

% --- 3. Add Vertical Lines ---
% A. Add the line for your actual, achieved accuracy (62.34%)
line([real_accuracy real_accuracy], ylim, 'Color', 'b', 'LineWidth', 3, 'DisplayName', 'Actual Model Accuracy');

% B. Add the line for the 95th percentile (the chance boundary)
line([chance_threshold_95 chance_threshold_95], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', '95% Significance Threshold');

% --- 4. Aesthetics ---
title('Model Performance vs. Null Distribution (Permutation Test)', 'FontSize', 14);
xlabel('Prediction Accuracy (Proportion Correct)');
ylabel('Frequency (N of Null Models)');
legend('show', 'Location', 'NorthWest');
hold off;


end