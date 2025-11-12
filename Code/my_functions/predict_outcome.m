function [actual_outcome, predicted_outcome, accuracy] = predict_outcome(data, model, threshold)

if threshold < 0.5
    baseline_accuracy = 1 - threshold;
else
    baseline_accuracy = threshold;
end

% --- 1. Get Predicted Probabilities for all trials in the MasterTable ---
% The predict function returns the probability of the outcome being '1' (Seen).
predicted_prob = predict(model, data);

% --- 2. Determine Binary Prediction (Threshold = 0.5) ---
% Prediction is 1 (Seen) if probability >= 0.5, else 0 (Unseen).
predicted_outcome = (predicted_prob >= threshold); 

% --- 3. Compare Prediction to Actual Outcome ---
actual_outcome = double(data.SubjectiveOutcome); % Convert logical back to double (0/1)

% Calculate Accuracy: (Correct Predictions) / (Total Trials)
N_Correct = sum(predicted_outcome == actual_outcome);
N_Total = length(actual_outcome);
accuracy = N_Correct / N_Total;

disp('------------------------------------------');
disp('MODEL PERFORMANCE:');
disp(['Total Trials Classified: ' num2str(N_Total)]);
disp(['Overall Prediction Accuracy: ' num2str(accuracy * 100, 4) '%']);
disp('------------------------------------------');


figure('Units', 'normalized', 'Position', [0.7 0.2 0.3 0.4]);
bar_data = [accuracy, baseline_accuracy]; % Compare Accuracy to Chance

bar(bar_data);
set(gca, 'XTickLabel', {'Model Accuracy', 'Chance Level'});
ylim([0 1]);
title('Overall Single-Trial Prediction Accuracy', 'FontSize', 14);
ylabel('Accuracy (Proportion Correct)');
grid on;

end