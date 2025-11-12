function  [TP, TN, FP, FN]  = confusion_matrix(actual_outcome, predicted_outcome)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% CONFUSION MATRIX CALCULATION (TN, FP, FN, TP)

% Note: We assume the outcome is: 1 = Seen (Positive), 0 = Unseen (Negative)

% --- 1. Calculate the Four Outcomes ---
N_Total = length(predicted_outcome);

% True Positives (TP): Actual = 1 AND Predicted = 1 (Correctly predicted "Seen")
TP = sum(actual_outcome == 1 & predicted_outcome == 1);

% True Negatives (TN): Actual = 0 AND Predicted = 0 (Correctly predicted "Unseen")
TN = sum(actual_outcome == 0 & predicted_outcome == 0);

% False Positives (FP): Actual = 0 AND Predicted = 1 (Type I Error: Predicted "Seen," but subject missed it)
FP = sum(actual_outcome == 0 & predicted_outcome == 1);

% False Negatives (FN): Actual = 1 AND Predicted = 0 (Type II Error: Predicted "Unseen," but subject saw it)
FN = sum(actual_outcome == 1 & predicted_outcome == 0);


% --- 2. Display the Results ---

disp(' ');
disp('--- CONFUSION MATRIX BREAKDOWN (1=SEEN, 0=UNSEEN) ---');
disp(['True Positives (Correctly Predicted SEEN): ' num2str(TP/N_Total)]);
disp(['True Negatives (Correctly Predicted UNSEEN): ' num2str(TN/N_Total)]);
disp(['False Positives (Predicted SEEN, Actual UNSEEN - Type I Error): ' num2str(FP/N_Total)]);
disp(['False Negatives (Predicted UNSEEN, Actual SEEN - Type II Error): ' num2str(FN/N_Total)]);

disp(['Total Classified Trials: ' num2str(TP + TN + FP + FN)]);

end