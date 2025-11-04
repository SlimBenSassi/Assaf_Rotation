function SDATA = reref_to_nose(SDATA)
% SDATA = apply_conditional_reref(SDATA) applies a re-reference from Mastoids (69, 70) to Nose (71)
% ONLY if the current reference is confirmed to be Mastoids.
%
% This function assumes channels 69, 70 are Mastoids and channel 71 is Nose.
%
% INPUTS:
%   SDATA: The main data structure containing SDATA.data and SDATA.metadata.reference.
%
% OUTPUTS:
%   SDATA: The modified data structure.

% Define the Mastoid and Nose indices as expected by the logic
MASTOID_CHANS = [69, 70];
NOSE_CHAN = 71;
MASTOID_REFERENCE_LABEL = 'Mastoids';

% Check if the current reference (SDATA.metadata.reference) is the Mastoid array.
% The 'isequal' function is necessary for comparing two arrays.
if isequal(SDATA.metadata.reference, MASTOID_REFERENCE_LABEL) || ...
   (isnumeric(SDATA.metadata.reference) && isequal(SDATA.metadata.reference, MASTOID_CHANS))
    
    disp('Reference detected as Mastoids. Applying M-to-N re-reference...');
    
    % 1. Calculate the current reference signal (Average of M1 and M2)
    current_ref_signal = mean(SDATA.data(:, MASTOID_CHANS), 2);
    
    % 2. Calculate the difference between the Nose electrode (new reference) and the old reference
    nose_to_mastoid_diff = SDATA.data(:, NOSE_CHAN) - current_ref_signal;
    
    % 3. Apply the re-reference: Subtract the difference from ALL channels.
    SDATA.data = SDATA.data - nose_to_mastoid_diff;
    
    % 4. Update the metadata log
    SDATA.metadata.reference = NOSE_CHAN; % Store the new reference index (71)
    disp('Re-reference successful: New reference is Nose (71).');
else
    disp(['Data reference (' num2str(SDATA.metadata.reference) ') retained.']);
end

end
