%% 1. INITIALIZE AND PATH SETUP
clear; close all; clc

% --- PROJECT CONFIGURATION FLAG ---


% NECESARRY STEPS TO RUN THIS WITHOUT DEPENDENCY ERRORS:
% Add Code and its subfolder to path
% Add eeglab to path (not subfolders)
% run eeglab; in command window



USE_SIMULATED_DATA = false; 
% Set to TRUE for quick testing with fake data (Requires generate_simulated_eeg.m)
% Set to FALSE to open the file selection dialog and load real BDF files.
% ----------------------------------

% --- CONFIGURATION (UPDATE THESE LINES) ---
raw_data_path = "\\wks3\pr_breska\el-Slim\random assaf data (controls)"; 

% --- NECESSARY INFO ---
% We will use EEGLAB's pop_biosig to load BDF files, bypassing the crashing read_biosemi_bdf.

%% 2. CHOOSE DATA FILE(S), LOAD, AND APPEND

if USE_SIMULATED_DATA
    
    % --- SIMULATED DATA MODE ---
    disp('*** Running in SIMULATED DATA MODE ***');
    % This script loads SDATA, Fs, and time_vec into the workspace.
    generate_simulated_eeg; 
    
    nSessions = 1; 
    
else % --- REAL DATA MODE (Original UI Logic) ---
    
    if ~exist('raw_data_path','var')
        raw_data_path = cd;
    end
    
    % 1. Select files
    [data_file_list, new_data_path] = uigetfile({'*.bdf','BDF file (*.bdf)' ;'*.*', 'All files (*.*)'},...
                                                'Select raw BDF file(s)',raw_data_path,'multiselect','on');
    
    % --- Handle File Selection ---
    if isscalar(data_file_list) && isequal(data_file_list, 0)
        disp('No files selected. Script terminated.');
        return; 
    else
        raw_data_path = new_data_path;
        if ~iscell(data_file_list)        
            data_file_list = {data_file_list}; 
        end
    end
    
    nSessions = length(data_file_list);
    
    % 2. Initialize the SDATA structure and start loading loop
    SDATA = struct('info', struct(), 'metadata', struct(), 'events', struct());
    EEGdata = []; 
    disp(['Found ' num2str(nSessions) ' session(s). Reading data files...']);
    
    tic
    % 3. Run over sessions
    for sessIdx = 1:nSessions
        
        fileName = fullfile(raw_data_path, data_file_list{sessIdx});
        disp(['Processing file: ' data_file_list{sessIdx}]);
        
        [EEG_temp] = pop_biosig(fileName)%, 'importevent', 'off'); 
        
        % Data Extraction and Structuring
        session_data = single(EEG_temp.data)'; 
        EEGdata=[EEGdata; session_data];
        
        hdr.Fs = EEG_temp.srate;
        hdr.label = {EEG_temp.chanlocs.labels};
        toc
    end
    
    % 4. Finalize SDATA structure after loop
    SDATA.data = EEGdata; 
    SDATA.info.sampling_rate = hdr.Fs;
    SDATA.info.channel_labels = hdr.label;
    SDATA.data = double(SDATA.data);
     % --- CRITICAL FIX: TRANSFER REAL EVENT MARKERS (from EEG_temp) ---
    % Assumes EEG_temp from the last iteration contains the complete event list.
    if isfield(EEG_temp, 'event') && ~isempty(EEG_temp.event)
        
        % Extract latencies (which are in samples/time points)
        % We convert the list of structures to a vector of sample indices
        event_latencies = [EEG_temp.event.latency];
        
        % We only care about the time point indices (in samples)
        SDATA.events.trigger_indices = event_latencies'; 
        disp(['Transferred ' num2str(length(event_latencies)) ' real event markers from BIOSIG.']);
        
    else
        disp('Warning: No events found in EEG_temp structure.');
    end

end % END MODE SWITCH

% --- Log the data sizes (Crucial Check) ---
disp(['Total data loaded: ' num2str(size(SDATA.data, 1)) ' time points x ' num2str(size(SDATA.data, 2)) ' channels']);

% --- Retrieve Time Vector (Needed later for plotting) ---
Fs = SDATA.info.sampling_rate;
SDATA.time_vec = (0:size(SDATA.data, 1) - 1) / Fs;


% Note: In a real analysis, you would run the uigetfile block here, but we 
% skip it for simplicity since we have no physical file to select.
   

%% 2. MARK BAD ELECTRODES (Check and Remove EXG8)

% We plot the first 10 seconds of data for a quick scan.
% --- FIXED CALL ---
multichanplot(SDATA.data, 10, 'srate', SDATA.info.sampling_rate, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);

channel_to_inspect = SDATA.data(:,end);
multichanplot(SDATA.data(:,end), 10, 'srate', SDATA.info.sampling_rate, 'ylim', [-100 100]);
% Decision prompt (The script pauses here)
removeEXG8 = input('Remove EXG8/Status channel? (1=yes, 0=no): '); 

if removeEXG8 == 1
    % The lab's removal logic (removes the last column)
    SDATA.data = SDATA.data(:, 1:end-1);
    SDATA.info.channel_labels = SDATA.info.channel_labels(1:end-1);
    disp('EXG8/Status channel removed.');
else
    disp('EXG8/Status channel retained.');
end


%% 3.1 FILTERING: HIGH PASS + LINE NOISE REMOVAL




% --- ASSUMPTION: The lab's custom HPF and LPF functions are now available on the path. ---
% The custom HPF function is used here for demonstration.
hpf_cutoff = 0.5;
lpf_cutoff = 50;
filter_order = 3; % Using 3 for numerical stability

disp('Applying High-Pass Filter...');
% *** Using Lab's Custom HPF ***
SDATA.data = HPF(SDATA.data, Fs, hpf_cutoff, filter_order); 

disp('Applying Low-Pass Filter...');
% *** Using Lab's Custom LPF ***
SDATA.data = LPF(SDATA.data, Fs, lpf_cutoff, filter_order); 

disp('Filtering complete.');



%% 3.2 INSPECT FILTERING RESULT

% Visualize the effect of filtering on the DC-drift channel (Channel 2 in our simulation)

hpf_cutoff_for_disp=0.1;
% Apply a visualization filter (optional, just to remove heavy drift for plotting)
show_data = HPF(SDATA.data, Fs, hpf_cutoff_for_disp, 3); 

% Plotting using the lab's function
multichanplot(show_data, 10, 'srate', Fs, 'channelnames', SDATA.info.channel_labels, 'ylim', [-100 100]);

clear show_data hpf_cutoff_for_disp hpf_cutoff lpf_cutoff filter_order % Clear temporary variables

%% 4.0 Reference

disp('Applying Re-Reference...');

% --- CONFIGURATION: DEFINE THE REFERENCE(S) ---
% NOTE: In a real script, this would be M1 and M2, or a single electrode.
% We will use two simulated channels (e.g., Channel 15 and 20) as the reference sites.
ref_channel_indices = [69, 70]; 

% 1. Extract the data for the reference channels
ref_data = SDATA.data(:, ref_channel_indices);

% 2. Calculate the reference signal (the average time course across the reference channels)
% We calculate the mean across dimension 2 (the channels in ref_data)
mean_ref_signal = mean(ref_data, 2); 

% 3. Apply the Reference Subtraction (Using bsxfun for efficiency)
% bsxfun(@minus, A, B) automatically subtracts the mean_ref_signal [Time x 1] 
% from every column of SDATA.data [Time x Channels].
SDATA.data = bsxfun(@minus, SDATA.data, mean_ref_signal);

disp('Reference signal (average of selected channels) applied successfully.');

% --- REMOVE REFERENCE CHANNELS ---
% After referencing, the reference channels themselves are often discarded 
% from the analysis data, as their new data is redundant or unusable.
channels_to_keep = setdiff(1:size(SDATA.data, 2), ref_channel_indices);

SDATA.data = SDATA.data(:, channels_to_keep);
SDATA.info.channel_labels = SDATA.info.channel_labels(channels_to_keep);
disp(['Removed ' num2str(length(ref_channel_indices)) ' reference channel(s).']);

% --- INSPECTION ---
disp(['New channel count: ' num2str(size(SDATA.data, 2))]);

%% 4.1 DETREND 
DATA_TO_DETREND = SDATA.data;

% Remove linear trends from the filtered data
DATA_DETRENDED = detrend(DATA_TO_DETREND, 'linear'); 

SDATA.data = DATA_DETRENDED;

%% 4.2 Visualize Detrended data

 

% --- CONFIGURATION ---
Fs = SDATA.info.sampling_rate;
% Assuming DATA_TO_DETREND holds the data before this step, 
% and SDATA.data holds the data after this step.

% --- INSPECTION 1: BEFORE DETRENDING ---
% Plot the data before detrending. This plot should show a slight tilt or drift.
multichanplot(DATA_TO_DETREND, 10, 'srate', Fs, ...
              'channelnames', SDATA.info.channel_labels, ...
              'ylim', [-100 100]);

% --- INSPECTION 2: AFTER DETRENDING ---
% Plot the data after detrending. The baseline should be perfectly flat/horizontal.
multichanplot(SDATA.data, 10, 'srate', Fs, ...
              'channelnames', SDATA.info.channel_labels, ...
              'ylim', [-100 100]);

disp('Check Figure 1 and Figure 2 side-by-side to verify baseline flattening.');

%% 5.1 PREP: Convert SDATA to EEGLAB Format for ICA

% ICA/EEGLAB functions require data in the format: [Channels x Time]

Fs = SDATA.info.sampling_rate;
n_channels = size(SDATA.data, 2);

% Create a temporary EEG structure (required for EEGLAB plotting/inspection)
EEG_ICA = struct();

% Transpose data to the [Channels x Time] format
EEG_ICA.data = SDATA.data'; 
EEG_ICA.srate = Fs;
EEG_ICA.nbchan = n_channels;
EEG_ICA.pnts = size(EEG_ICA.data, 2);
EEG_ICA.xmin = 0; % Start time

disp(['ICA Input Data Size: ' num2str(size(EEG_ICA.data)) ' (Channels x Time)']);

%% 7.2: RUN ICA (Adopting Lab's N-1 Rank Strategy with Time Limit)

% --- Configuration ---
n_channels = size(EEG_ICA.data, 1);
% The lab's strategy: Reduce rank by one for stability/speed (N-1 components).
pca_rank = n_channels - 1; 
TARGET_MAXSTEPS = 10; % Time constraint: Stop after 50 steps for quick debugging

disp(['Starting runica algorithm with PCA reduction to ' num2str(pca_rank) ' components and maxsteps=' num2str(TARGET_MAXSTEPS) '...']);

% 1. Run ICA: The 'pca' argument forces PCA reduction before ICA runs.
[EEG_ICA.icaweights, EEG_ICA.icasphere] = runica(EEG_ICA.data, ...
    'pca', pca_rank, ... 
    'extended', 1, ...
    'maxsteps', TARGET_MAXSTEPS); % <-- IMPLEMENTED TIME CONTROL

% 2. Calculate the Mixing Matrix (icawinv = Inverse of Unmixing)
EEG_ICA.icawinv = pinv(EEG_ICA.icaweights * EEG_ICA.icasphere); 

% 3. Calculate the Component Time-Courses (scores)
EEG_ICA.icadata = EEG_ICA.icaweights * EEG_ICA.icasphere * EEG_ICA.data;

disp(['ICA Decomposition complete (early stop at step ' num2str(TARGET_MAXSTEPS) '). Found ' num2str(size(EEG_ICA.icadata, 1)) ' components.']);
%% 7.3 & 7.4: INSPECT COMPONENTS (Time Course and Topography)

disp('Generating component inspection plots...');

% --- 7.3 Plot Component Time Courses ---
% Plots the unmixed activity of the components. Look for massive spikes (blinks).
figure;
multichanplot(EEG_ICA.icadata', 'srate', Fs);
% Note: Use the scroll bar to find the blink and muscle components!

%% 7.4: PLOT ICA COMPONENT TOPOGRAPHIES (The Final Working Visualization)


disp('Starting interactive component inspection...');

% --- 1. Map ICA Results to Lab's Structure (A proxy for SDATA.ica) ---
% We use the N_components_found calculated previously.
N_components = size(EEG_ICA.icadata, 1); 
all_channels = 1:size(SDATA.data, 2);

% Assuming NO electrodes were marked as bad initially (only EXG8 was potentially removed)
SDATA.metadata.good_electrodes = all_channels; 
SDATA.metadata.bad_electrodes = []; % Assuming no bad channels for now

% The Mixing Matrix (A) needs to be transposed to be compatible with how the lab 
% extracts the topography weights for topoplot.
mixing_matrix = EEG_ICA.icawinv; 


% --- 2. Start the Interactive Plotting Loop ---
clc
h = figure('Position', [500 50 500 500]); 
done = 0;

while ~done
    
    clc
    % Prompt the user for the component number to plot
    compToPlot = input('Input number of component to plot (1 to N): ');
    
    if compToPlot > N_components || compToPlot < 1
        disp(['Error: Component number must be between 1 and ' num2str(N_components)]);
        continue; % Skip to the next loop iteration
    end
        
    % --- 3. Build the Topography Vector for topoplot ---
    % The topography is built by extracting the mixing weights (the column of icawinv).
    compTopo = zeros(length(SDATA.info.channel_labels), 1);
    
    % Get the weights for the current component from the mixing matrix
    % NOTE: The mixing matrix (A) is [Channels x Components]. We take the correct column.
    compTopo = mixing_matrix(:, compToPlot);
    
    % --- Apply Masking for Bad Electrodes (The lab's logic) ---
    % If any electrodes were marked as bad, set their weight to NaN to hide them.
    compTopo(SDATA.metadata.bad_electrodes) = NaN;
    
    % --- 4. Generate the Plot ---
    % Topoplot is now called with the weight vector and the head71 location file.
    topoplot(compTopo, 'head71.locs', 'electrodes', 'on', 'style', 'map', 'shading', 'interp'); 
    title(['ICA Component ' num2str(compToPlot)], 'FontSize', 14);

    % --- 5. User Decision ---
    figure(h.Number); % Bring the figure window forward
    userInp = input('Finished inspecting this component? (1=yes, 0=no): ');
    if userInp == 1
        done = 1;
    end
    
end
close(h); % Close the figure when done
disp('Interactive inspection finished. Proceed to component rejection.');

%% 7.5: REMOVE COMPONENTS AND RECONSTRUCT DATA

% --- Decision Step (Manual Input) ---
% Based on inspection (EOG/EMG patterns), we assume components 1 and 2 are artifacts.
artifact_components_str = input('Enter component indices to remove (e.g., [1 2 5]): ', 's');
artifact_components = str2num(artifact_components_str); 

disp(['Removing components: ' num2str(artifact_components)]);

% 1. CRITICAL: Find the actual channel count (should be 70, due to N-1 rank reduction)
N_channels_processed = size(EEG_ICA.icawinv, 1); 
N_components_found = size(EEG_ICA.icadata, 1); 

components_to_keep = setdiff(1:N_components_found, artifact_components);

% 2. Reconstruct the data: We must ensure the mixing matrix (icawinv) is sliced 
%    to the correct channel dimensionality (N_channels_processed).
%    The problem is often that icawinv has an extra row that needs to be removed 
%    if your initial data was 70 channels.

% --- The Correct Reconstruction (Ensuring Channel Dimension is Correct) ---
EEG_CLEAN.data = (EEG_ICA.icawinv(1:N_channels_processed, components_to_keep) * ...
                  EEG_ICA.icadata(components_to_keep, :));


%% 7.6 & 7.7: INSPECT AND APPROVE RECONSTRUCTION (Multi-Channel Comparison)

% --- Update the core data variable ---
% Data must be transposed back to the project's standard [Time x Channels] format.
SDATA.data = EEG_CLEAN.data'; 
disp('SDATA updated with ICA-cleaned data.');

% --- CONFIGURATION ---
Fs = SDATA.info.sampling_rate;
data_labels = SDATA.info.channel_labels;



% -------------------------------------------------------------
% --- INSPECTION 1: BEFORE ICA (FULL DURATION) ---
% Plot the raw, filtered data with artifacts.
multichanplot(DATA_DETRENDED, 10, 'srate', Fs, ...
              'channelnames', data_labels, ...
              'ylim', [-100 100]);


% -------------------------------------------------------------
% --- INSPECTION 2: AFTER ICA (FULL DURATION) ---
% Plot the final, clean data.
multichanplot(SDATA.data, 10, 'srate', Fs, ...
              'channelnames', data_labels, ...
              'ylim', [-100 100]);


disp('Check the two new figure windows to compare the full-duration traces.');
disp('ICA artifact removal pipeline finished.');





%% 10. SAVE SDATA (Final Preprocessed File)

clc
% We use a specific, unique filename for the preprocessed data.
subject_ids = 1; % Assume subject ID 1 for simulation
% Creates a filename like 'subj1_pp.mat'
fileNameSave = fullfile('Results', ['subj' num2str(subject_ids) '_pp.mat']); 

disp(['Saving clean, preprocessed data to: ' fileNameSave]);

% The clean data structure is now fully finalized and saved
save(fileNameSave, 'SDATA', '-v7.3'); % Use -v7.3 for large files (optional)

disp(['Done saving clean data for subject ' num2str(subject_ids)]);