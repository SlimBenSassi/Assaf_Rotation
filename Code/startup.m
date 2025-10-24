%% MATLAB Startup Script for EEG Lab Rotation

% 1. Clean up the environment
clc;
clear;

% --- A. Define Project Root and Paths ---
this_script_path = fileparts(which('startup.m'));
repo_root = fileparts(this_script_path);

% Define the full path to the Lab's Custom Functions
CUSTOM_TOOLS_PATH = fullfile(repo_root, 'Code', 'tools'); 


% -----------------------------------------------------------
% *** CRUCIAL CHANGE: ADD REPO PATHS BEFORE EEGLAB ***
% -----------------------------------------------------------

% 2. Add the Lab's specific custom functions (HPF, read_biosemi_bdf, etc.) FIRST
addpath(CUSTOM_TOOLS_PATH);
% Add the rest of the repo structure (main_scripts, etc.)
addpath(genpath(repo_root)); 
disp('Project structure and Lab Tools added to path.');

CHANNEL_LOCS_PATH = fullfile(repo_root, 'Code', 'tools'); 

% Add the channel locations folder
addpath(CHANNEL_LOCS_PATH); 
disp('Channel Locations added to path.');


% --- C. INITIALIZE EEGLAB (Lower Priority) ---
EEGLAB_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Code\eeglab2025.1.0'; 

% Remove any conflicting paths first (critical)
rmpath(genpath(EEGLAB_PATH)); 

% Add ONLY the root EEGLAB folder to the path
addpath(EEGLAB_PATH); 

% Initialize EEGLAB (This now has lower precedence than your custom tools)
try
    eeglab; 
    disp('EEGLAB initialized (Lower Precedence).');
catch ME
    disp('WARNING: EEGLAB init failed.');
end
