%% MATLAB Startup Script (Project Initialization)

% 1. Clean the environment to ensure a fresh start
clc;
clear;

% --- A. DEFINE ALL NECESSARY PATHS (UPDATE THESE) ---
% These variables locate your resources outside the repo.
EEGLAB_ROOT_PATH = 'C:\Users\ssassi\Desktop\eeglab2025.1.0'; 
% BIOSIG_CORE_PATH removed, replaced by specific read_24bit path.

% Find the current repository root based on this script's location
this_script_path = fileparts(which('startup.m'));
repo_root = fileparts(this_script_path); 

% --- NEW CRITICAL PATH: FOLDER CONTAINING read_24bit.m and .mexw64 ---
% This path MUST point directly to the folder with the compatible read_24bit files.
LAB_IO_HELPER_PATH = fullfile(repo_root, '02_CODE', 'tools', 'fieldtrip_io_helpers'); % <--- FIX THIS PATH

% --- B. SET PRIORITY (The Fix for Conflicts) ---
% We use '-begin' to push custom/essential functions to the top of the path.

% 1. Add Lab I/O Helpers (Highest Priority for read_24bit)
addpath(LAB_IO_HELPER_PATH, '-begin');
disp('Lab I/O Helpers (read_24bit) added to path.');

% 2. Add Custom Lab Tools (HPF, multichanplot, etc.)
addpath(fullfile(repo_root, '02_CODE', 'tools'), '-begin');
addpath(fullfile(repo_root, '02_CODE', 'tools', 'channel_locs'), '-begin');

% 3. Add EEGLAB Root
addpath(EEGLAB_ROOT_PATH, '-begin'); 
disp('EEGLAB initialized to high priority.');


% --- C. FINAL ACTIONS ---

% Remove any conflicting paths that may have persisted
rmpath(genpath(EEGLAB_ROOT_PATH)); 

% 4. Initialize EEGLAB (This loads all plugins and GUI)
try
    eeglab; 
    disp('Environment fully set up and stable.');
catch ME
    disp('WARNING: EEGLAB GUI failed to open. Check paths.');
end

