%% 1. LOAD PREPROCESSED DATA

%file reading code from ERP.m script

clear; close all; clc
disp('--- Power x Subjective ---');

% --- Configuration ---
DEFAULT_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Data'; % Default folder where your processed data is saved
%example christina data: \\wks3\pr_breska\el-Christina\Backup Copy Christina\PF_Poster\Data\EEG\
% christina eeg trigger list "\\wks3\pr_breska\el-Christina\SxA\SxA_Data\EEGTriggerList.docx"

% --- 1. Use UIGETFILE for Interactive Selection (GUI Dialog) ---
[filename, filepath] = uigetfile({'*.mat','MATLAB Data File (*.mat)' ;'*.*', 'All files (*.*)'},...
                                    'Select Clean Preprocessed SDATA File', DEFAULT_PATH);

if isequal(filename, 0)
    disp('No file selected. Aborting script.');
    return; 
end

full_file_path = fullfile(filepath, filename);

% --- 2. Load the Data ---
try
    % Load only the SDATA variable from the selected .mat file
    load(full_file_path, 'SDATA');
    disp(['Loaded clean data from: ' filename]);
catch ME
    disp(['ERROR: Failed to load SDATA structure from ' filename]);
    disp(['MATLAB Error: ' ME.message]);
    error('Aborting due to critical data loading failure.');
end

