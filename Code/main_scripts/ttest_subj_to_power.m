%% First Step of the Actual Project: t-test subjective to power
%% 1. LOAD PREPROCESSED DATA

%file reading code from ERP.m script

clear; close all; clc
disp('--- t-test subjective to power ---');

% --- Configuration ---
DEFAULT_PATH = 'C:\Users\ssassi\Desktop\Assaf_Rotation\Data'; % Default folder where your processed data is saved
%example christina data: \\wks3\pr_breska\el-Christina\Backup Copy Christina\PF_Poster\Data\EEG\
% christina eeg trigger list "\\wks3\pr_breska\el-Christina\SxA\SxA_Data\EEGTriggerList.docx"


% Use uigetfile to locate the Alpha Results .mat file
[filename, filepath] = uigetfile({'*AlphaResults*.mat','Alpha Results File (*.mat)' ;'*.*', 'All files (*.*)'},...
                                    'Select Alpha Power Results File');

if isequal(filename, 0)
    disp('No file selected. Aborting.');
    return; 
end

full_file_path = fullfile(filepath, filename);

% Load the data structure directly into the workspace
% NOTE: This brings ALL variables inside the .mat file into your workspace.
load(full_file_path);

disp(['Loaded data from: ' filename]);

%% 2. Next Step
