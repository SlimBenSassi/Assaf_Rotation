function [wvlt_amp, wvlt_phase]  = morletwave(frequencies,c,signal,fs,varargin)
% Compute absolute of Morlet wavelet spectrograms.
% 
% Inputs:
% frequencies: a vector of the frequencies for the wavelet transform
% c: the wavelet constant - the ratio between the window width and the cycle length.
% c: the wavelet constant - the ratio between the mean (the center
% frequency) and the standard deviation (the window width) of the frequency
% Gaussian window, or the number of oscillation cycles within a +- 1
% standard deviation of the temporal Gaussian window times pi: 
% c = fc / sig_f = 2pi*sig_t / T.   
%     c = 0 yields the original time-courses (a zero width time window);
%     c -->inf yields the Fourier transform
%     default is 12 (equals 8 in brainVision Analyzer). 
% signal: the data matrix - each row is a time course of a single trial or channel.
% fs: Sampling frequency of the signal. Defualt = 1024Hz.
% bl_range: Baseline Correction - 0=no BC; [a b]=custom range. 
% 
% Optional input:
% 'waitbar' followed by 'on' or 'off' - controlls whether a waitbar appears (default is 'on').
%
%
% Output:
% wvlt_amp: the time-frequency wavelet matrices for all trials or channels.
%     It is of size (# of frequencies) x (# of time points) x (# of trials or channels)
% wvlt_phase: the time-phase wavelet matrices for all trials or channels.
%     It is the same size and format as wvlt_amp.
% 


% handle inputs
if mod(size(varargin),2) == 1 % check input arguments
    error('Wrong number of input arguments');
end


if size(signal,2) == 1 % handle column vector input
    signal = signal';
end

ShowWB = false;
detrend = true;
arg  = 1;
while arg < size(varargin,2)
    switch varargin{arg}        
        case 'waitbar'
            if strcmp(varargin{arg+1}, 'on')
                ShowWB = true;
            elseif strcmp(varargin{arg+1}, 'off')
                ShowWB = false;
            else
                error('Illegal value for ''waitbar'' setting: use ''on'' or ''off''.');
            end
            arg = arg + 2;
        case 'detrend'
            if strcmp(varargin{arg+1}, 'yes')
                detrend = true;
            elseif strcmp(varargin{arg+1}, 'no')
                detrend = false;
            else
                error('Illegal value for ''detrend'' setting: use ''yes'' or ''no''.');
            end
            arg = arg + 2;            
        otherwise
            error(['Unknown optional argument name: ' varargin{arg} '.']);
    end
end

if ~exist('c','var')
    c = 12; 
end

% create wavelets    
sig_t = c./(2*pi*frequencies);  % calculate width of gaussian for each frequency 
twin_wav = 4 * sig_t'; % time window for wavelets of 4 sd, broad enough to absorb the decay of slowest frequency
[~,imax] = min(frequencies);
t_wav= -twin_wav(imax):(1/fs):twin_wav(imax);
morletwavsMat  = exp( 2*1i*pi*frequencies'*t_wav ) .* exp( - sig_t'.^-2 * t_wav.^2 /2);

% Normalize
nFreqs = length(frequencies);
for freq = 1:nFreqs
    morletwavsMat(freq,:) = morletwavsMat(freq,:) / sum(abs(morletwavsMat(freq,:))) * 2;
end

% calculate time parameters for convolution
[trials, nTimePoints] = size(signal);
winw = floor(twin_wav*fs);
win = winw*ones(1,nTimePoints) + ones(nFreqs,1)*(1:nTimePoints);
Ntw = length(t_wav);    % # of time points in each wavelet
wlim = ceil(Ntw/2) + winw * [-1 1];


% conduct TF analysis
if ShowWB
    h = waitbar(0,'Calculating wavelets...');
end

% 1-D convolution of each single frequency with each single trial/channel:
wvlt_amp = single(zeros(nFreqs,nTimePoints, trials));
wvlt_phase = single(zeros(nFreqs,nTimePoints, trials));
for trial = 1:trials
    if ShowWB 
        if trials > 1 
            waitbar(trial/trials);
        end
    end
    
    trialData=signal(trial,:);
    if detrend
        trialData=trialData-linspace(trialData(1), trialData(end),length(trialData));
    end
%     figure; plot(trialData)

    for freq = (1:nFreqs)         
        wvlt_trial = conv(morletwavsMat(freq,wlim(freq,1):wlim(freq,2)),trialData); % convolution of wavelet and signal
        
        wvlt_amp(freq,:,trial) = abs(wvlt_trial(win(freq,:)));   % taking amp while cutting extra data create by convolution
        wvlt_phase(freq,:,trial) = angle(wvlt_trial(win(freq,:))); % same for phase 
    end    
end

if ShowWB
    close(h)
end

end