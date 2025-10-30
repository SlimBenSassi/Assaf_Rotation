function [hil_amp, hil_phase]=TFhilbert(frequencies, filtWidth, signal, fs, causalFilt)

% Inputs:
% frequencies: a vector of the frequencies for the TF transform
% filtWidth: width in octaves of the filter around each freq. small values favor spectral resolution, large values favor temporal resolution
% signal: trials/channels X time_points data matrix
% fs: Sampling frequency of the signal
% causalFilt (optional): use causal filter, default false
%
% written by assaf breska, bug reports to assaf.breska@gmail.com


if nargin<5 % option to use a causal filter
    causalFilt=false;
end

nFreq=length(frequencies);
[nTrials, nSamp]=size(signal);

hil_amp=zeros(nFreq, nSamp, nTrials);
hil_phase=zeros(nFreq, nSamp, nTrials);

freqBand_lowBound=frequencies*2^-filtWidth; % band width for each frequency according to filter width
freqBand_highBound=frequencies*2^filtWidth;

for currentFreq=1:nFreq
    
    if causalFilt
        filtered=bandPassFilterCasual_forTFh(freqBand_lowBound(currentFreq), freqBand_highBound(currentFreq), signal', fs);
    else
        filtered=bandPassFilter_forTFh(freqBand_lowBound(currentFreq), freqBand_highBound(currentFreq), signal', fs);
    end
    
    hil=hilbert(filtered);
    
    hil_amp(currentFreq,:,:)=abs(hil); % amplitude
    hil_phase(currentFreq,:,:)=angle(hil); % phase
    
end

end


% % AUXILIARY FUNCTIONS

function bandPassFiltered=bandPassFilter_forTFh(low,high,signal,rate)
% non-causal butterworth filter

% filter parameters
stopbandDistanceFromCutoff=1; % in octaves
attenuateAtStopband=24; % in dB
maxRipplesAllowed=3; % in dB

highpass = low;
lowpass = high;

% get the lowset order possible for butterworth filter
normalizedCutoffs=[highpass lowpass]/(0.5*rate);
normalizedStopbands=normalizedCutoffs.*[2^-stopbandDistanceFromCutoff 2^stopbandDistanceFromCutoff];
if normalizedStopbands(2)>=1 % check whether requested attentuation range (second input of buttord) does not exceed what is possible by sampling rate
    error(['bad high cutoff frequency. In current filter params, it must be less than ' num2str(0.5*rate*2^-stopbandDistanceFromCutoff)]);
else
    [nband,Wnband]=buttord(normalizedCutoffs, normalizedStopbands, maxRipplesAllowed ,attenuateAtStopband);
end

[b,a] = butter(nband,Wnband,'bandpass') ;
bandPassFiltered = filtfilt(b,a,signal);

if max(max(abs(bandPassFiltered)))> 1000*max(max(signal)) || max(sum(isnan(bandPassFiltered)))==length(bandPassFiltered)
    [b,a] = butter(nband-1,Wnband,'bandpass') ;
    bandPassFiltered = filtfilt(b,a,signal);
    disp('filter order reduced to prevent data distortion')
end

end



function bandPassFiltered=bandPassFilterCasual_forTFh(low,high,signal,rate)
% causal butterworth filter

% filter parameters
stopbandDistanceFromCutoff=1; % in octaves
attenuateAtStopband=24; % in dB
maxRipplesAllowed=3; % in dB

highpass = low;
lowpass = high;

% get the lowset order possible for butterworth filter
normalizedCutoffs=[highpass lowpass]/(0.5*rate);
normalizedStopbands=normalizedCutoffs.*[2^-stopbandDistanceFromCutoff 2^stopbandDistanceFromCutoff];
if normalizedStopbands(2)>=1 % check whether requested attentuation range (second input of buttord) does not exceed what is possible by sampling rate
    error(['bad high cutoff frequency. In current filter params, it must be less than ' num2str(0.5*rate*2^-stopbandDistanceFromCutoff)]);
else
    [nband,Wnband]=buttord(normalizedCutoffs, normalizedStopbands, maxRipplesAllowed ,attenuateAtStopband);
end

[b,a] = butter(nband,Wnband,'bandpass') ;
bandPassFiltered = filter(b,a,signal);

if max(max(abs(bandPassFiltered)))> 1000*max(max(signal)) || max(sum(isnan(bandPassFiltered)))==length(bandPassFiltered)
    [b,a] = butter(nband-1,Wnband,'bandpass') ;
    bandPassFiltered = filter(b,a,signal);
    disp('filter order reduced to prevent data distortion')
end

end

