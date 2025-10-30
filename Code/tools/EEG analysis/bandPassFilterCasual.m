function bandPassFiltered=bandPassFilterCasual(low,high,signal,rate)

% filter parameters
stopbandDistanceFromCutoff=1; % in octaves, the lower it is the steeper the filter function (and more ripples)
attenuateAtStopband=24; % in dB, the higher it is the steeper the filter function (and more ripples)
maxRipplesAllowed=3; % in dB, the lower it is the filter function must be less steep

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

[b,a] = butter(nband,Wnband,'bandpass'); % design filter
bandPassFiltered = filter(b,a,signal); % apply filter

% filter again with reduced filter order if the filtered signal is distorted or all NaN
% (which happens if normalized low cutoff is too close to 0)
if max(max(abs(bandPassFiltered)))> 1000*max(max(signal)) || max(sum(isnan(bandPassFiltered)))==length(bandPassFiltered)
    [b,a] = butter(nband-1,Wnband,'bandpass');
    bandPassFiltered = filter(b,a,signal);
    disp('filter order reduced to prevent data distortion')
end

end