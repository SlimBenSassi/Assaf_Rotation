function filteredSig=bandPassFilter(lowCutoff,highCutoff,signal,rate)
% bandPassFilter: simple butterworth non-causal bandpass filter


highpass = lowCutoff;
lowpass = highCutoff;

if (2*lowpass)/(0.5*rate)>=1 % check whether requested attentuation range (second input of buttord) does not exceed what is possible by sampling rate
    error(['In current filter params, highest cutoff must be less than ' num2str(rate/4)]);
else
    [nband,Wnband]=buttord([highpass/(0.5*rate) lowpass/(0.5*rate)],[0.5*highpass/(0.5*rate) (2*lowpass)/(0.5*rate)], 3 ,24);
end

[b,a] = butter(nband,Wnband,'bandpass') ;
filteredSig = filtfilt(b,a,signal);

if max(filteredSig)> 1000*max(signal)
    [b,a] = butter(nband-1,Wnband,'bandpass') ;
    filteredSig = filtfilt(b,a,signal);
    disp('filter order reduced to prevent data distortion')
end

end