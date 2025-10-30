function [segmentedData, isNotArtifact, timeVec]=segmentContEEGdata(triggerCode, segBoundsTime, contEEGdata, triggerVec, artifactVec, sampRate)
% segmentContEEGdata: segmentation of continuous EEG data according to event codes
% INPUTS:
% triggerCode: k x 1 vector of k different event codes in the event vector that mark the event 
%              relative to which segments should be extracted
% segBoundsTime: [tmin tmax] indicating the time window (in ms) to segment relative to the event, e.g. [-200 700])
% contEEGdata: data to be segmented, must be N (time in samples) by C (channels)
% triggerVec: vector with same length as the number of samples in the data (N x 1) with event
%             codes marking event time stamps
% artifactVec: vector with same length as the number of samples in the data (N x 1) with 1
%             marking samples with artifacts and 0 without
% sampRate: sampling rate of the data file
%
% OUTPUTS:
% segmentedData: n (segment length) x C (channels) x nTrials sized matrix of segments
% ifNotArifact: vector (size nTrials x 1), with 1 if trial doesnt contain artifacts and 0 if it does
% timeVec: vector (size n x 1) of time stamps of the segment relative to the event around which segmentation
%          was conducted

conditionTriggerSamps=find(ismember(triggerVec,triggerCode)); % find time stamps of events
nTrials=length(conditionTriggerSamps);
segmentBorderSamples=round(sampRate*segBoundsTime/1000); % calculate time stamps in each segment
currentSegment=segmentBorderSamples(1):segmentBorderSamples(2);

segmentedData=zeros(length(currentSegment), size(contEEGdata,2), nTrials);
isNotArtifact=zeros(nTrials,1);

for tr=1:nTrials
    segmentedData(:,:,tr)=contEEGdata(conditionTriggerSamps(tr)+currentSegment, :);
    isNotArtifact(tr)=sum(artifactVec(conditionTriggerSamps(tr)+currentSegment))==0;
end

timeVec=linspace(segBoundsTime(1), segBoundsTime(2), size(segmentedData,1)); % create segment time stamp vector
