function bcData=baselineCorrectSegmentedData(data, timeVec, baselineRange)

if length(size(data))==2
    bcData=bsxfun(@minus, data, mean(data(timeVec>=baselineRange(1) & timeVec<=baselineRange(2),:),1));
elseif length(size(data))==3
    bcData=bsxfun(@minus, data, mean(data(timeVec>=baselineRange(1) & timeVec<=baselineRange(2),:,:),1));
end

