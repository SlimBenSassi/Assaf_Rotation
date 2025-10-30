function bcData=baselineCorrectSegmentedDataDiv(data, timeVec, baselineRange)

bcData=bsxfun(@rdivide, data, mean(data(timeVec>=baselineRange(1) & timeVec<=baselineRange(2),:)));

