function referencedData=referenceContEEGdata(EEGdata, ref_chans)

if size(EEGdata,1)>size(EEGdata,2)
    referencedData = bsxfun(@minus,EEGdata,mean(EEGdata(:,ref_chans),2));
else
    EEGdata=EEGdata';
    referencedData = (bsxfun(@minus,EEGdata,mean(EEGdata(:,ref_chans),2)))';
end
