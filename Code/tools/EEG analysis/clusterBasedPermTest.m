function [clustersTrue, trueT_P, maxSumPermDistribution]=clusterBasedPermTest(data1,data2, testType, criterion, nPerms)

% clusterBasedPermTest - identify clusters of consecutive timepoints in a
% timeseries that differ between two conditions over random samples of the
% timeseries (e.g. subjects, electrodes...), with statistical inference
% done on cluster level instead of on individual time points.
% Based on comparing cluster-level tSum statistic to a shuffling-based null
% distribution of maxSum values (addresses the multiple comparison problem
% and also posthoc testing). See Maris & Oostenveld, 2007.
%
% Inputs -
% data1, data2: sN X tN data matrices, sN = levels of random factor
%   (e.g. number of subjects), tN = time samples. tN must be identical
%   between two matrices, sN must be identical for within-subject paired test
%   but can be different for between-subject independent test.
% testType: 1=two datasets are measured for the same units of random factor
%   ("within-subject" paired test), 2=two datasets are measured for different
%   levels of the random factor ("between-subject" independent test).
% criterion: p-value criterion for cluster identification. Lower values
%   prioritize short clusters with a large difference over long clusters of
%   smaller differences. Default 0.05.
% nPerms: number of permutations. Default 10000.
% 
% Outputs -
% clusters: a clusterNumber X 4 matrix with true detected clusters (in non-shuffled
%   data). columns: 1: first sample of cluster; 2: number of samples in
%   cluster; 3: tSum statistic of cluster; 4: p-value of the cluster relative
%   to the permutation distribution.
% trueT_P: a 2 X tN matrix with point by point t and p values (rows 1 and 2).
% maxSumPermDistribution: the constructed null distribution of maxTsum values.
%
% Notes
% 1: The reported cluster-level p-value is two tailed (divide by 2 for one-tailed test)
% 2: The function uses 'parfor' from the parallel toolbox, if not available change to 'for'.
%
% assaf.breska@gmail.com


% assign default variables
if nargin<5
    nPerms=10000;
end
if nargin<4
    criterion=0.05;
end


rng('default') % initialize rng with default for replicability, otherwise change to rng('shuffle')


% step 1: find cluster in true data
% 1.1: run ttest on true data
if testType==1
    if size(data1,1)~=size(data2,1)
        error('data mats must have same number of rows for within-subject test')
    else
        [~,p,~,statsTrue]=ttest(data1-data2); % one group, "within-subject" test
        tThreshold=abs(tinv(criterion/2,size(data1,1)-1));
    end
else
    [~,p,~,statsTrue]=ttest2(data1,data2); % two groups, "between-subject" test
    tThreshold=abs(tinv(criterion/2,size(data1,1)+size(data2,1)-2));
end

% 1.2: get clusters
clustersTrue=findClust(abs(statsTrue.tstat)>tThreshold);
clustersTrue=[clustersTrue, zeros(size(clustersTrue,1),1)];
for i=1:size(clustersTrue,1)
    clustersTrue(i,3)=sum(abs(statsTrue.tstat(clustersTrue(i,1):(clustersTrue(i,1)+(clustersTrue(i,2)-1)))));
end

trueT_P=[statsTrue.tstat; p]; % for output


% step 2: generate shuffling data and permutation distribution
% 2.1: create permutation matrix, while making sure none is repeated (if possible given system memory limits)
if testType==1
    nPossiblePerms=2^(size(data1,1));
    if nPossiblePerms>100*nPerms || size(data1,1)>24 % create random perms if sufficient perms exist or creating all exceeds memory limits
        permsToUse=round(rand(size(data1,1),nPerms));
    else
        possiblePerms=(dec2bin(0:(nPossiblePerms-1))-'0')'; % create all possible perms
        if nPossiblePerms<=nPerms
            disp(['requested perm num is larger than max possible perms, ' num2str(size(possiblePerms,2)) ' perms used instead of ' num2str(nPerms)])
            permsToUse=possiblePerms;
            nPerms=nPossiblePerms;
        else
            shuffledPerms=possiblePerms(:,randperm(size(possiblePerms,2))); 
            permsToUse=shuffledPerms(:,1:nPerms);
            clear shuffledPerms            
        end        
    end
    permsToUse(permsToUse==0)=-1;
    diffData=data1-data2;
else
    nPossiblePerms=factorial(size(data1,1)+size(data2,1))/(factorial(size(data1,1))*factorial(size(data2,1)));
    if nPossiblePerms>100*nPerms || size(data1,1)+size(data2,1)>24 % create random perms if sufficient perms exist or creating all exceeds memory limits
        possiblePerms=zeros(size(data1,1)+size(data2,1), nPerms);
        parfor p=1:nPerms
            possiblePerms(:,p)=randperm(size(data1,1)+size(data2,1))';                        
        end
        permsToUse=double(possiblePerms<=size(data1,1));
    else
        possiblePerms=(dec2bin(0:(2^(size(data1,1)+size(data2,1))-1))-'0')'; % create all possible perms
        possiblePerms(:,sum(possiblePerms)~=size(data1,1))=[];
        if nPossiblePerms<=nPerms
            disp(['requested perm num is larger than max possible perms, ' num2str(size(possiblePerms,2)) ' perms used instead of ' num2str(nPerms)])
            permsToUse=possiblePerms;
            nPerms=nPossiblePerms;
        else
            shuffledPerms=possiblePerms(:,randperm(size(possiblePerms,2)));
            permsToUse=shuffledPerms(:,1:nPerms);
            clear shuffledPerms            
        end
    end
end


% 2.2: create shuffled data and run ttest, depending on test type
% (separated to reduce memory load on parallel workers)
maxSumPermDistribution=zeros(nPerms,1);
if testType==1 % within
    parfor perm=1:nPerms
        tVec=abs(simpleTTestWithin(diffData.*repmat(permsToUse(:,perm), 1, size(data1,2)))); % flip sign of randomly selected data units
        clustersPerm=findClust(tVec>tThreshold);
        if isempty(clustersPerm)
            maxSumPermDistribution(perm)=0; % for permutations with no significant cluster
        else
            clustersPerm=[clustersPerm, zeros(size(clustersPerm,1),1)];
            for i=1:size(clustersPerm,1)
                clustersPerm(i,3)=sum(tVec(clustersPerm(i,1):(clustersPerm(i,1)+(clustersPerm(i,2)-1))));
            end
            maxSumPermDistribution(perm)=max(clustersPerm(:,3)); % save max cluster, to address posthoc/multiple comparison problem
        end
    end
else % between
    parfor perm=1:nPerms
        allData=[data1; data2]; %sending data1 and data2 to parallel worker instead of alldata twice
        tVec=abs(simpleTTestBetween(allData(permsToUse(:,perm)==1,:), allData(permsToUse(:,perm)==0,:))); % divide into two randomly allocated groups
        clustersPerm=findClust(tVec>tThreshold);
        if isempty(clustersPerm)
            maxSumPermDistribution(perm)=0; % for permutations with no significant cluster
        else
            clustersPerm=[clustersPerm, zeros(size(clustersPerm,1),1)];
            for i=1:size(clustersPerm,1)
                clustersPerm(i,3)=sum(tVec(clustersPerm(i,1):(clustersPerm(i,1)+(clustersPerm(i,2)-1))));
            end
            maxSumPermDistribution(perm)=max(clustersPerm(:,3)); % save max cluster, to address posthoc/multiple comparison problem
        end
    end
end


% step 3: compare true data to permutation distribution
clustersTrue=[clustersTrue, zeros(size(clustersTrue,1),1)];
for i=1:size(clustersTrue,1)
    clustersTrue(i,4)=sum(maxSumPermDistribution>clustersTrue(i,3))/nPerms; % calculate two-tailed p-value of true clusters
end

end


% auxiliary functions
function clusterInfo=findClust(inpVec)

clusterStart=find(diff([0 inpVec])==1)';
clusterEnd=find(diff([inpVec 0])==-1)';
clusterStart=clusterStart(1:length(clusterEnd));

if isempty(clusterStart)
    clusterInfo=[];
else
    clusterInfo=[clusterStart, clusterEnd-clusterStart+1];
    clusterInfo=clusterInfo(clusterInfo(:,2)>1,:); % cluster length must be more than 1 time point to be considered a cluster
end

end


function t=simpleTTestWithin(data) % runs faster than ttest - does not compute p-val

t=mean(data)*sqrt(size(data,1))./std(data);

end


function t=simpleTTestBetween(data1,data2) % runs faster than ttest2 - does not compute p-val

n1=size(data1,1);
n2=size(data2,1);

pooledSE=sqrt(((1/n1)+(1/n2))*((n1-1)*var(data1,[],1)+(n2-1)*var(data2,[],1))/(n1+n2-2));
t=(mean(data1)-mean(data2))./pooledSE;

end

