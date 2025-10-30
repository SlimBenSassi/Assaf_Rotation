function [fValuesTrue, pValuesPerm]=permMixed2WayANOVA(dataWithLabels, nPerms)

% permRepeated2WayANOVA - permutation test for two-way mixed ANOVA designs.
% Exact tests for between-subject main effect (A), approximate tests for
% within-subject main effect (B) and interaction (AB)
%
% Inputs:
% dataWithLabels - data matrix with subjects as rows, first column: conditions labels, columns 2 to end: levels of within-subject factor in columns
% nPerms - number of permutations
% 
% Outputs:
% fValuesTrue - f values of the effects of A ('between' factor), B ('within' factor), and AB (interaction) from original ANOVA (should NOT be reported)
% pValuesPerm - p values from permutation test (should be reported)
%
% Based on Anderson & TerBraak (2003) permutation tests for multi-factorial analysis of variance
% Written by Assaf Breska, assaf.breska@gmail.com

% check equal Ns
withinCellN=grpstats(dataWithLabels(:,1), dataWithLabels(:,1), 'numel');
if length(unique(withinCellN))>1
    warning('data is unbalanced, using type II sum-of-squares')
end

rng('shuffle')

% get true f vals (to copmare against perm distributions)
[fValuesTrue, effectMatricesTrue]=mixed2WayANOVA_forPerm(dataWithLabels);


% exact test for effect of A (between-subjects factor) - shuffle raw values between levels of A
permDistA=zeros(nPerms,1);
for perm=1:nPerms    
    mix=randperm(size(dataWithLabels,1));  
    dataShuffledA=[dataWithLabels(:,1) dataWithLabels(mix, 2:end)];
    fValuesPerm=mixed2WayANOVA_forPerm(dataShuffledA);
    permDistA(perm)=fValuesPerm.A;    
end
pValuesPerm.A=sum(permDistA>fValuesTrue.A)/nPerms;

% approximate test for effect of B (within-subjects factor) - permutation of residuals on reduced model
reducedModelB=effectMatricesTrue.mu+effectMatricesTrue.A+effectMatricesTrue.AB;
reducedResidualsB=dataWithLabels(:,2:end)-reducedModelB;

permDistB=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(reducedResidualsB)),size(reducedResidualsB,1),size(reducedResidualsB,2));
    mixedReducedResidualsB=reducedResidualsB(mixingMatrix);
    fValuesPerm=mixed2WayANOVA_forPerm([dataWithLabels(:,1) mixedReducedResidualsB]);
    permDistB(perm)=fValuesPerm.B;    
end
pValuesPerm.B=sum(permDistB>fValuesTrue.B)/nPerms;

% approximate test for interaction - permutation of residuals on reduced model
reducedModelAB=effectMatricesTrue.mu+effectMatricesTrue.A+effectMatricesTrue.B;
reducedResidualsAB=dataWithLabels(:,2:end)-reducedModelAB;

permDistAB=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(reducedResidualsAB)),size(reducedResidualsAB,1),size(reducedResidualsAB,2));
    mixedReducedResidualsAB=reducedResidualsAB(mixingMatrix);
    fValuesPerm=mixed2WayANOVA_forPerm([dataWithLabels(:,1) mixedReducedResidualsAB]);
    permDistAB(perm)=fValuesPerm.AB;    
end
pValuesPerm.AB=sum(permDistAB>fValuesTrue.AB)/nPerms;

end

% % %

function [fValues, effectMatrices]=mixed2WayANOVA_forPerm(dataWithLabels)

dataWithLabels=sortrows(dataWithLabels,1);

% get data parameters
labels=dataWithLabels(:,1);
data=dataWithLabels(:,2:end);
levelsA=unique(labels);
nlevelsA=length(levelsA);
nlevelsB=size(data,2);
nSubjectsTotal=size(data,1);

%overall mu
overallMean=mean(mean(data));
muMatrix=overallMean*ones(size(data));

% effect of A
Aeffects=zeros(size(data));
for i=1:nlevelsA    
    Aeffects(labels==levelsA(i),:)=mean(mean(data(labels==levelsA(i),:)))-overallMean;    
end
msA=sum(sum(Aeffects.^2))/(nlevelsA-1);

% effect of B
Beffects=repmat(mean(data), nSubjectsTotal,1)-overallMean;
msB=sum(sum(Beffects.^2))/(nlevelsB-1);

% effect of S(A)
Seffects=repmat(mean(data,2),1,size(data,2))-(muMatrix+Aeffects);
msS=sum(sum(Seffects.^2))/(nSubjectsTotal-nlevelsA);

%effect of AB
ABmeansMat=zeros(size(data));
for i=1:nlevelsA
    for j=1:nlevelsB        
        ABmeansMat(labels==levelsA(i),j)=mean(data(labels==levelsA(i),j));
    end
end
ABeffects=ABmeansMat-(muMatrix+Aeffects+Beffects);
msAB=sum(sum(ABeffects.^2))/((nlevelsA-1)*(nlevelsB-1));

%effect of BS(A)
BSeffects=data-(muMatrix+Aeffects+Beffects+Seffects+ABeffects);
msBS=sum(sum(BSeffects.^2))/((nSubjectsTotal-nlevelsA)*(nlevelsB-1));

% outputs
undefFs=0;

% f values
if msS==0
    fValues.A=NaN;
    undefFs=1;
else
    fValues.A=msA/msS;
end

if msBS==0
    fValues.B=NaN;
    fValues.AB=NaN;
    undefFs=1;
else
    fValues.B=msB/msBS;
    fValues.AB=msAB/msBS;
end

if undefFs    
    disp('no error variance in one of the effects')
end

% return effect matrices, necessary for perm test
effectMatrices.mu=muMatrix;
effectMatrices.A=Aeffects;
effectMatrices.B=Beffects;
effectMatrices.AB=ABeffects;

end


