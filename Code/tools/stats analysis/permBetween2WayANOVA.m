function [fValuesTrue, pValues]=permBetween2WayANOVA(dataWithLabels, nPerms)

% permBetween2WayANOVA - permutation test for two-way between-subjects ANOVA designs.
% Approximate tests for main effects and interaction (conservative)
%
% Inputs:
% data - data matrix with subjects as rows, levels of between-subject factors in first two columns, dependent variable in third column.
% nPerms - number of permutations
%
% Outputs:
% fValuesTrue - f values of the effects of A, B and the interaction from original ANOVA (should not be reported)
% pValues - p values from permutation test (should be reported)
%
% Based on Anderson & TerBraak (2003) permutation tests for multi-factorial analysis of variance
% Created by Assaf Breska, assaf.breska@gmail.com

if size(dataWithLabels,2)~=3
    error('dataset should have two columns with factor levels and column 3 with dependent variable values')
end

rng('shuffle')

% get true f values
[fValuesTrue, effectMatricesTrue]=between2WayANOVA_forPerm(dataWithLabels);

% approximate test for effect of B (within-subjects factor) - permutation of residuals on reduced model
reducedModelA=effectMatricesTrue.mu+effectMatricesTrue.B+effectMatricesTrue.AB;
reducedResidualsA=dataWithLabels(:,end)-reducedModelA;

permDistA=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(reducedResidualsA)),size(reducedResidualsA,1),size(reducedResidualsA,2));
    mixedReducedResidualsA=reducedResidualsA(mixingMatrix);
    fValues=between2WayANOVA_forPerm([dataWithLabels(:,[1 2]) mixedReducedResidualsA]);
    permDistA(perm)=fValues.A;    
end
pAapprox=sum(permDistA>fValuesTrue.A)/nPerms;

% approximate test for effect of B (within-subjects factor) - permutation of residuals on reduced model
reducedModelB=effectMatricesTrue.mu+effectMatricesTrue.A+effectMatricesTrue.AB;
reducedResidualsB=dataWithLabels(:,end)-reducedModelB;

permDistB=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(reducedResidualsB)),size(reducedResidualsB,1),size(reducedResidualsB,2));
    mixedReducedResidualsB=reducedResidualsB(mixingMatrix);
    fValues=between2WayANOVA_forPerm([dataWithLabels(:,[1 2]) mixedReducedResidualsB]);
    permDistB(perm)=fValues.B;    
end
pBapprox=sum(permDistB>fValuesTrue.B)/nPerms;

% approximate test for interaction - permutation of residuals on reduced model
reducedModelAB=effectMatricesTrue.mu+effectMatricesTrue.A+effectMatricesTrue.B;
reducedResidualsAB=dataWithLabels(:,end)-reducedModelAB;

permDistAB=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(reducedResidualsAB)),size(reducedResidualsAB,1),size(reducedResidualsAB,2));
    mixedReducedResidualsAB=reducedResidualsAB(mixingMatrix);
    fValues=between2WayANOVA_forPerm([dataWithLabels(:,[1 2]) mixedReducedResidualsAB]);
    permDistAB(perm)=fValues.AB;    
end
pABapprox=sum(permDistAB>fValuesTrue.AB)/nPerms;

pValues=[pAapprox, pBapprox, pABapprox];

end

% % %

function [fValues, effectMatrices]=between2WayANOVA_forPerm(dataWithLabels)

dataWithLabels=sortrows(dataWithLabels,[1 2]);

labelsA=unique(dataWithLabels(:,1));
labelsB=unique(dataWithLabels(:,2));
nlabelsA=length(labelsA);
nlabelsB=length(labelsB);
nSubjectsTotal=size(dataWithLabels,1);

withinCellN=grpstats(dataWithLabels(:,3), dataWithLabels(:,[1 2]), 'numel');
if length(unique(withinCellN))>1
    error('dataset must have equal group sizes')
end

dvData=dataWithLabels(:,3);

%overall mu
overallMean=mean(dvData);
muMatrix=overallMean*ones(size(dvData));

% effect of A
Aeffects=zeros(size(dvData));
for i=1:nlabelsA    
    Aeffects(dataWithLabels(:,1)==labelsA(i))=mean(dvData(dataWithLabels(:,1)==labelsA(i)))-overallMean;    
end
dfA=nlabelsA-1;
msA=sum(sum(Aeffects.^2))/dfA;

% effect of B
Beffects=zeros(size(dvData));
for i=1:nlabelsB    
    Beffects(dataWithLabels(:,2)==labelsB(i))=mean(dvData(dataWithLabels(:,2)==labelsB(i)))-overallMean;    
end
dfB=nlabelsB-1;
msB=sum(sum(Beffects.^2))/dfB;

%effect of AB
ABeffects=zeros(size(dvData));
for i=1:nlabelsA
    for j=1:nlabelsB        
        ABeffects(dataWithLabels(:,1)==labelsA(i) & dataWithLabels(:,2)==labelsB(j))=mean(dvData(dataWithLabels(:,1)==labelsA(i) & dataWithLabels(:,2)==labelsB(j)));
    end
end
ABeffects=ABeffects-(muMatrix+Aeffects+Beffects);
dfAB=dfA*dfB;
msAB=sum(sum(ABeffects.^2))/dfAB;

%effect of s(AB)
Seffects=dvData-(muMatrix+Aeffects+Beffects+ABeffects);
dfS=nSubjectsTotal-(dfA*dfB);
msS=sum(sum(Seffects.^2))/dfS;

% outputs
if msS==0 % dont calculate Fs
    disp('no error variance');
    fValues.A=NaN;
    fValues.B=NaN;
    fValues.AB=NaN;
else
    fValues.A=msA/msS;
    fValues.B=msB/msS;
    fValues.AB=msAB/msS;
end

effectMatrices.mu=muMatrix;
effectMatrices.A=Aeffects;
effectMatrices.B=Beffects;
effectMatrices.AB=ABeffects;

end
