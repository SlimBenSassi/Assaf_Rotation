function [fValues, dfValues, pValues]=mixed2WayANOVA(dataWithLabels)

% mixed2WayANOVA - Two-way mixed ANOVA, for designs with one 'between-' factor and one 'within-' factor
% (each level of the random factor, e.g. subjects or electrodes, is measured within each level of the within factor,
%   but only within one level of the between factor).
%
% Inputs:
% dataWithLabels - data effects with subjects as rows, first column:conditions labels, columns 2 to end: levels of within-subject factor
%
% Outputs: 
% fValues - f values of the effects of A ('between' factor), B ('within' factor), and AB (interaction)
% dfValues - same for df
% pValues - same for p


dataWithLabels=sortrows(dataWithLabels,1);

% get data parameters
labels=dataWithLabels(:,1);
data=dataWithLabels(:,2:end);
levelsA=unique(labels);
nlevelsA=length(levelsA);
nlevelsB=size(data,2);
nSubjectsTotal=size(data,1);

% check equal Ns
withinCellN=grpstats(dataWithLabels(:,1), dataWithLabels(:,1), 'numel');
if length(unique(withinCellN))>1
    warning('data is unbalanced, using type II sum-of-squares')
end

%overall mu
overallMean=mean(mean(data));
muMatrix=overallMean*ones(size(data));

% effect of A
Aeffects=zeros(size(data));
for i=1:nlevelsA    
    Aeffects(labels==levelsA(i),:)=mean(mean(data(labels==levelsA(i),:)))-overallMean;    
end
dfA=nlevelsA-1;
msA=sum(sum(Aeffects.^2))/dfA;

% effect of B
Beffects=repmat(mean(data), nSubjectsTotal,1)-overallMean;
dfB=nlevelsB-1;
msB=sum(sum(Beffects.^2))/dfB;

% effect of S(A)
Seffects=repmat(mean(data,2),1,size(data,2))-(muMatrix+Aeffects);
dfS=nSubjectsTotal-nlevelsA;
msS=sum(sum(Seffects.^2))/dfS;

%effect of AB
ABmeansMat=zeros(size(data));
for i=1:nlevelsA
    for j=1:nlevelsB        
        ABmeansMat(labels==levelsA(i),j)=mean(data(labels==levelsA(i),j));
    end
end
ABeffects=ABmeansMat-(muMatrix+Aeffects+Beffects);
dfAB=(nlevelsA-1)*(nlevelsB-1);
msAB=sum(sum(ABeffects.^2))/dfAB;

%effect of BS(A)
BSeffects=data-(muMatrix+Aeffects+Beffects+Seffects+ABeffects);
dfBS=(nSubjectsTotal-nlevelsA)*(nlevelsB-1);
msBS=sum(sum(BSeffects.^2))/dfBS;

% outputs
%df
dfValues.A_between=[dfA,dfS];
dfValues.B_within=[dfB,dfBS];
dfValues.AB_int=[dfAB,dfBS];

% f and p
undefFs=0;
if msS==0
    fValues.A_between=NaN;
    pValues.A_between=NaN;
    undefFs=1;
else
    fValues.A_between=msA/msS;
    pValues.A_between=cdf('F',fValues.A_between,dfValues.A_between(1), dfValues.A_between(2), 'upper');
end

if msBS==0
    fValues.B_within=NaN;
    fValues.AB_int=NaN;
    pValues.B_within=NaN;
    pValues.AB_int=NaN;
    undefFs=1;
else
    fValues.B_within=msB/msBS;
    pValues.B_within=cdf('F',fValues.B_within,dfValues.B_within(1), dfValues.B_within(2), 'upper');
    fValues.AB_int=msAB/msBS;
    pValues.AB_int=cdf('F',fValues.AB_int,dfValues.AB_int(1), dfValues.AB_int(2), 'upper');
end

if undefFs    
    disp('no error variance in one of the effects')
end
