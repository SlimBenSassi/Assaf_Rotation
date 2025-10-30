function [fValues, dfValues, pValues]=between2WayANOVA(dataWithLabels)

% between2WayANOVA - Two-way factorial ANOVA, for designs with two 'between-' fixed factors
% (each level of the random factor, e.g. subjects or electrodes, is only measured within one combination of the fixed factors).
%
% Input:
% dataWithLabels - data matrix with subjects as rows, levels of between-subject factors in first two columns, dependent variable in third column.
%
% Outputs:
% fValues - f values of the effects of A, B and the interaction
% dfValues - same for df
% pValues - same for p


if size(dataWithLabels,2)~=3
    error('dataset should have two columns with factor levels and column 3 with DV values')
end
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

data=dataWithLabels(:,3);

%overall mu
overallMean=mean(data);
muMatrix=overallMean*ones(size(data));

% effect of A
Aeffects=zeros(size(data));
for i=1:nlabelsA    
    Aeffects(dataWithLabels(:,1)==labelsA(i))=mean(data(dataWithLabels(:,1)==labelsA(i)))-overallMean;    
end
dfA=nlabelsA-1;
msA=sum(sum(Aeffects.^2))/dfA;

% effect of B
Beffects=zeros(size(data));
for i=1:nlabelsB    
    Beffects(dataWithLabels(:,2)==labelsB(i))=mean(data(dataWithLabels(:,2)==labelsB(i)))-overallMean;    
end
dfB=nlabelsB-1;
msB=sum(sum(Beffects.^2))/dfB;

%effect of AB
ABeffects=zeros(size(data));
for i=1:nlabelsA
    for j=1:nlabelsB        
        ABeffects(dataWithLabels(:,1)==labelsA(i) & dataWithLabels(:,2)==labelsB(j))=mean(data(dataWithLabels(:,1)==labelsA(i) & dataWithLabels(:,2)==labelsB(j)));
    end
end
ABeffects=ABeffects-(muMatrix+Aeffects+Beffects);
dfAB=dfA*dfB;
msAB=sum(sum(ABeffects.^2))/dfAB;

%effect of s(AB)
Seffects=data-(muMatrix+Aeffects+Beffects+ABeffects);
dfS=nSubjectsTotal-(dfA*dfB);
msS=sum(sum(Seffects.^2))/dfS;

% outputs
dfValues.A=[dfA,dfS]; % A tested against variance of A effect within subjects
dfValues.B=[dfB,dfS]; % B tested against variance of B effect within subjects
dfValues.AB=[dfAB,dfS]; % AB tested against variance of AB effect within subjects


if msS==0 % dont calculate Fs
    disp('no error variance');
    fValues.A=NaN;
    fValues.B=NaN;
    fValues.AB=NaN;
else
    fValues.A=msA/msS;
    pValues.A=cdf('F',fValues.A, dfValues.A(1), dfValues.A(2), 'upper');
    fValues.B=msB/msS;
    pValues.B=cdf('F',fValues.B, dfValues.B(1), dfValues.B(2), 'upper');
    fValues.AB=msAB/msS;
    pValues.AB=cdf('F',fValues.AB, dfValues.AB(1), dfValues.AB(2), 'upper');
end

