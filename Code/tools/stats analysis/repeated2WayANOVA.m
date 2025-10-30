function [fValues, dfValues, pValues]=repeated2WayANOVA(data, levelsA, levelsB)

% repeated2WayANOVA - Two-way repeated measures ANOVA, for designs with two 'within-' fixed factors
% (each level of the random factor, e.g. subjects or electrodes, is measured within each combination of the fixed factors).
%
% Inputs:
% data - data matrix with subjects as rows, levels of within-subject factors in columns, sorted ascending
% levelsA, levelsB - number of levels in each WS factor, A being higher in hierarchy than B
%   (e.g. for a 3x2 orthogonal design, data columns should be: A1B1 A1B2 A2B1 A2B2 A3B1 A3B2)
%
% Outputs:
% fValues - f values of the effects of A, B and the AB interaction
% dfValues - same for df
% pValues - same for p


%check design
nSubjects=size(data,1);
if levelsA*levelsB~=size(data,2)
    error('number of levels in data set does not match requested design')
end

%overall mu
overallMean=mean(mean(data));
muMatrix=overallMean*ones(size(data));

% effect of A
Aeffects=zeros(size(data));
for i=1:levelsA    
    Aeffects(:,(i-1)*levelsB+1:i*levelsB)=mean(mean(data(:,(i-1)*levelsB+1:i*levelsB)))-overallMean;    
end
dfA=levelsA-1;
msA=sum(sum(Aeffects.^2))/dfA;

% effect of B
Beffects=zeros(size(data));
for i=1:levelsB    
    Beffects(:,i:levelsB:levelsB*levelsA)=mean(mean(data(:,i:levelsB:levelsB*levelsA)))-overallMean;    
end
dfB=levelsB-1;
msB=sum(sum(Beffects.^2))/dfB;

% effect of S (ms is not needed in this design)
Seffects=repmat(mean(data,2),1,size(data,2))-overallMean;

%effect of AB
ABeffects=repmat(mean(data,1),nSubjects,1)-(muMatrix+Aeffects+Beffects);
dfAB=(levelsA-1)*(levelsB-1);
msAB=sum(sum(ABeffects.^2))/dfAB;

% effect of AS
AcrossBeffects=zeros(size(data));
for i=1:levelsA    
    AcrossBeffects(:,(i-1)*levelsB+1:i*levelsB)=repmat(mean(data(:,(i-1)*levelsB+1:i*levelsB),2),1,levelsB);    
end
ASeffects=AcrossBeffects-(muMatrix+Aeffects+Seffects);
dfAS=(levelsA-1)*(nSubjects-1);
msAS=sum(sum(ASeffects.^2))/dfAS;

% effect of BS
AcrossAeffects=zeros(size(data));
for i=1:levelsB    
    AcrossAeffects(:,i:levelsB:levelsB*levelsA)=repmat(mean(data(:,i:levelsB:levelsB*levelsA),2),1,levelsA);    
end
BSeffects=AcrossAeffects-(muMatrix+Beffects+Seffects);
dfBS=(levelsB-1)*(nSubjects-1);
msBS=sum(sum(BSeffects.^2))/dfBS;

%effect of ABS
ABSeffects=data-(muMatrix+Aeffects+Beffects+Seffects+ABeffects+ASeffects+BSeffects);
dfABS=(levelsA-1)*(levelsB-1)*(nSubjects-1);
msABS=sum(sum(ABSeffects.^2))/dfABS;

% outputs
dfValues.A=[dfA,dfAS]; % A tested against variance of A effect within subjects
dfValues.B=[dfB,dfBS]; % B tested against variance of B effect within subjects
dfValues.AB=[dfAB,dfABS]; % AB tested against variance of AB effect within subjects

% f values
undefFs=0;
if msAS==0
    fValues.A=NaN;
    pValues.A=NaN;
    undefFs=1;
else
    fValues.A=msA/msAS;
    pValues.A=cdf('F',fValues.A,dfValues.A(1), dfValues.A(2), 'upper');
end

if msBS==0
    fValues.B=NaN;
    pValues.B=NaN;
    undefFs=1;
else
    fValues.B=msB/msBS;
    pValues.B=cdf('F',fValues.B,dfValues.B(1), dfValues.B(2), 'upper');    
end

if msABS==0
    fValues.AB=NaN;
    pValues.AB=NaN;
    undefFs=1;
else
    fValues.AB=msAB/msABS;
    pValues.AB=cdf('F',fValues.AB,dfValues.AB(1), dfValues.AB(2), 'upper');
end

if undefFs    
    disp('no error variance in one of the effects')
end

