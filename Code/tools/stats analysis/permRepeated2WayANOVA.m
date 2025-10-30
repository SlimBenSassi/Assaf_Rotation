function [fValuesTrue, pValuesPerm]=permRepeated2WayANOVA(data, levelsA, levelsB, nPerms)

% permRepeated2WayANOVA - permutation test for two-way repeated-measures ANOVA designs. 
% Exact tests for main effect, approximate test for interaction
%
% Inputs:
% data - data matrix with subjects as rows, levels of within-subject factors in columns, sorted ascending
% levelsA, levelsB - number of levels in each WS factor, A being higher in hierarchy than B
%   (e.g. for a 3x2 orthogonal design, data columns should be: A1B1 A1B2 A2B1 A2B2 A3B1 A3B2)
% nPerms - number of permutations
% 
% Outputs:
% fValuesTrue - f values of the effects of A, B and the interaction from original ANOVA (should not be reported)
% pValues - p values from permutation test (should be reported)
%
% Based on Anderson & TerBraak (2003) permutation tests for multi-factorial analysis of variance
% Written by Assaf Breska, assaf.breska@gmail.com


%check design
if levelsA*levelsB~=size(data,2)
    error('number of levels in data set does not match requested design')
end

rng('shuffle')

% get true f vals (to copmare against perm distributions)
[fValuesTrue, effectMatricesTrue]=repeated2WayANOVA_forPerm(data, levelsA, levelsB);
nSubjects=size(data,1);


% exact test for effect of A - shuffle raw values of levels of A, restricted within levels of B
permDistA=zeros(nPerms,1);
for perm=1:nPerms    
    dataShuffledA=zeros(size(data));    
    for i=1:nSubjects            
        mix=randperm(levelsA);        
        for j=1:levelsA            
            dataShuffledA(i,(j-1)*levelsB+1:j*levelsB)=data(i,(mix(j)-1)*levelsB+1:mix(j)*levelsB);
        end        
    end    
    fValuesPerm=repeated2WayANOVA_forPerm(dataShuffledA, levelsA, levelsB);
    permDistA(perm)=fValuesPerm.A;    
end
pValuesPerm.A=sum(permDistA>fValuesTrue.A)/nPerms;

% exact test for effect of B - shuffle raw values of levels of B, restricted within levels of A
permDistB=zeros(nPerms,1);
for perm=1:nPerms    
    dataShuffledB=zeros(size(data));    
    for i=1:nSubjects            
        mix=randperm(levelsB);        
        for j=1:levelsB            
            dataShuffledB(i,j:levelsB:levelsB*levelsA)=data(i,mix(j):levelsB:levelsB*levelsA);
        end        
    end    
    fValuesPerm=repeated2WayANOVA_forPerm(dataShuffledB, levelsA, levelsB);
    permDistB(perm)=fValuesPerm.B;    
end
pValuesPerm.B=sum(permDistB>fValuesTrue.B)/nPerms;

% approximate test for interaction - non-restricted permutation of residuals from reduced model
reducedModelAB=effectMatricesTrue.mu+effectMatricesTrue.A+effectMatricesTrue.B+effectMatricesTrue.S+effectMatricesTrue.AS+effectMatricesTrue.BS;
residualsAB=data-reducedModelAB;
permDistAB=zeros(nPerms,1);
for perm=1:nPerms    
    mixingMatrix=reshape(randperm(numel(residualsAB)),nSubjects,levelsB*levelsA);
    mixedResidualsAB=residualsAB(mixingMatrix);
    fValuesPerm=repeated2WayANOVA_forPerm(mixedResidualsAB,levelsA, levelsB);
    permDistAB(perm)=fValuesPerm.AB;    
end
pValuesPerm.AB=sum(permDistAB>fValuesTrue.AB)/nPerms;

end

% % %

function [fValues, effectMatrices]=repeated2WayANOVA_forPerm(data,levelsA, levelsB)

nSubjects=size(data,1);

%overall mu
overallMean=mean(mean(data));
muMatrix=overallMean*ones(size(data));

% effect of A
Aeffects=zeros(size(data));
for i=1:levelsA    
    Aeffects(:,(i-1)*levelsB+1:i*levelsB)=mean(mean(data(:,(i-1)*levelsB+1:i*levelsB)))-overallMean;    
end
msA=sum(sum(Aeffects.^2))/(levelsA-1);

% effect of B
Beffects=zeros(size(data));
for i=1:levelsB    
    Beffects(:,i:levelsB:levelsB*levelsA)=mean(mean(data(:,i:levelsB:levelsB*levelsA)))-overallMean;    
end
msB=sum(sum(Beffects.^2))/(levelsB-1);

% effect of S (ms is not needed in this design)
Seffects=repmat(mean(data,2),1,size(data,2))-overallMean;

%effect of AB
ABeffects=repmat(mean(data,1),nSubjects,1)-(muMatrix+Aeffects+Beffects);
msAB=sum(sum(ABeffects.^2))/((levelsA-1)*(levelsB-1));

% effect of AS
AcrossBeffects=zeros(size(data));
for i=1:levelsA    
    AcrossBeffects(:,(i-1)*levelsB+1:i*levelsB)=repmat(mean(data(:,(i-1)*levelsB+1:i*levelsB),2),1,levelsB);    
end
ASeffects=AcrossBeffects-(muMatrix+Aeffects+Seffects);
msAS=sum(sum(ASeffects.^2))/((levelsA-1)*(nSubjects-1));

% effect of BS
AcrossAeffects=zeros(size(data));
for i=1:levelsB    
    AcrossAeffects(:,i:levelsB:levelsB*levelsA)=repmat(mean(data(:,i:levelsB:levelsB*levelsA),2),1,levelsA);    
end
BSeffects=AcrossAeffects-(muMatrix+Beffects+Seffects);
msBS=sum(sum(BSeffects.^2))/((levelsB-1)*(nSubjects-1));

%effect of ABS
ABSeffects=data-(muMatrix+Aeffects+Beffects+Seffects+ABeffects+ASeffects+BSeffects);
msABS=sum(sum(ABSeffects.^2))/((levelsA-1)*(levelsB-1)*(nSubjects-1));

% outputs
% f values
undefFs=0;
if msAS==0
    fValues.A=NaN;
    undefFs=1;
else
    fValues.A=msA/msAS;
end

if msBS==0
    fValues.B=NaN;
    undefFs=1;
else
    fValues.B=msB/msBS;
end

if msABS==0
    fValues.AB=NaN;
    undefFs=1;
else
    fValues.AB=msAB/msABS;
end

if undefFs    
    disp('no error variance in one of the effects')
end

% return effect matrices for interaction perm test
effectMatrices.mu=muMatrix;
effectMatrices.A=Aeffects;
effectMatrices.B=Beffects;
effectMatrices.S=Seffects;
effectMatrices.AS=ASeffects;
effectMatrices.BS=BSeffects;

end
