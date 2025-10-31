function out = rdi(data, varargin)
%RDI: % raw data inspection for automatic artifact identification
%
% INPUT:
% data - a time X channel data matrix.
%
% OUTPUT:
% out - a time X channel binary matrix, with 1 in time points marked as artifact in individual channels, 0 otherwise.
%
% OPTIONAL INPUTS:
% out = RDI(...,'channels', c,...):                     Selects which channels should be inspected, where c is a vector with
%                                                       channel numbers
% out = RDI(...,'gradient', [mvs, mb, ma],...):         Include bad gradient in artifact search. A timepoint is marked as artifact
%                                                       if the absolute voltage difference from the preceding timepoint is larger
%                                                       than mvs, with mb and ma being the margins to mark before and after the
%                                                       artifact, respectively. Alternative syntax: RDI(...,'g',[mvs, mb, ma],...).
% out = RDI(...,'minmax', [mxad, il, mb, ma],...):      Include bad minmax in artifact search. A timepoint is marked as artifact
%                                                       if an absolute voltage difference of more than mxad is found in an interval
%                                                       of length il, with mb and ma being the margins to mark before and after the
%                                                       artifact, respectively. Alternative syntax: RDI(...,'m',[mxad, il, mb, ma],...).
% out = RDI(...,'lowactivity', [mnad, il, mb, ma],...): Include low activity in artifact search. A timepoint is marked as artifact
%                                                       if an absolute voltage difference of less than mnad is found in an interval
%                                                       of length il, with mb and ma being the margins to mark before and after the
%                                                       artifact, respectively. Alternative syntax: RDI(...,'l',[mnad, il, mb, ma],...).
% out = RDI(...,'extreme', [mna, mxa, mb, ma],...):     Include extreme amplitude in artifact search. A timepoint is marked as artifact
%                                                       if the absolute voltage is less than mna or more than mxa, with mb and ma being
%                                                       the margins to mark before and after the artifact, respectively.
%                                                       Alternative syntax: RDI(...,'e',[mna, mxa, mb, ma],...).

%% HANDLE INPUT
% check if input data is transposed:
if size(data,1) < size(data,2)
    warning('rdi: Input data matrix might be transposed. Expected matrix is time x channels');
end
% initialize channel list
channels = 1: size(data,2);
% gradient test
CheckGradient = false;
Gradient_MaximalVoltageStep = 100;
Gradient_EventMargins = [150 150];
% min max test
CheckMinMax = false;
MinMax_MaxAbsoluteDifference = 100;
MinMax_IntervalLength = 100;
MinMax_EventMargins = [150 150];
% low activity test
CheckLowActivity = false;
LowAct_LowestAllowed = 0.5;
LowAct_IntervalLength = 100;
LowAct_EventMargins = [150 150];
% extreme value test
CheckExtremeValue = false;
Ext_Minimal = -100;
Ext_Maximal = 100;
Ext_EventMargins = [150 150];

% parse optional input arguments
narg = size(varargin,2);
arg = 1;
while arg <= narg
    switch lower(varargin{arg})
        case 'channels'
            % check that the next argument exists, and that it is a vector
            % of natural numbers
            if narg>arg && isvector(varargin{arg+1}) && all(round(varargin{arg+1})==varargin{arg+1})
                channels = varargin{arg+1};
                arg = arg + 2;
            else
                error('rdi: ''channels'' expected to be followed by vector');
            end
        case {'gradient','g'}
            CheckGradient = true;
            arg = arg + 1;
            if narg>=arg && isnumeric(varargin{arg})
                if length(varargin{arg}) == 3
                    Gradient_MaximalVoltageStep = varargin{arg}(1);
                    Gradient_EventMargins(1) = varargin{arg}(2);
                    Gradient_EventMargins(2) = varargin{arg}(3);
                    arg = arg + 1;
                else
                    error('rdi: Vector following ''gradient'' argument should have three values (maximal voltage step, margin before event, margin after event)');
                end
            end
        case {'minmax','m'}
            CheckMinMax = true;
            arg = arg + 1;
            if narg>=arg && isnumeric(varargin{arg})
                if length(varargin{arg}) == 4
                    MinMax_MaxAbsoluteDifference = varargin{arg}(1);
                    MinMax_IntervalLength = varargin{arg}(2);
                    MinMax_EventMargins(1) = varargin{arg}(3);
                    MinMax_EventMargins(2) = varargin{arg}(4);
                    arg = arg + 1;
                else
                    error('rdi: Vector following ''minmax'' argument should have four values (max difference, interval length, margin before event, margin after event)');
                end
            end
        case {'lowactivity','l'}
            CheckLowActivity = true;
            arg = arg + 1;
            if narg>=arg && isnumeric(varargin{arg})
                if length(varargin{arg}) == 4
                    LowAct_LowestAllowed = varargin{arg}(1);
                    LowAct_IntervalLength = varargin{arg}(2);
                    LowAct_EventMargins(1) = varargin{arg}(3);
                    LowAct_EventMargins(2) = varargin{arg}(4);
                    arg = arg + 1;
                else
                    error('rdi: Vector following ''lowactivity'' argument should have four values (minimal amplitude difference, interval length, margin before event, margin after event)');
                end
            end
        case {'extreme','e'}
            CheckExtremeValue = true;
            arg = arg + 1;
            if narg>=arg && isnumeric(varargin{arg})
                if length(varargin{arg}) == 4
                    Ext_Minimal = varargin{arg}(1);
                    Ext_Maximal = varargin{arg}(2);
                    Ext_EventMargins(1) = varargin{arg}(3);
                    Ext_EventMargins(2) = varargin{arg}(4);
                    arg = arg + 1;
                else
                    error('rdi: Vector following ''extreme'' argument should have four values (minimal extreme value, maximal extreme value, margin before event, margin after event)');
                end
            end        
        otherwise
            error(['rdi: Unexpected input argument: ' num2str(varargin{arg})]);
    end
    
end

%% Initialize
[nT, nCh_all] = size(data);
nCh = length(channels);
data = data(:,channels);
out = zeros(nT,nCh_all);

%% Check Gradient
if CheckGradient
    mxStep = Gradient_MaximalVoltageStep;
    margins = Gradient_EventMargins;
    
    D = [zeros(1,nCh) ; diff(data)];
    gradientIdx = D > abs(mxStep);
    gradientIdx = conv2(single(gradientIdx),ones(sum(margins),1));
    gradientIdx = gradientIdx(margins(1)+1:nT+margins(1),:);
    gradientIdx = logical(gradientIdx);
    
    out(:,channels) = out(:,channels) | gradientIdx;
end

%% Check sliding window MinMax difference and low activity
minMaxIdx = zeros(size(data));
lowActIdx = zeros(size(data));
if CheckMinMax || CheckLowActivity
    
    mxDiff = MinMax_MaxAbsoluteDifference;
    mmWinLen = MinMax_IntervalLength;
    mmMargins = MinMax_EventMargins;
    
    mnDiff = LowAct_LowestAllowed;
    laWinLen = LowAct_IntervalLength;
    laMargins = LowAct_EventMargins;
    
    for i = 1:(nT-mmWinLen+1)
        [mn, mn_idx] = min(data(i:i+mmWinLen-1,:));
        [mx, mx_idx] = max(data(i:i+mmWinLen-1,:));
        if CheckMinMax
        mmBadWin = find(mx-mn>mxDiff);
            for j = 1:length(mmBadWin)
                a = min(mn_idx(mmBadWin(j)),mx_idx(mmBadWin(j)));
                b = max(mn_idx(mmBadWin(j)),mx_idx(mmBadWin(j)));
                minMaxIdx(i+a-1:i+b-1,mmBadWin(j)) = 1;
            end
        end
        if CheckLowActivity
            laBadWin = find(mx-mn<mnDiff);
            for j = 1:length(laBadWin)
                a = min(mn_idx(laBadWin(j)),mx_idx(laBadWin(j)));
                b = max(mn_idx(laBadWin(j)),mx_idx(laBadWin(j)));
                lowActIdx(i+a-1:i+b-1,laBadWin(j)) = 1;
            end
        end
    end
    
    minMaxIdx = conv2(single(minMaxIdx),ones(sum(mmMargins),1));
    minMaxIdx = minMaxIdx(mmMargins(1)+1:nT+mmMargins(1),:);
    minMaxIdx = logical(minMaxIdx);
    lowActIdx = conv2(single(lowActIdx),ones(sum(laMargins),1));
    lowActIdx = lowActIdx(laMargins(1)+1:nT+laMargins(1),:);
    lowActIdx = logical(lowActIdx);
    
    if CheckMinMax
        out(:,channels) = out(:,channels) | minMaxIdx;
    end
    if CheckLowActivity
        out(:,channels) = out(:,channels) | lowActIdx;
    end
end


%% Check sliding window MinMax difference and low activity
minMaxIdx = zeros(size(data));
lowActIdx = zeros(size(data));
if CheckMinMax || CheckLowActivity
    
    mxDiff = MinMax_MaxAbsoluteDifference;
    mmWinLen = MinMax_IntervalLength;
    mmMargins = MinMax_EventMargins;
    
    mnDiff = LowAct_LowestAllowed;
    laWinLen = LowAct_IntervalLength;
    laMargins = LowAct_EventMargins;
    
    lastWindInd=nT-min(mmWinLen, laWinLen)+1;
    checkMinMaxTimepoints=ones(size(data,1),1); checkMinMaxTimepoints(nT-mmWinLen+1:end)=0;
    checkLowActTimepoints=ones(size(data,1),1); checkLowActTimepoints(nT-laWinLen+1:end)=0;
    
    for i = 1:lastWindInd
        if CheckMinMax && checkMinMaxTimepoints(i)
            [mn, mn_idx] = min(data(i:i+mmWinLen-1,:));
            [mx, mx_idx] = max(data(i:i+mmWinLen-1,:));
            mmBadWin = find(mx-mn>mxDiff);
            for j = 1:length(mmBadWin)
                a = min(mn_idx(mmBadWin(j)),mx_idx(mmBadWin(j)));
                b = max(mn_idx(mmBadWin(j)),mx_idx(mmBadWin(j)));
                minMaxIdx(i+a-1:i+b-1,mmBadWin(j)) = 1;
            end
        end
        if CheckLowActivity && checkLowActTimepoints
            [mn, mn_idx] = min(data(i:i+laWinLen-1,:));
            [mx, mx_idx] = max(data(i:i+laWinLen-1,:));
            laBadWin = find(mx-mn<mnDiff);
            for j = 1:length(laBadWin)
                a = min(mn_idx(laBadWin(j)),mx_idx(laBadWin(j)));
                b = max(mn_idx(laBadWin(j)),mx_idx(laBadWin(j)));
                lowActIdx(i+a-1:i+b-1,laBadWin(j)) = 1;
            end
        end
    end
    
    minMaxIdx = conv2(single(minMaxIdx),ones(sum(mmMargins),1));
    minMaxIdx = minMaxIdx(mmMargins(1)+1:nT+mmMargins(1),:);
    minMaxIdx = logical(minMaxIdx);
    lowActIdx = conv2(single(lowActIdx),ones(sum(laMargins),1));
    lowActIdx = lowActIdx(laMargins(1)+1:nT+laMargins(1),:);
    lowActIdx = logical(lowActIdx);
    
    if CheckMinMax
        out(:,channels) = out(:,channels) | minMaxIdx;
    end
    if CheckLowActivity
        out(:,channels) = out(:,channels) | lowActIdx;
    end
end



%% Check extreme values
if CheckExtremeValue
    mnVal = Ext_Minimal;
    mxVal = Ext_Maximal;
    margins = Ext_EventMargins;
    
    extremeIdx = data > mxVal | data < mnVal;
    extremeIdx = conv2(single(extremeIdx),ones(sum(margins),1));
    extremeIdx = extremeIdx(margins(1)+1:nT+margins(1),:);
    extremeIdx = logical(extremeIdx);
    
    out(:,channels) = out(:,channels) | extremeIdx;
end

end