function denoised = remove_line_noise(s,fline,fs,winw)
% remove_line_noise removes 50Hz/60Hz line noise from a signal s sampled in
% frequency fs (default 1024Hz) by filtering out windowed sine waves 
% of the line frequency fline (default 50Hz) and of its harmonics.
% s is of size Ns x Nc, with Ns time samples in Nc channels. 
% winw is the filter window width (default 1sec). The wider it is, the more
% the filter ignores transient fluctuations and looks for consistant
% oscillations. 
% If s is a matrix, remove_line_noise(s) treats the columns of s as vectors
% i.e: time points = rows, channels=columns


if ~exist('fline','var')
    fline = 50  ;   % Default line frequency [Hz]
end
if ~exist('fs','var')
    fs = 1024  ;    % Default sampling frequency [Hz]
end
if ~exist('winw','var')
    winw = 1  ;     % Default filter window width [sec]
end
Nwinsamp = floor(winw * fs);   % number of samples within filter window
if Nwinsamp < 2
    error(['Filter window must be at least two samples long. winw*fs = ' num2str(winw * fs) '.'])
end
[Ns,Nc] = size(s);
% Transpose if s is a single row channel:
if Ns == 1, s = s'; [Ns,Nc] = size(s); end

t = 1/fs:1/fs:winw;    % Time points of filter window
Nt = length(t);
% Shift window leftwards such that the zero time point is in the middle:
mid = floor(Nt/2);  % Index of the middle time point
t = t - t(mid); 
Nharm = floor(fs/fline/2); % # of line frequency harmonics
f = (1:Nharm)' * fline;       % Harmonic frequencies
Harms = cos(2*pi*f*t);      % Matrix of harmonic time courses
% Harm_norm = diag(Harms*Harms');
% Harms = diag(Harm_norm) \ Harms;    % Normalize harmonics
Harms = Harms / (Nt/2);
win = cos(2*pi*t/winw)+1;   % Raised cosine time window
noise_filt = sum(Harms,1) .* win;  % Noise filter
impulse = zeros(1,Nt);   % Unit filter (impulse)
impulse(mid) = 1;
denoise_filt = impulse - noise_filt;    % Denoising filter impulse response
denoised = zeros(Ns+Nt-1,Nc);
% Filter channel by channel:
for ch = 1:Nc
    denoised(:,ch) = conv(s(:,ch),denoise_filt);
end
% Crop the convolution margins
denoised = denoised(mid+(0:end-Nt),:);