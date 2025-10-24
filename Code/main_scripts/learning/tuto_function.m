function [data_out] = tuto_function(data_in)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
meano = mean(data_in)
stdo = std(data_in)
data_out = (data_in - 30*meano)/stdo;



end