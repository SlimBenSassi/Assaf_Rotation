toy = randn(4,10);
time_vector = 1:10;
channel_3 = toy(3, :);
min_value = min(channel_3);
title = "trial" + 10/10;
figure;
plot(time_vector,channel_3,'black-')
title(title);
xlabel('Time Points');
ylabel('Amplitude');