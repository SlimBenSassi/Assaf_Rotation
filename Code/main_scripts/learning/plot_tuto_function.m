figure;

subplot(1,2,1);
plot(time_vec, my_eeg(1, :), 'r-');
xlim([0 1]);
title("raw");

subplot(1,2,2);
plot(time_vec, tuto_function(my_eeg(1, :)), 'bl-');
xlim([0 1]);
title("z-normalized")
xlabel("seconds")
ylabel("microvolts")