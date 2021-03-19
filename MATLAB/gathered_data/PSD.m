%% Collecting Data with Low-Pass Filter
% filename = 'C:\Users\mcgrathp\PycharmProjects\EEG-Decoder2\python\saved_data\train_02_10_21_3_fast_with_clench_blink.json';
% json = jsondecode(fileread(filename));
buffer = flip(json.raw_list{2}, 1);
finger_data = repelem(mean(json.finger_data(2, [2:5])), [size(buffer, 1)], [1]);
N = length(json.raw_list);

for i=3:N
   finger_data = [finger_data; repelem(mean(json.finger_data(i, [2:5])), [size(json.raw_list{i}, 1)], [1])];
   buffer = [buffer; flip(json.raw_list{i}, 1)]; 
end

% Scale buffer up by 100
buffer = buffer / 24;
buffer = buffer * 1000;

%% Classifying Finger Data
% Save classifiaction finger labels to .txt file
% The labels are supplied as follows:
%   0: Open hand label
%   1: Closing hand label
%   2: Closed hand label
%   3: Openning hand label
% Note that the threshold for open and closed hand readings are different.
% This is because it is easy to get closed hand reading while the open hand
% readings are a bit more volatile and must have a lower threshold (based
% on magnitude) as a result.

classification_finger_data = zeros(size(finger_data, 1), 1);
classification_finger_data(1) = 3;
for i=2:size(finger_data,1)
    average_hand_measurement = finger_data(i);
    % Closed hand has a label of 2
    if average_hand_measurement > .98
        classification_finger_data(i) = 2;
        % Open hand has a label of 0
    elseif average_hand_measurement < -.96
        classification_finger_data(i) = 0;
        % Closing hand has a label of 1
    elseif classification_finger_data(i-1) == 1 || classification_finger_data(i-1) == 0
        classification_finger_data(i) = 1;
        % Openning hand has a label of 3
    else
        classification_finger_data(i) = 3;
    end
end

%% Break Up Data For EEGLAB
event_matrix = cell(length(classification_finger_data), 3);
row_number = 1;
i = 1;
while (i <= length(classification_finger_data))
   if classification_finger_data(i) == 1
       data_duration = 1;
       for j=i+1:length(classification_finger_data)
           if classification_finger_data(j) == 1
               data_duration = data_duration + 1;
           else
              break 
           end
       end
       if data_duration > 50
           event_matrix{row_number, 1} = i;
           event_matrix{row_number, 2} = "RH_open";
           event_matrix{row_number, 3} = data_duration;
           row_number = row_number + 1;
           i = j;
       end
   elseif classification_finger_data(i) == 3
       data_duration = 1;
       for j=i+1:length(classification_finger_data)
           if classification_finger_data(j) == 3
               data_duration = data_duration + 1;
           else
              break 
           end
       end
       if data_duration > 50
           event_matrix{row_number, 1} = i;
           event_matrix{row_number, 2} = "RH_close";
           event_matrix{row_number, 3} = data_duration;
           row_number = row_number + 1;
           i = j;
       end
   end
   i = i + 1;
end

%% Apply High-Pass and Butterworth Low-Pass filter
% Resembles unsharp-masking

beta = .85;
beta_1 = 1-beta;

y_1 = mean(buffer(1,:));

for i=2:size(buffer,1)
    y = beta*y_1 + beta_1*buffer(i,:);
    y_1 = y;
    buffer(i, :) = buffer(i, :) - y;
end

% Apply high-pass Butterworth filter

Fs = 900; % Set sampling frequency to 1000Hz
[b, a] = butter(6, 50/(Fs/2));
filtered_buffer = filtfilt(b, a, buffer);

% % % Set notch filter at: 38Hz, 70Hz
% [num, den] = iirnotch(70/500, 6/(500));
% filtered_buffer = filtfilt(num, den, filtered_buffer);

% [num, den] = iirnotch(39/500, 7/500);
% filtered_buffer = filtfilt(num, den, filtered_buffer);

% We through out the first 120 samples of data to get rid of noise
filtered_buffer = filtered_buffer(120:size(filtered_buffer, 1), :);
finger_data = finger_data(120:size(finger_data, 1), :);

%% Generating Raw Data Graph
prompt = 'Input the channel you wish to analyze [1, 16]\n';
input_channel = input(prompt)

figure(1);
for i=1:30
    %yyaxis left
    hold on
    for i=5:10
        plot(1:1000, filtered_buffer((i-1)*1000 + 1:i*1000, i))
    end
    %yyaxis right
    %plot(1:1000, finger_data((i-1)*1000 + 1:i*1000))
    %ylim([-1.1, 1])
    %legend('Raw Data','Finger Labels')
    title('Raw Channel')
    pause
end

%% Generating PSD Data Graph

prompt = 'How many sequential windows would you like to analyze [1, 158]\n';
windows_viewed = input(prompt)

for i=1:windows_viewed
    [Pxx_filtered, F] = pwelch(filtered_buffer((i-1)*200 + 1:i*200, :), hann(100), 50, 4000, 900);
    
    % Calculate Mean
    Pxx_mean = mean(Pxx_filtered, 2);
    figure(1)
    delta_band = mean(Pxx_mean(1:17, 1)); % 4*value + 1
    theta_band = mean(Pxx_mean(18:33, 1));
    alpha_band = mean(Pxx_mean(34:53, 1));
    beta_band = mean(Pxx_mean(54:121, 1));
    gamma_band = mean(Pxx_mean(122:length(Pxx_mean)));
    bar_graph_display = [delta_band, theta_band, alpha_band, beta_band, gamma_band];
    bar_graph_naming = categorical({'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'});
    bar_graph_naming = reordercats(bar_graph_naming,{'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'});
    bar(bar_graph_naming, bar_graph_display)
    
    
    figure(5);
    
    hold off
    
    for j =1:16
        if j>1
            hold on
        end
        plot(F, Pxx_filtered(:, j) - Pxx_mean * max(Pxx_filtered(:, 1)) / max(Pxx_mean), 'DisplayName', compose("Channel %d", j))
    end

    title('PSD Readings With Average Removed')
    xlim([0, 100])
    xlabel('Frequency (Hz)')
    ylabel('Power')
    
%     % Remove scalar quantity per channel
%     Pxx_filtered = Pxx_filtered - Pxx_mean;
    
    figure(2);
    hold off
    
    for j =1:16
        if j>1
            hold on
        end
        plot(F, Pxx_filtered(:, j), 'DisplayName', compose("Channel %d", j))
    end

    title('PSD Readings')
    xlim([0, 100])
    xlabel('Frequency (Hz)')
    ylabel('Power')
    
    

    figure(3);
    plot((i-1)*200 + 1:i*200,finger_data((i-1)*200 + 1:i*200))
    %plot((i-1)*1000 + 1:i*1000,classification_finger_data((i-1)*1000 + 1:i*1000))
    hold off
    ylim([-1.1, 5])
    title("Finger Readings")
    xlabel('Time Stamp (ms)')
    ylabel('Finger Value [-1, 1]')
    pause
end

%% Analyze Hand Movement Classification

% The labels are supplied as follows:
%   0: Open hand label
%   1: Closing hand label
%   2: Closed hand label
%   3: Openning hand label
prompt = 'How many sequential windows would you like to analyze [1, 158]\n';
windows_viewed = input(prompt)

open_hand = classification_finger_data;
closing_hand = classification_finger_data;
closed_hand = classification_finger_data;
openning_hand = classification_finger_data;

open_hand(classification_finger_data~=0) = NaN;
closing_hand(classification_finger_data~=1)= NaN;
closed_hand(classification_finger_data~=2) = NaN;
openning_hand(classification_finger_data~=3) = NaN;

for i=1:windows_viewed
    figure(4);
    plot((i-1)*1000 + 1:i*1000, finger_data((i-1)*1000 + 1:i*1000), 'b', (i-1)*1000 + 1:i*1000 , open_hand((i-1)*1000 + 1:i*1000), 'y',(i-1)*1000 + 1:i*1000 , closed_hand((i-1)*1000 + 1:i*1000), 'k', (i-1)*1000 + 1:i*1000 , openning_hand((i-1)*1000 + 1:i*1000), 'm',(i-1)*1000 + 1:i*1000 , closing_hand((i-1)*1000 + 1:i*1000), 'c');
    ylim([-1.1, 4])
    title("Finger Readings")
    xlabel('Time Stamp (ms)')
    ylabel('Finger Value [-1, 1]')
    legend('Continuous Movement', 'Open Hand', 'Closed Hand', 'Openning Hand', 'Closing Hand')
    pause
end

%% Separating Frequency Bands
output_labels = finger_data(500:10000);
% Size is number of labels, # EEG Channels, Feature Per channel
new_features = zeros(length(output_labels), 16, 3);

% Set upper bound to 1000
for i=500:10000
   [Pxx_filtered, F] = pwelch(filtered_buffer(i-499:i, :), hann(100), 50, 4000, 1000);
    
    % Band 1: 15-18Hz
    band1 = Pxx_filtered(15*4+1:18*4+1,:);
    band1 = mean(band1, 1);
    new_features(i, :, 1) = band1';

    % Band 2: 36-40Hz
    band2 = Pxx_filtered(36*4+1:40*4+1,:);
    band2 = mean(band2, 1);
    new_features(i, :, 2) = band2';
    
    % Band 3: 50-54Hz
    band3 = Pxx_filtered(50*4+1:54*4+1,:);
    band3 = mean(band3, 1);
    new_features(i, :, 3) = band3';

end
% Utilizing SVM


%

%% Generating PSD/PCA Data Graph

figure(4);
for i=1:30
    [Pxx_filtered, F] = pwelch(filtered_buffer((i-1)*1000 + 1:i*1000, input_channel), hann(100), 50, 4000, 1000);
    %Pxx_filtered_pca = pca(Pxx_filtered);
    yyaxis left
    plot(F, Pxx_filtered_pca)
    yyaxis right
    plot(finger_data((i-1)*1000 + 1:i*1000))
    ylim([-1.1, 1])
    legend('PCA Output','Finger Labels')
    pause
end