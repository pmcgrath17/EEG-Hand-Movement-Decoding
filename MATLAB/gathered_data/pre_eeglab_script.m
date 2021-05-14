%% Collecting Data with Low-Pass Filter
% filename = 'C:\Users\mcgrathp\PycharmProjects\EEG-Decoder2\python\saved_data\train_02_10_21_3_fast_with_clench_blink.json';
foldername = '/Users/patrickmcgrath/PycharmProjects/EEG-Decoder2/python/hc_data/';
filename = 'Subject4_EEG_Right_Hand_5_7.json';

filename = strcat(foldername, filename)

json = jsondecode(fileread(filename));
buffer = flip(json.raw_list{2}, 1);

pattern = strfind(filename, 'Right')
if pattern ~= [] % If this is true, this is right hand movement
    finger_data = mean(json.finger_data(:,[2:5]), 2);
else % Else, this is left hand movement
    finger_data = mean(json.finger_data(:,[1 3:5]), 2);
end

%% Scale readings to [-1, 1] range

finger_data(:) = 2*(finger_data(:) - min(finger_data))/(max(finger_data) - min(finger_data)) - 1;

%% Filter Hand Movements

Fs = 33; % Set sampling frequency to 33Hz
[b, a] = butter(6, 4/(Fs/2));
finger_data = filtfilt(b, a, finger_data);

%% Miscelanious Graphing For Hand Classification
derivative_finger_data = zeros(size(finger_data));
second_derivative_finger_data = zeros(size(finger_data));
for i=1:size(finger_data, 1)-1
    derivative_finger_data(i) = finger_data(i+1)/finger_data(i) - 1;
    if derivative_finger_data(i) > 10
        derivative_finger_data(i) = 10;
    elseif derivative_finger_data(i) < -10
        derivative_finger_data(i) = -10; 
    end
end

for i=1:size(finger_data, 1)-1
    numerator = derivative_finger_data(i+1);
    denominator = derivative_finger_data(i);
    if numerator == 0 && denominator == 0
        % Make sure we don't divide 0 by 0
        numerator = 1;
        denominator = 1;
    elseif denominator == 0
        % Make sure denominator is not 0
        denominator = numerator;
    elseif abs(numerator - denominator) < .1
        numerator = 1;
        denominator = 1;
    end
    
    second_derivative_finger_data(i) = numerator/denominator - 1;
    
end

for i=1:size(finger_data, 1)/100
    figure(1)
    title('Finger Data and Derivative')
    yyaxis left
    plot((i-1)*100 + 1:i*100,finger_data((i-1)*100 + 1:i*100))
    ylim([-1,1])
    ylabel('Finger Data Value [-1, 1]')
    
    yyaxis right
    plot((i-1)*100 + 1:i*100,derivative_finger_data((i-1)*100 + 1:i*100))
    ylim([-2, 2])
    ylabel('Discrete Finger Data Derivative [-2, 2]')
    
    figure(2)
    title('Derivative and Second Derivative')
    yyaxis left
    plot((i-1)*100 + 1:i*100,derivative_finger_data((i-1)*100 + 1:i*100))
    ylim([-2,2])
    ylabel('Discrete Finger Data Derivative [-2, 2]')
    
    yyaxis right
    plot((i-1)*100 + 1:i*100,second_derivative_finger_data((i-1)*100 + 1:i*100))
    ylim([-2, 2])
    ylabel('Discrete Finger Data Second Derivative [-2, 2]')
    pause
end

%% Classify Finger Movement From Derivative Data
% The labels are supplied as follows:
%   0: Open hand label
%   1: Closing hand label
%   2: Closed hand label
%   3: Openning hand label
magnitude_first_derivative = abs(derivative_finger_data);
magnitude_second_derivative = abs(second_derivative_finger_data);

classified_hand_labels = zeros(size(finger_data));
classified_hand_labels(1) = 0;

hand_movements = 0;

for i=2:size(finger_data, 1)
    if magnitude_first_derivative(i) > 0.04 || magnitude_second_derivative(i) > 0.04
        if hand_movements == 1 || hand_movements == 3
            classified_hand_labels(i) = hand_movements;
        elseif hand_movements == 0 || hand_movements == 2
            relabel_elements = 0;
            for j=1:20
                if classified_hand_labels(i-j) == 1 || classified_hand_labels(i-j) == 3
                    % Make sure there's not a sudden jump in classification
                    % values
                    hand_movements = classified_hand_labels(i-j);
                    % Fix misalbelled values
                    classified_hand_labels((i-j):i) = hand_movements;
                    relabel_elements = 1;
                    break
                end
            end
            if relabel_elements == 1
                relabel_elements = 0;
            else
                hand_movements = hand_movements + 1;
                classified_hand_labels(i) = hand_movements; 
            end

        end
    else
        if classified_hand_labels(i-1) == 1 || classified_hand_labels(i-1) == 3
            
            hand_movements = mod(hand_movements + 1, 4);
        end
        
        classified_hand_labels(i) = hand_movements;
        
    end
end

%new_event_matrix = generate_event_matrix(classified_hand_labels);
%writecell(classified_hand_labels, 'new_dataset_event_matrix.csv', 'Delimiter', ',');
%% Looking at Classification Labels

for i=1:size(finger_data, 1)/100
    figure(5)
    title('Finger Data and Labels')
    yyaxis left
    plot((i-1)*100 + 1:i*100,finger_data((i-1)*100 + 1:i*100))
    ylim([-1,1])
    ylabel('Finger Data Value [-1, 1]')
    
    yyaxis right
    plot((i-1)*100 + 1:i*100,classified_hand_labels((i-1)*100 + 1:i*100))
    ylim([-1, 3])
    ylabel('Classified Finger Label [-1, 3]')
    pause
end

%% Utility Function (Reading Finger Data)
%function event_matrix = generate_event_matrix(finger_classification_data)

buffer = flip(json.raw_list{2}, 1);
finger_classification_data = repelem(classified_hand_labels(1), [size(buffer, 1)], [1]);
N = length(json.raw_list);

for i=3:N
   finger_classification_data = [finger_classification_data; repelem(classified_hand_labels(i-1), [size(json.raw_list{i}, 1)], [1])];
   buffer = [buffer; flip(json.raw_list{i}, 1)];
   i
end
%%
event_matrix = cell(length(finger_classification_data), 3);
row_number = 1;
i = 1;
while (i <= length(finger_classification_data))
    if finger_classification_data(i) == 1
        data_duration = 1;
        for j=i+1:length(finger_classification_data)
            if finger_classification_data(j) == 1
                data_duration = data_duration + 1;
            else
                break
            end
        end
        if data_duration > 30
            event_matrix{row_number, 1} = i;
            event_matrix{row_number, 2} = "RH_close";
            event_matrix{row_number, 3} = data_duration;
            row_number = row_number + 1;
            i = j;
        end
    elseif finger_classification_data(i) == 3
        data_duration = 1;
        for j=i+1:length(finger_classification_data)
            if finger_classification_data(j) == 3
                data_duration = data_duration + 1;
            else
                break
            end
        end
        if data_duration > 30
            event_matrix{row_number, 1} = i;
            event_matrix{row_number, 2} = "RH_open";
            event_matrix{row_number, 3} = data_duration;
            row_number = row_number + 1;
            i = j;
        end
    elseif finger_classification_data(i) == 0
        data_duration = 1;
        for j=i+1:length(finger_classification_data)
            if finger_classification_data(j) == 0
                data_duration = data_duration + 1;
            else
                break
            end
        end
        if data_duration > 30
            event_matrix{row_number, 1} = i;
            event_matrix{row_number, 2} = "NM";
            event_matrix{row_number, 3} = data_duration;
            row_number = row_number + 1;
            i = j;
        end
    end
    i = i + 1;
end
writecell(event_matrix, 'hc_event_matrix_right_subject4.csv', 'Delimiter', ',');
%end