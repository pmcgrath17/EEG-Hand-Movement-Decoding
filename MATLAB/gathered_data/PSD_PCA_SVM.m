filepath = "/Users/patrickmcgrath/PycharmProjects/EEG-Hand-Movement-Decoding-Updated/MATLAB/gathered_data/";

subject_name = "subject4"

event_file_right = strcat(filepath, subject_name, "_right_events.txt");
event_file_left = strcat(filepath, subject_name, "_left_events.txt");

ica_file_right = strcat(filepath, subject_name, "_right_ica.txt");
ica_file_left = strcat(filepath, subject_name, "_left_ica.txt");

%% Read in Event Data (Open Closed Labels) RIGHT HAND
[number_right, latency_right, type_right, duration_right] = readvars(event_file_right);
type_right = string(type_right(:));

[number_left, latency_left, type_left, duration_left] = readvars(event_file_left);
type_left = string(type_left(:));

% We will the following labels:
%   RH_close = 0
%   RH_open  = 0
%   LH_close = 1
%   LH_open  = 1
%   NM       = 2

%hand_labels = type == "RH_close"; % We will treat right hand movement as 1
%hand_labels = type == "NM"; % The other class from the data will be no hand movement

%% Read in ICA Data & Normalize It
ica_matrix_right = readmatrix(ica_file_right);
ica_matrix_right = ica_matrix_right(:, 2:size(ica_matrix_right, 2));
ica_matrix_right = normalize(ica_matrix_right);

ica_matrix_left = readmatrix(ica_file_left);
ica_matrix_left = ica_matrix_left(:, 2:size(ica_matrix_left, 2));
ica_matrix_left = normalize(ica_matrix_left);
%% Run Power Spectral Density
"Running PSD"

% Hyperparameters:
window_size = 100; % .1 seconds (originally .2 seconds) 
window_shift = 100; % shift every 100 samples (originally 31)
percent_overlap = (window_size - window_shift) / window_size
number_of_buckets = 20; % number of frequency buckets for PSD
max_frequency = 60;
min_frequency = 1;

feature_number = 1; % internal variables
labeled_features = {};

for idx=1:length(latency_right)
    time_period = latency_right(idx);
    time_length = duration_right(idx);
    % Set the period of interest large enough to start calculating features
    % on the onset of hand movement
    if type_right(idx) == "NM"
        time_period = time_period + 1000;
        time_length = time_length - 2000;
    end
    period_of_interest = ica_matrix_right(time_period - window_size/2:time_period + time_length + window_size/2, :); 
    % here we define all the features for this window
    num_feature_groups = (time_length - 1)/window_shift;
    
    % we take steps of size window_shift
    for j=1:window_shift:time_length
        % Other hyperparmeters: pwelch(X, WINDOW, NOVERLAP, NFFT, Fs)
        % WINDOW - how each data column is split up
        % NOVERLAP - % of samples to shift
        % NFFT - number of FFT computations to compute
        % Fs - sampling frequency
        NFFT = 4000;
        Fs = 1000;
        [Pxx, F] = pwelch(period_of_interest(j: window_size + j, :), hann(100), 50, NFFT, Fs);
        bucket_iterator = int64((NFFT/Fs)*(max_frequency - min_frequency)/number_of_buckets);
        features = zeros([number_of_buckets, size(Pxx, 2)]);
        
        for k=1:number_of_buckets
            bucket_low = (k-1)*bucket_iterator + min_frequency*(NFFT/Fs);
            bucket_high = k*bucket_iterator + min_frequency*(NFFT/Fs) - 1;
            features(k, :) = mean(Pxx(bucket_low:bucket_high, :));
        end
        
        flattened_features = features(:);
        labeled_features{feature_number} = {flattened_features, type_right(idx)};
        feature_number = feature_number + 1;
    end
    
end

for idx=1:length(latency_left)
    time_period = latency_left(idx);
    time_length = duration_left(idx);
    % Set the period of interest large enough to start calculating features
    % on the onset of hand movement
    if type_left(idx) == "NM"
        time_period = time_period + 1000;
        time_length = time_length - 2000;
    end
    period_of_interest = ica_matrix_left(time_period - window_size/2:time_period + time_length + window_size/2, :); 
    % here we define all the features for this window
    num_feature_groups = (time_length - 1)/window_shift;
    
    % we take steps of size window_shift
    for j=1:window_shift:time_length
        % Other hyperparmeters: pwelch(X, WINDOW, NOVERLAP, NFFT, Fs)
        % WINDOW - how each data column is split up
        % NOVERLAP - % of samples to shift
        % NFFT - number of FFT computations to compute
        % Fs - sampling frequency
        NFFT = 4000;
        Fs = 1000;
        [Pxx, F] = pwelch(period_of_interest(j: window_size + j, :), hann(100), 50, NFFT, Fs);
        bucket_iterator = int64((NFFT/Fs)*(max_frequency - min_frequency)/number_of_buckets);
        features = zeros([number_of_buckets, size(Pxx, 2)]);
        
        for k=1:number_of_buckets
            bucket_low = (k-1)*bucket_iterator + min_frequency*(NFFT/Fs);
            bucket_high = k*bucket_iterator + min_frequency*(NFFT/Fs) - 1;
            features(k, :) = mean(Pxx(bucket_low:bucket_high, :));
        end
        
        flattened_features = features(:);
        labeled_features{feature_number} = {flattened_features, type_left(idx)};
        feature_number = feature_number + 1;
    end
end

%% Run Principal Component Analysis
"Running PCA"
% Generate matrix of variables for PCA algorithm
feature_matrix = zeros([size(labeled_features, 2), length(labeled_features{1,1}{1,1})]);
y_label = zeros([size(labeled_features, 2), 1]);
row_to_delete = [];
for i=1:size(labeled_features, 2)  
    
    switch (labeled_features{1,i}{1,2})
        case "RH_close"
            y_label(i) = 1;
        case "RH_open"
            y_label(i) = 1;
        case "LH_close"
            y_label(i) = 0;
        case "LH_open"
            y_label(i) = 0;
        case "NM"
            y_label(i) = Nan;
        otherwise
            y_label(i) = NaN;
    end
    
    if isnan(y_label(i))
        row_to_delete = [row_to_delete, i];
    end
    
    features_col = labeled_features{1, i}{1, 1};
    feature_matrix(i, :) = features_col.';
end

y_label(row_to_delete) = [];
feature_matrix(row_to_delete, :) = [];

% Normalize values prior to running PCA (using z-score)
normalized_features = normalize(feature_matrix);

% Get principal component:
%   coefficient (eigenvectors)
%   scores (matrix of oberservations projected onto eigenvectors)
%   latent (eigenvectors [aka variance])
[coeff,score,latent] = pca(normalized_features);

%% Visualization For PCA

cummulative_variance = zeros(size(latent));
cummulative_variance(1) = latent(1);


for i=2:length(latent)
    cummulative_variance(i) = cummulative_variance(i-1) + latent(i);
end

cummulative_variance = cummulative_variance / sum(latent);

percentile_98 = 0;
for j=1:length(cummulative_variance)
    if cummulative_variance(j) >= .98
        percentile_98 = j
        break
    end
end

%score_98th = score(:, 1:percentile_98);

% plot(1:length(latent), cummulative_variance, 1:length(latent), .98*ones(size(latent)));
% title('Percent Variance From Cummulative Eigenvectors (Subject 1)');
% xlabel('Eigenvalue Size (largest to smallest)')
% ylabel('Percentage of Total Variance')
% 
% legend('Cummulative Magnitude','98 Percentile of Variance')

%% Break Up Data

% Take average out before PSD
% Take PSD
% normalize -> find maximum and minumum PSD reading among all buckets
%   normalize data based on these values
% Apply PCA & pass data into SVM and MLP

random_shuffle = randperm(length(y_label));
y_label = y_label(random_shuffle);
score = score(random_shuffle, :);

% Get subset of data that is split 50/50 (movement vs. no movement)
y_pos = length(y_label(y_label == 1))
y_neg = length(y_label(y_label == 0))

row_to_delete = [];

if y_pos ~= y_neg
    num_need_removal = abs(y_pos - y_neg);
    idx = 1;
    % Remove positive labels
    if y_pos > y_neg
        while num_need_removal ~= 0 && idx < length(y_label)
            if y_label(idx) == 1
                row_to_delete = [row_to_delete, idx];
                num_need_removal = num_need_removal - 1; 
            end
            idx = idx + 1;
        end
    % Remove negative labels    
    else
        while num_need_removal ~= 0 && idx < length(y_label)
            if y_label(idx) == 0
                row_to_delete = [row_to_delete, idx];
                num_need_removal = num_need_removal - 1; 
            end
            idx = idx + 1;
        end
    end
end

y_label(row_to_delete) = [];
score(row_to_delete, :) = [];

%% Write Matrix to File For Python To Implement Decoder

% Shuffle data
random_shuffle = randperm(length(y_label));
y_label = y_label(random_shuffle);
score = score(random_shuffle, :);

% 98th percentile of scores?
score = score(:, 1:percentile_98);

writematrix(score, '/Users/patrickmcgrath/PycharmProjects/EEG-Hand-Movement-Decoding-Updated/models/classifiers/EEGLAB/pca_processed_features.csv');
writematrix(y_label, '/Users/patrickmcgrath/PycharmProjects/EEG-Hand-Movement-Decoding-Updated/models/classifiers/EEGLAB/pca_processed_labels.csv');
%% Run SVM
svm_model = fitcsvm(score, y_label, 'KernelFunction', 'rbf','Leaveout', 'on', 'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'NumGridDivisions', 30))

loss = kfoldLoss(svm_model);

%% Statistics on Y

count_pos_label = 0;
for i=1:length(y_label)
    if y_label(i) == 1
        count_pos_label = count_pos_label + 1;
    end
end

percentage_breakdown = count_pos_label / length(y_label)