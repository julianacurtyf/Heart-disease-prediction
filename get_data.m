function [data, val_set] = get_data()

    % This function returns the dataset of the problem in a structure, with
    % X being the features, y the target for a particular scenario, targets 
    % being the four dependents variables, num the nember of samples and
    % dim the dimension of the problem. All the information comes from a
    % .mat file whereas the original csv is loaded to.

    all_data = load('all_data.mat').all_data;
    data_val = all_data(200001:end,:);
    all_data = all_data(1:200e3, :);
    targets = all_data(:,1:4);
    features = all_data(:,5:end);
    val_targets = data_val(:,1:4);
    val_features = data_val(:,5:end);
    
    data.X = features';
    data.y = zeros(size(data.X,2),1);
    data.targets = targets;
    data.num = size(data.X,2);  
    data.dim = size(data.X,1);

    val_set.X = val_features';
    val_set.y = zeros(size(val_set.X,2),1);
    val_set.targets = val_targets;
    val_set.num = size(val_set.X,2);  
    val_set.dim = size(val_set.X,1);


end