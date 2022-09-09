function  [data_tr, data_te, val_set] = select_features(data_tr, data_te, val_set,features_index)

    % This function takes as an input the train and test data structure and 
    % the features' index that we want to mantain, and returns the data 
    % structure only with the features that we kept. 

    data_tr.X = data_tr.X(features_index,:);  
    data_tr.dim = size(data_tr.X,1);

    data_te.X = data_te.X(features_index,:);  
    data_te.dim = size(data_te.X,1);

    val_set.X = val_set.X(features_index,:);  
    val_set.dim = size(val_set.X,1);

    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);

    if data_tr.num_class == 3
        data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);
    end
end