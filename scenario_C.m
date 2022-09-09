function [data, data_tr, data_te, val_set] = scenario_C(data, val_set)

    % In this fucntion we preprocess the data for the scenario C, i.e., we
    % define the specific target, we divide into train and test set, we
    % balance the data using undersampling, and finally, we standardize the
    % data. It returns the structured data that came as an input with the 
    % correct values for this class, also the train and test sets.

    % Defining the target for this scenario

    for i = 1:data.num
        if (data.targets(i,1) == 1 || data.targets(i,2) == 1) && (data.targets(i,3) ~= 1 || data.targets(i,4) ~= 1)
            data.y(i) = 1;
    
        elseif (data.targets(i,1) == 1 || data.targets(i,2) == 1) && (data.targets(i,3) == 1 || data.targets(i,4) == 1)
            data.y(i) = 2;
    
        elseif data.targets(i,1) ~= 1 && data.targets(i,2) ~= 1
            data.y(i) = 3;
        end
    end

    for i = 1:val_set.num
        if (val_set.targets(i,1) == 1 || val_set.targets(i,2) == 1) && (val_set.targets(i,3) ~= 1 || val_set.targets(i,4) ~= 1)
            val_set.y(i) = 1;
    
        elseif (val_set.targets(i,1) == 1 || val_set.targets(i,2) == 1) && (val_set.targets(i,3) == 1 || val_set.targets(i,4) == 1)
            val_set.y(i) = 2;
    
        elseif val_set.targets(i,1) ~= 1 && val_set.targets(i,2) ~= 1
            val_set.y(i) = 3;
        end
    end

    data.num_class = 3;
    data.Xclass1 = data.X(:, data.y == 1);
    data.Xclass2 = data.X(:, data.y == 2);
    data.Xclass3 = data.X(:, data.y == 3);


    % Splitting the data into training set and testing set

    ix = randperm(data.num);

    ixtr = ix(1:floor(data.num*0.7));
    ixte = ix(floor(data.num*0.7)+1:end);
     
    data_tr.X = data.X(:,ixtr);
    data_tr.y = data.y(ixtr);
    data_tr.dim = size(data_tr.X,1);
    data_tr.num = size(data_tr.X,2);
    data_tr.name = 'training dataset';
    data_tr.num_class = 3;
    
    data_te.X = data.X(:,ixte);
    data_te.y = data.y(ixte);
    data_te.dim = size(data_te.X,1);
    data_te.num = size(data_te.X,2);
    data_te.name = 'testing dataset';
    data_te.num_class = 3;
    
    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);
    data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);

    data_tr.numel_classes = [size(data_tr.Xclass1, 2); size(data_tr.Xclass2, 2); size(data_tr.Xclass3, 2)];


    % Balancing the training set using under sampling 

    [m, i] = min(data_tr.numel_classes);
    
    iz = randperm(m); % smallest class

    if i == 1

    data_tr.Xclass2 = data_tr.Xclass2(:, iz);
    data_tr.Xclass3 = data_tr.Xclass3(:, iz);
    data_tr.X = cat(2, data_tr.Xclass1, data_tr.Xclass2, data_tr.Xclass3);
    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2) size(data_tr.Xclass3, 2)];
    data_tr.y = [ones(data_tr.numel_classes(1),1); 2*ones(data_tr.numel_classes(2),1); 3*ones(data_tr.numel_classes(3),1)];

    elseif i == 2
    data_tr.Xclass1 = data_tr.Xclass1(:, iz);
    data_tr.Xclass3 = data_tr.Xclass3(:, iz);
    data_tr.X = cat(2, data_tr.Xclass1, data_tr.Xclass2, data_tr.Xclass3);
    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2) size(data_tr.Xclass3, 2)];
    data_tr.y = [ones(data_tr.numel_classes(1),1); 2*ones(data_tr.numel_classes(2),1); 3*ones(data_tr.numel_classes(3),1)];

    else
    data_tr.Xclass1 = data_tr.Xclass1(:, iz);
    data_tr.Xclass2 = data_tr.Xclass2(:, iz);
    data_tr.X = cat(2, data_tr.Xclass1, data_tr.Xclass2, data_tr.Xclass3);
    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2) size(data_tr.Xclass3, 2)];
    data_tr.y = [ones(data_tr.numel_classes(1),1); 2*ones(data_tr.numel_classes(2),1); 3*ones(data_tr.numel_classes(3),1)];
    
    end

    % Standardizing the data


    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);
    data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);
    data_te.Xclass1 = data_te.X(:, data_te.y == 1);
    data_te.Xclass2 = data_te.X(:, data_te.y == 2);
    data_te.Xclass3 = data_te.X(:, data_te.y == 3);

    data_te.X = normalize(data_te.X);
    data_tr.X = normalize(data_tr.X);
    val_set.X = normalize(val_set.X);
    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);
    data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);
    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2) size(data_tr.Xclass3, 2)];
    data_te.Xclass1 = data_te.X(:, data_te.y == 1);
    data_te.Xclass2 = data_te.X(:, data_te.y == 2);
    data_te.Xclass3 = data_te.X(:, data_te.y == 3);
    val_set.Xclass1 = val_set.X(:, val_set.y == 1);
    val_set.Xclass2 = val_set.X(:, val_set.y == 2);
    val_set.Xclass2 = val_set.X(:, val_set.y == 3);


end