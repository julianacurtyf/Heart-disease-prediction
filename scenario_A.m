function [data, data_tr, data_te, val_set] = scenario_A(data, val_set)

    % In this fucntion we preprocess the data for the scenario A, i.e., we
    % define the specific target, we divide into train and test set, we
    % balance the data using undersampling, and finally, we standardize the
    % data. It returns the structured data that came as an input with the 
    % correct values for this class, also the train and test sets.

    % Defining the target for this scenario

    for i = 1:data.num
        if data.targets(i,1) == 1
            data.y(i) = 2;
        else
            data.y(i) = 1;
        end
    end

    for i = 1:val_set.num
        if val_set.targets(i,1) == 1
            val_set.y(i) = 2;
        else
            val_set.y(i) = 1;
        end
    end

    data.num_class = 2;
    data.Xclass1 = data.X(:, data.y == 1);
    data.Xclass2 = data.X(:, data.y == 2);

    % Splitting the data into training set and testing set

    ix = randperm(data.num);
    
    ixtr = ix(1:floor(data.num*0.7));
    ixte = ix(floor(data.num*0.7)+1:end);
     
    data_tr.X = data.X(:,ixtr);
    data_tr.y = data.y(ixtr);
    data_tr.dim = size(data_tr.X,1);
    data_tr.num = size(data_tr.X,2);
    data_tr.name = 'training dataset';
    data_tr.num_class = 2;
    
    data_te.X = data.X(:,ixte);
    data_te.y = data.y(ixte);
    data_te.dim = size(data_te.X,1);
    data_te.num = size(data_te.X,2);
    data_te.name = 'testing dataset';
    data_te.num_class = 2;
    
    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);

    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2)];

    % Balancing the training set using under sampling 

    [m, i] = min(data_tr.numel_classes);
    
    iz = randperm(m); % smallest class

    if i == 1

        data_tr.Xclass2 = data_tr.Xclass2(:, iz);
        data_tr.X = cat(2, data_tr.Xclass1, data_tr.Xclass2);
        data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2)];
        data_tr.y = [ones(data_tr.numel_classes(1), 1); 2*ones(data_tr.numel_classes(2), 1)];

    else
        data_tr.Xclass1 = data_tr.Xclass1(:, iz);
        data_tr.X = cat(2, data_tr.Xclass1, data_tr.Xclass2);
        data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2)];
        data_tr.y = [ones(data_tr.numel_classes(1), 1); 2*ones(data_tr.numel_classes(2), 1)];

    end
    
    data_tr.num = size(data_tr.X,2);

%     % Balancing the training set using over sampling 
% 
%     [m, i] = max(data_tr.numel_classes);
%     
%     if i == 1
% 
%         for i = 1:m-data_tr.numel_classes(2)
%             iz = randi(data_tr.numel_classes(2));
%             data_tr.X = [data_tr.X, data_tr.Xclass2(:, iz)];
%             data_tr.y = [data_tr.y; 2];
% 
%         end
%         
%     else
% 
%         for i = 1:m-data_tr.numel_classes(1)
%             iz = randi(data_tr.numel_classes(1));
%             data_tr.X = [data_tr.X, data_tr.Xclass1(:, iz)];
%             data_tr.y = [data_tr.y; 1];
% 
%         end
% 
%     end
%     
%     data_tr.num = size(data_tr.X,2);

    % Standardizing the data

    data_te.X = normalize(data_te.X);
    data_tr.X = normalize(data_tr.X);
    val_set.X = normalize(val_set.X);
    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);
    data_tr.numel_classes = [size(data_tr.Xclass1, 2) size(data_tr.Xclass2, 2)];
    data_te.Xclass1 = data_te.X(:, data_te.y == 1);
    data_te.Xclass2 = data_te.X(:, data_te.y == 2);
    val_set.Xclass1 = val_set.X(:, val_set.y == 1);
    val_set.Xclass2 = val_set.X(:, val_set.y == 2);
    
    


end