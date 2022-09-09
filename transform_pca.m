function [data_tr, data_te, val_set] = transform_pca(data_tr, data_te, val_set, num_dim)

    % This function transforms the data accordingly to the Principal
    % Component Analysis. The resulting number of dimensions corresponds
    % to the value num_dim given as an input. This function also returns, also, the
    % test data transformed.

    [eigen_vectors, eigen_values] = eig(cov(data_tr.X'));
    [~, index] = sort(sum(eigen_values),'descend');
    
    eigen_vectors = eigen_vectors(:, index);

    W = eigen_vectors(:, 1:num_dim);

    data_tr.X = (data_tr.X' * W)';
    data_tr.dim = size(data_tr.X,1);

    data_te.X = (data_te.X' * W)';
    data_te.dim = size(data_te.X,1);

    val_set.X = (val_set.X' * W)';
    val_set.dim = size(val_set.X,1);

    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);

    if data_tr.num_class == 3
        data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);
    end

end