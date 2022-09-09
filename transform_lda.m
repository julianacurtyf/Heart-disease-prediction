function [data_tr, data_te, val_set] = transform_lda(data_tr, data_te, val_set)

    % This function transforms the data accordingly to the Linear
    % Discriminant Analysis. The resulting number of dimensions corresponds
    % to the number of classes - 1. This function also returns, also, the
    % test data transformed.

    if data_tr.num_class == 2
%%
        prototype1 = mean(data_tr.Xclass1,2)';
        prototype2 = mean(data_tr.Xclass2,2)';
        
        S1 = cov(data_tr.Xclass1');
        S2 = cov(data_tr.Xclass2');
               
        Sw = S2 + S1;

        Sb = (prototype1 - prototype2)'*(prototype1 - prototype2);
        
        m = Sw\Sb;

        [V, ~]  = eig(m);

        W = V(:, 1);

    elseif data_tr.num_class == 3
        
        prototype1 = mean(data_tr.Xclass1,2)';
        prototype2 = mean(data_tr.Xclass2,2)';
        prototype3 = mean(data_tr.Xclass3,2)';

        overal_mean = (prototype1 + prototype2 + prototype3)./3;

        S1 = cov(data_tr.Xclass1');
        S2 = cov(data_tr.Xclass2');
        S3 = cov(data_tr.Xclass3');

        Sw = S1 + S2 + S3;
        
        N1 = size(data_tr.Xclass1,2);
        N2 = size(data_tr.Xclass2,2);
        N3 = size(data_tr.Xclass3,2);
        
        Sb1 = N1 .* (prototype1 - overal_mean)'*(prototype1 - overal_mean);
        Sb2 = N2 .* (prototype2 - overal_mean)'*(prototype2 - overal_mean);
        Sb3 = N3 .* (prototype3 - overal_mean)'*(prototype3 - overal_mean);
        
        Sb = Sb1 + Sb2 + Sb3;
        
        m = Sw\Sb;

        [V, ~]  = eig(m);

        W = V(:, 1:2);

    end

    data_tr.X = (data_tr.X'*W)';
    data_tr.dim = size(data_tr.X,1);

    data_te.X = (data_te.X'*W)';
    data_te.dim = size(data_te.X,1);

    val_set.X = (val_set.X'*W)';
    val_set.dim = size(val_set.X,1);

    data_tr.Xclass1 = data_tr.X(:, data_tr.y == 1);
    data_tr.Xclass2 = data_tr.X(:, data_tr.y == 2);

    if data_tr.num_class == 3
        data_tr.Xclass3 = data_tr.X(:, data_tr.y == 3);
    end

end

