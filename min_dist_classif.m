function predicted = min_dist_classif(data_tr, data_te, dist)

    predicted = zeros(size(data_te.X, 2),1);

    if data_tr.num_class == 2
%%
        prototype1 = mean(data_tr.Xclass1,2);
        prototype2 = mean(data_tr.Xclass2,2);

        if strcmp(dist, 'euc')
            g2 = @(x) prototype2' * x - 0.5 .* (prototype2' * prototype2);
            g1 = @(x) prototype1' * x - 0.5 .* (prototype1' * prototype1);
            classif  = @(x) g2(x) - g1(x);
            
        
        elseif strcmp(dist, 'mah')

            C = 0.5.*(cov(data_tr.Xclass1') + cov(data_tr.Xclass2'));
        
            g2 = @(x) prototype2'/C * x - 0.5 .* ((prototype2'/C) * prototype2);
            g1 = @(x) prototype1'/C * x - 0.5 .* ((prototype1'/C) * prototype1);
            classif = @(x) g2(x) - g1(x);
            
        end

        for i = 1:size(data_te.X, 2)
    
            if classif(data_te.X(:,i)) > 0
                predicted(i) = 1; 
            else
                predicted(i) = 2;
            end
        end
        
    elseif data_tr.num_class == 3
    
        % 1 contra 2 e 3
        
        prototype1 = mean(data_tr.Xclass1,2);
        prototype2 = mean([data_tr.Xclass2, data_tr.Xclass3],2);
    
        if strcmp(dist, 'euc')
            g2 = @(x) prototype2'*x - 0.5*(prototype2'*prototype2);
            g1 = @(x) prototype1'*x - 0.5*(prototype1'*prototype1);
            classif1  = @(x) g2(x) - g1(x);
            
        
        elseif strcmp(dist, 'mah')
            C = 0.5.*(cov(data_tr.Xclass1') + cov([data_tr.Xclass2, data_tr.Xclass3]'));
        
            g2 = @(x) prototype2'/C*x - 0.5.*((prototype2'/C)*prototype2);
            g1 = @(x) prototype1'/C*x - 0.5.*((prototype1'/C)*prototype1);
            classif1 = @(x) g2(x) - g1(x);
            
        end
    
        % 2 contra 1 e 3
        
        prototype1 = mean(data_tr.Xclass2,2);
        prototype2 = mean([data_tr.Xclass1, data_tr.Xclass3],2);
    
        if strcmp(dist, 'euc')
            g2 = @(x) prototype2'*x - 0.5*(prototype2'*prototype2);
            g1 = @(x) prototype1'*x - 0.5*(prototype1'*prototype1);
            classif2  = @(x) g2(x) - g1(x);
            
        
        elseif strcmp(dist, 'mah')
            C = 0.5.*(cov(data_tr.Xclass2') + cov([data_tr.Xclass1, data_tr.Xclass3]'));
        
            g2 = @(x) prototype2'/C*x - 0.5.*((prototype2'/C)*prototype2);
            g1 = @(x) prototype1'/C*x - 0.5.*((prototype1'/C)*prototype1);
            classif2 = @(x) g2(x) - g1(x);
            
        end
    
         % 3 contra 1 e 2
        
        prototype1 = mean(data_tr.Xclass3,2);
        prototype2 = mean([data_tr.Xclass1, data_tr.Xclass2],2);
    
        if strcmp(dist, 'euc')
            g2 = @(x) prototype2'*x - 0.5*(prototype2'*prototype2);
            g1 = @(x) prototype1'*x - 0.5*(prototype1'*prototype1);
            classif3 = @(x) g2(x) - g1(x);
            
        
        elseif strcmp(dist, 'mah')
            C = 0.5.*(cov(data_tr.Xclass3') + cov([data_tr.Xclass1, data_tr.Xclass2]'));
        
            g2 = @(x) prototype2'/C*x - 0.5.*((prototype2'/C)*prototype2);
            g1 = @(x) prototype1'/C*x - 0.5.*((prototype1'/C)*prototype1);
            classif3 = @(x) g2(x) - g1(x);
            
        end
    
        for i = 1:size(data_te.X,2)
    
            vector = [classif1(data_te.X(:,i)) classif2(data_te.X(:,i)) classif3(data_te.X(:,i))];
            [~, index] = max(vector); % confirmar isso
            predicted(i) = index;
    
        end
            
    end
    
end

