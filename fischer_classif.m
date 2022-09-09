function predicted = fischer_classif(data_tr, data_te)

    predicted = zeros(size(data_te.X, 2),1);
  
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
           
        g2 = @(x) (W'*prototype2')'*W'*x - 0.5*prototype2'*W'*prototype2;
        g1 = @(x) prototype1'*W'*x - 0.5*prototype1'*W'*prototype1;
        
        classif  = @(x) g2(x) - g1(x);

        %%
    
        for i = 1:size(data_te, 2)
    
            if classif(data_te.X(:,i)') > 0
                predicted(i) = 1; % confirmar isso
            else
                predicted(i) = 2;
            end
        end
        

    elseif data_tr.num_class == 3
        
        [data_tr, data_te] = transform_lda(data_tr,data_te);
        min_dist_classif(data_tr, data_te, 'euc')
    
    end
    
   
end

