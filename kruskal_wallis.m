function kruskal_wallis(data)
    %%
    % This function calculates the p-value for each feature in both
    % scenarios: 2 or 3 classes. Then, using the threshold that was given
    % as an input, it calculates the indexes of the features that have a
    % p-value greater than the threshold. Those indexes are returned by the
    % function.

    H = zeros(data.dim, 2);
    H(:, 1) = (1:data.dim)';
    features_names = {'BMI5', 'Smoking','AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',...
        'DiffWalking', 'Sex', 'AgeCategory', 'Race',	'Diabetic',	'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma'};
    
    if data.num_class == 2
        for i = 1:data.dim
            pvalue = kruskalwallis([data.Xclass1(i, :)', data.Xclass2(i, :)'], [], "off");
            H(i, 2) = pvalue;
        
        end

    elseif data.num_class == 3
        for i = 1:data.dim
            pvalue = kruskalwallis([data.Xclass1(i, :)', data.Xclass2(i, :)', data.Xclass3(i, :)'], [], "off");
            H(i, 2) = pvalue;
        end
    end
    
    [~, order] = sort(H(:,2));
    H = H(order, :);
    features_names = features_names(order);
    pvalues = H(:,2);
        
    figure
    bar(pvalues);
    xticklabels(features_names);
    ylabel('p-values')

end

