function roc_curves(data)
    
    % This function calculates the area under the curve (AUC) for each feature.
    % Then, using the threshold that was given as an input, it calculates 
    % the indexes of the features that have AUC greater than the threshold. 
    % Those indexes are returned by the function.

    R = zeros(data.dim, 2);
    R(:, 1) = (1:data.dim)';
    features_names = {'BMI5', 'Smoking','AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',	'Race',	'Diabetic',	'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma'};
        
    for i = 1:data.dim
        [FPR, FNR] = roc1(data.X(i,:), data.y);
    
        SS = 1 - FNR; % Sensitivity 
    
        AUC = 0;
    
        for j = 2:numel(FPR)
            AUC = AUC + (SS(j)*(FPR(j-1)-FPR(j)));
        end
    
        R(i, 2) = AUC;

    end
    [~, order] = sort(R(:,2),'descend');
    R = R(order, :);
    features_names = features_names(order);

    AUC_values = R(:,2);

    figure
    bar(AUC_values);
    xticklabels(features_names);
    ylabel('AUC');



end

