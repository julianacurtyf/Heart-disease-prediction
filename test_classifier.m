function conf_per_class = test_classifier(real_targets, predicted_targets, scenario)

    % This function calculates the metrics to evaluate the classifiers. The
    % calculated metrics are: sensitivity, precision and f1 score. Since we
    % have one scenario with three classes, the metrics are calculated
    % class by class, i.e., we cannot give the result as positive and
    % negative classes. The function returns a row matrix with all the
    % metrics. 
    
    if strcmp(scenario, 'A') || strcmp(scenario, 'B')
    
        conf_matrix = confusionmat(real_targets, predicted_targets, 'Order',[1 2]);
    
        % Class 1
    
        TP = conf_matrix(1,1);
    
        TN = conf_matrix(2,2);
    
        FN = conf_matrix(2,1);
    
        FP = conf_matrix(1,2);
    
        sensitivity = TP/(TP + FN); % recall
    
        precision = TP/(TP + FP);
    
        fscore = 2 * sensitivity * precision/(sensitivity + precision);

        specificity = TN / (TN + FP);
    
  
    
        conf_per_class = [sensitivity, precision, fscore, specificity];
    
    else
    
        conf_matrix = confusionmat(real_targets, predicted_targets, 'Order',[1 2 3]);
    
        % Class 1
    
        TP_1 = conf_matrix(1,1);
    
        TN_1 = conf_matrix(2,2) + conf_matrix(2,3) + conf_matrix(3,2) + conf_matrix(3,3);
    
        FN_1 = conf_matrix(2,1) + conf_matrix(3,1);
    
        FP_1 = conf_matrix(1,2) + conf_matrix(1,3);
    
        sensitivity_1 = TP_1/(TP_1 + FN_1);
    
        precision_1 = TP_1/(TP_1 + FP_1);
    
        fscore_1 = 2 * sensitivity_1 * precision_1/(sensitivity_1 + precision_1);
    
        % Class 2
    
        TP_2 = conf_matrix(2,2);
    
        TN_2 = conf_matrix(1,1) + conf_matrix(1,3) + conf_matrix(3,1) + conf_matrix(3,3);
    
        FN_2 = conf_matrix(1,2) + conf_matrix(3,2);
    
        FP_2 = conf_matrix(2,1) + conf_matrix(2,3);
    
        sensitivity_2 = TP_2/(TP_2 + FN_2);
    
        precision_2 = TP_2/(TP_2 + FP_2);
    
        fscore_2 = 2 * sensitivity_2 * precision_2/(sensitivity_2 + precision_2);
    
        % Class 3
    
        TP_3 = conf_matrix(3,3);
    
        TN_3 = conf_matrix(1,1) + conf_matrix(1,2) + conf_matrix(2,1) + conf_matrix(2,2);
    
        FN_3 = conf_matrix(1,3) + conf_matrix(2,3);
    
        FP_3 = conf_matrix(3,1) + conf_matrix(3,2);
    
        sensitivity_3 = TP_3/(TP_3 + FN_3);
    
        precision_3 = TP_3/(TP_3 + FP_3);
    
        fscore_3 = 2 * sensitivity_3 * precision_3/(sensitivity_3 + precision_3);
    
        conf_per_class = [sensitivity_1, precision_1, fscore_1; sensitivity_2, precision_2, fscore_2; sensitivity_3, precision_3, fscore_3];

    end



end

