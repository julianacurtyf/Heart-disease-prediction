
function metric_value = main(scenario, features_index, dim_red, n_comp, classifier, neighbors)

    % This is the main function, and is called at the GUI. It calls all the
    % other functions dependending on the scenario, method of feature
    % selection, dimensionality reduction and classifier, which are parameters input. 
    % This function is separated by sections, in each section we have an
    % if-else statatement that calls the right function depending on the
    % input.
    %
    % This function returns the metrics for each class, since we deal with
    % problems that are not binary. The metrics are in a array, in the
    % following format: 
    % [sensitivity_1, precision_1, fscore_1, sensitivity_2, precision_2,
    % fscore_2, sensitivity_3, precision_3, fscore_3]
    % The numbers 1 up to 3 represent the classes.

    [data, val_set] = get_data();

    % SELECT THE SCENARIO

    if  strcmp(scenario, 'A') 
        [data, data_tr, data_te, val_set] = scenario_A(data,val_set);
    
    elseif strcmp(scenario, 'B') 
        [data, data_tr, data_te, val_set] = scenario_B(data,val_set);

    elseif strcmp(scenario, 'C') 
        [data, data_tr, data_te, val_set] = scenario_C(data,val_set);
    end

%     % HISTOGRAM AND BOXPLOT
% 
%     for i = 1:15
%         figure(1)
%         subplot(4,4,i)
%         boxplot(data_tr.X(i,:)) % plot the boxplot of each feature
%         
%         figure(2)
%         subplot(4,4,i)
%         histogram(data_tr.X(i,:)) % plot the histogram of each feature
%     end
    
    % SELECT FEATURE SELECTION METHOD
   

%     correlation_matrix(data_tr);
%     
%     kruskal_wallis(data_tr);
% 
%     roc_curves(data_tr);

    [data_tr, data_te, val_set] = select_features(data_tr, data_te,val_set, features_index);

    
    % SELECT DIMENSIONALITY REDUCTION TECHNIQUE

    if strcmp(dim_red, 'PCA') 
    
        %show_PCA(data_tr, 'scree')
        
        [data_tr, data_te, val_set] = transform_pca(data_tr, data_te, val_set, n_comp);
    
    elseif strcmp(dim_red, 'LDA') 
    
        [data_tr, data_te, val_set] = transform_lda(data_tr, data_te, val_set);
    
    end
    
    %% SELECT CLASSSIFIER
    
    if strcmp(classifier, 'mdc-euc') 
    
        dist = 'euc';
    
        predicted = min_dist_classif(data_tr, val_set, dist);
    
    elseif strcmp(classifier, 'mdc-mah') 
    
        dist = 'mah';
    
        predicted = min_dist_classif(data_tr, val_set, dist);
    
    elseif strcmp(classifier, 'fisher') 
    
        predicted = min_dist_classif(data_tr, val_set);
    
    elseif strcmp(classifier, 'bayes') 
    
        predicted = bayesian_classif(data_tr, val_set);
    
    elseif strcmp(classifier, 'knn') 
    
        num_neighbors = neighbors;
    
        predicted = knn_classif(data_tr, val_set, num_neighbors);
    
    elseif strcmp(classifier, 'svm-linear') 
    
        predicted = svm_linear_classif(data_tr, val_set);
    
    elseif strcmp(classifier, 'svm-nonlinear') 
    
        predicted = svm_nonlinear_classif(data_tr, val_set); 
    
    end
    
    
    % TEST THE CLASSIFIER
    
    metric_value = test_classifier(val_set.y, predicted, scenario);

end



























