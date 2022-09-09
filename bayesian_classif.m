function predicted = bayesian_classif(data_tr, data_te)

    % This function creates a Bayesian classifier, and returns the predicted 
    % value obtained in the model for the test set.

    model = fitcnb(data_tr.X', data_tr.y);

    predicted = predict(model, data_te.X'); 
    

end

