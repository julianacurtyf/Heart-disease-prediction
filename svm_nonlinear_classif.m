function predicted = svm_nonlinear_classif(data_tr, data_te)

    % This function creates a SVM classifier, with the kernel function
    % being radial basis function (rbf) and returns the predicted value 
    % obtained in the model for the test set.

    model = fitcsvm(data_tr.X', data_tr.y, 'KernelFunction','rbf');

    predicted = predict(model, data_te.X');

end
