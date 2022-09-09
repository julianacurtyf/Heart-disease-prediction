function predicted = svm_linear_classif(data_tr, data_te)

    % This function creates a SVM classifier, with the kernel function being linear
    % and returns the predicted value obtained in the model for the test set.

    model = fitcsvm(data_tr.X', data_tr.y, 'KernelFunction','linear');

    predicted = predict(model, data_te.X');

end

