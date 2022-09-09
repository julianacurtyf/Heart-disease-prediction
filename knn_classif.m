function predicted = knn_classif(data_tr, data_te, num_neighbors)

    % This function creates a KNN classifier with the number of neighbors
    % specified on the input, and returns the predicted value obtained in 
    % the model for the test set.

    model = fitcknn(data_tr.X', data_tr.y, 'NumNeighbors',num_neighbors);

    predicted = predict(model, data_te.X');


end