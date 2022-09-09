function show_PCA(data, type)

    % This function shows the PCA's graphs to analyse the best number of
    % dimensions to reduce the data. The type can be 'scree' or 'cumsum', 
    % for the scree plot with the Kaiser's criteria or the graph of the
    % accumulated variance of the features

    [~, eigen_values] = eig(cov(data.X'));
    eigen_values = sort(sum(eigen_values),'descend');

    if strcmp(type, 'scree')
      
        figure; 
        plot(eigen_values); 
        xlim([1,size(eigen_values, 2)]); 

    elseif strcmp(type, 'cumsum')

        figure; 
        plot(cumsum(eigen_values)./sum(eigen_values));
        xlim([1,size(eigen_values, 2)]); 

    end
end

