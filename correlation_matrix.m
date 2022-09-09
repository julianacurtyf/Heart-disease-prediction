function  correlation_matrix = correlation_matrix(data)
    %%
    % This function calculates the correlation between the features and plots 
    % the matrix obtained.
    % Then, using the threshold that was given as an input, it returns the
    % indexes of the features that have a correlation smaller than the
    % threshold. 

    figure
    correlation_matrix = corrcoef(data.X');
    heatmap(correlation_matrix,'Colormap',autumn)
    ax = gca;
    ax.XData = {'BMI5', 'Smoking',	'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',	'Race',	'Diabetic',	'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma'};
    ax.YData = {'BMI5', 'Smoking',	'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',	'Race',	'Diabetic',	'PhysicalActivity',	'GenHealth', 'SleepTime', 'Asthma'};

end
