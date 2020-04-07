function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_choices = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
simga_choices = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
accuracy_container = zeros(64,3);
i = 1;

for iter1 = 1:8
    for iter2 = 1:8  
        
        C = C_choices(iter1);
        
        sigma = simga_choices(iter2);
        
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));

        predictions = svmPredict(model, Xval);
        
        accuracy = mean(double(predictions ~= yval));
        
        accuracy_container(i,:) = [C, sigma, accuracy];
        
        i = i + 1;       
    end
end

[M, I] = min(accuracy_container(:,3));

C = accuracy_container(I,1)
sigma = accuracy_container(I,2)

% =========================================================================

end
