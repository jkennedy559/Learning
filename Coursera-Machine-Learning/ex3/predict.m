function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% First_example = X(1,:), size = 1 row with 401 columns
% sigmoid(First_example(1x401) * theta1'(401x25)) = A2_weights (1X25)
% A2_weights = [1, A2_weights]
% sigmoid(A2_weights(1x26) * theta2'(26x10)) = predictions (1X10)
% predict_as_index = max(predictions)
% p(1) = predict_as_index

for iter = 1:m
    A2_weights = sigmoid(X(iter,:) * Theta1');
    A2_weights = [1, A2_weights];
    A3_weights = sigmoid(A2_weights * Theta2');
    [M, I] = max(A3_weights);
    p(iter) = I;
end


% =========================================================================


end
