function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1: 
X = [ones(m, 1) X]; % Add ones to the X data matrix

a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
hoX = a3;

for iter = 1:num_labels
    yiter_ = y == iter;
    vector_ = -(yiter_.*log(hoX(:,iter))) - (1-yiter_).*log(1-hoX(:,iter));
    iter_sum_ = sum(vector_);
    J = J + iter_sum_;
end

J = J/m;

o1 = sum(Theta1(:,2:end).^2, 'all');
o2 = sum(Theta2(:,2:end).^2, 'all');
J = J + lambda*((o1 + o2)/(2*m));

% Part 2:

for iter2 = 1:m
    a1 = X(iter2,:); %1x401
    z2 = a1 * Theta1';  %1x25
    a2 = sigmoid(z2); %1x25
    a2 = [1 a2]; %1x26
    z3 = a2 * Theta2'; %1x10
    a3 = sigmoid(z3); %1x10
    
    
    y1 = y(iter2); % target scalar 1-10
    yk = zeros(num_labels,1);
    for iter3 = 1:num_labels
        yk(iter3) = y1 == iter3;  
    end
    
    d_3 = a3' - yk;
    d_2 = (Theta2'*d_3) .* sigmoidGradient([1 z2]'); %sigmoidGradient([1 z2]') a2'
    
    
    Theta2_grad = Theta2_grad  + (d_3*a2); %10x1 * 1*26
    Theta1_grad = Theta1_grad  + (d_2(2:end)*a1); %25x1 * 1x401 
    
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

size(Theta1_grad); % 25x401 Don't regularise column  1 
size(Theta2_grad); % 10x26 Don't regularise column  1 


Theta2_pen = Theta2;
Theta2_pen(:,1) = 0;
Theta2_pen = Theta2_pen*(lambda/m);

Theta2_grad = Theta2_grad + Theta2_pen;

Theta1_pen = Theta1;
Theta1_pen(:,1) = 0;
Theta1_pen = Theta1_pen*(lambda/m);

Theta1_grad = Theta1_grad + Theta1_pen;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
