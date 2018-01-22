function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hx_y = X * theta - y;
j_hx = sum(hx_y .^ 2);
j_lambda = lambda * sum(theta(2:end) .^ 2);
J = (j_hx + j_lambda) / (2*m);

new_theta = theta;
new_theta(1) = 0;
hx_y_xj = repmat(hx_y, [1, size(X,2)]) .* X;
sum_hx = sum(hx_y_xj, 1);
#size(sum_hx)
grad = (sum_hx(:) .+ lambda * new_theta) / m;
#size(d_theta)
#d_theta(1,:) = 

% =========================================================================

grad = grad(:);

end
