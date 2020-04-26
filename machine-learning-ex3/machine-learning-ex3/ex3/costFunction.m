function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = size(X,1);
J = (-1/m)*(y'*log(sigmoid(X*theta)) + (1-y')*log(1-sigmoid(X*theta)));
grad = (1/m)*((sigmoid(X*theta)-y)'*X)';


end
