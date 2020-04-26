function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
thetaDiff = theta;
thetaDiff(1,:) = [];
[J, grad] = costFunction(theta, X, y);
regularization_term = (lambda/(2*m))*(thetaDiff'*thetaDiff);
J = J + regularization_term;
thetaDiff = theta;
thetaDiff(1) = 0;
grad = grad + thetaDiff*(lambda/m);

% =============================================================

end
