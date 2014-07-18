function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

first = 0;
for i=1:m
	first += - y(i) * log ( sigmoid( X(i,:) * theta ) ) - ( 1 - y(i) ) * log ( 1 - sigmoid( X(i,:) * theta ) );
end;
first = first / m;

second = 0;
% !!!! don't include theta(1) in the cost function.
for i=2:size(theta,1)
	second += theta(i)^2;
end;
second = second * lambda / (2 * m);

J = first + second;

temp = 0;
for j=1:size(theta,1),
	temp = 0;
	for i=1:m,
		temp += ( sigmoid( X(i, :) * theta ) - y(i) ) * X(i, j);
	end;
	grad(j) = temp / m;
	if j != 1
		grad(j) += lambda * theta(j) / m;
end;

% =============================================================

end
