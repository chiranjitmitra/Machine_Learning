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

n = size(X, 2);

for i = 1:m
  z = 0;
  for j = 1:n
    z = z + theta(j) * X(i, j);
  end
  h = sigmoid(z);
  J = J + (- y(i) * log(h) - (1 - y(i)) * log(1 - h));
end
J = J / m;
for j = 2:n
  J = J + (lambda / (2 * m)) * theta(j)^2;
end

grad = zeros(n, 1);
for j = 1:n
  for i = 1:m
    z = 0;
    for k = 1:n
      z = z + theta(k) * X(i, k);
    end
    h = sigmoid(z);
    grad(j) = grad(j) + (h - y(i)) * X(i, j);
  end
end
grad = grad / m;
for j = 2:n
  grad(j) = grad(j) + (lambda / m) * theta(j);
end

% =============================================================

end
