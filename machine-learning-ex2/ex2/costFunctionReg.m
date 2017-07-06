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

function hypothesis = hypothesisFunction(theta, X,m)
        hypothesis = sigmoid(X*theta);
end
regTerm = sum(theta(2:length(theta)).^2)*lambda/(2*m);

J=regTerm + sum((-y).*log(hypothesisFunction(theta,X,m))-(1-y).*log(1-hypothesisFunction(theta,X,m)))/m;
tempGrad = X'*(hypothesisFunction(theta, X,m)-y)/m+lambda*theta/m;
offset = zeros(size(theta));
offset(1)= -lambda*theta(1)/m;

grad = tempGrad + offset;




% =============================================================

end
