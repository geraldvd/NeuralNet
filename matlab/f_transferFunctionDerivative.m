function y = f_transferFunctionDerivative(x)
y = 1 - tanh(x) * tanh(x); % TODO: use y = 1-x*x (fast approximation)
end