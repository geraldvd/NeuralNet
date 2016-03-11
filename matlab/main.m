clear all, close all;

%% Initialize NN
nn = c_neuralnet(1, [4], 1);

%% Training
xt = rand(1000, 1);
yt = sin(2*pi * xt);

for j = 1:20
    for i = 1:length(xt)
        nn.feedForward(xt(i));
        nn.backProp(yt(i));
    end
end

% Plot
figure;
scatter(xt, yt);
title('Training');

%% Testing
x = [0:0.1:1]';
y = sin(2*pi * x);

y_nn = zeros(size(x));
for i = 1:length(x)
    nn.feedForward(x(i));
    y_nn(i) = nn.m_layers{end}{1}.m_output;
end

% Plot
figure;
plot(x, y, x, y_nn);
legend('Validation', 'NN output');
title('Testing');