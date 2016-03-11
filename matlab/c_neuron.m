classdef c_neuron < handle
    %% Neuron Class
    % Used in combination with c_neuralnet class. Each neuron keeps track
    % of it's output and output weights.

    properties
        m_myIndex;
        m_output;
        m_outputWeights;
        m_gradient;
        
        eta = 0.15;
        alpha = 0.5;
    end
    
    methods
        function obj = c_neuron(n_outputs, myIndex)
            % Set index (=position) in layer
            obj.m_myIndex = myIndex;
            
            % Initialize weight
            obj.m_outputWeights = cell(n_outputs, 1);
            for i = 1:n_outputs
                obj.m_outputWeights{i} = struct('weight', rand, 'deltaWeight', 0);
            end
            
        end
        
        function obj = feedForward(obj, prevLayer)
            s = 0;
            for i = 1:length(prevLayer)
                % prevLayer{i} is of type c_neuron!
                s = s + prevLayer{i}.m_output * prevLayer{i}.m_outputWeights{obj.m_myIndex}.weight;
            end
            
            obj.m_output = f_transferFunction(s);
        end
        
        function obj = calcHiddenGradients(obj, nextLayer)
            % Sum error contributions of nextLayer (for backprop)
            s = 0;
            for i = 1:length(nextLayer)
                s = s + obj.m_outputWeights{i}.weight * nextLayer{i}.m_gradient;
            end
            obj.m_gradient = s * f_transferFunctionDerivative(obj.m_output);
        end
        
        function obj = calcOutputGradients(obj, target)
            obj.m_gradient = (target - obj.m_output) * f_transferFunctionDerivative(obj.m_output);
        end
        
        function obj = updateInputWeights(obj, prevLayer)
            for i = 1:length(prevLayer)
                oldDeltaWeight = prevLayer{i}.m_outputWeights{obj.m_myIndex}.deltaWeight;
                newDeltaWeight = obj.eta * prevLayer{i}.m_output * obj.m_gradient ...
                    + obj.alpha * oldDeltaWeight;
                
                prevLayer{i}.m_outputWeights{obj.m_myIndex}.deltaWeight = newDeltaWeight;
                prevLayer{i}.m_outputWeights{obj.m_myIndex}.weight = ...
                    prevLayer{i}.m_outputWeights{obj.m_myIndex}.weight + newDeltaWeight;
            end
        end
    end    
end
