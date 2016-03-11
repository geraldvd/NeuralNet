classdef c_neuralnet < handle
    properties
        m_layers;
        m_error;
    end
    
    methods
        function obj = c_neuralnet(n_inputs, hiddenLayers, n_outputs)
            % Initialize layers
            obj.m_layers = cell(2 + length(hiddenLayers), 1);
            
            % Add cells (with placeholders for neurons) to layers,
            % including biases
            obj.m_layers{1} = cell(n_inputs+1, 1);
            for i = 1:length(hiddenLayers)
                obj.m_layers{i + 1} = cell(hiddenLayers(i)+1, 1);
            end
            obj.m_layers{end} = cell(n_outputs, 1);
            
            % Fill layers with neurons
            for li = 1:length(obj.m_layers)
                % Determine number of outputs for neurons in m_layers{li}
                if li == length(obj.m_layers)
                    n_neuronOutputs = 0;
                else
                    n_neuronOutputs = length(obj.m_layers{li+1});
                end
                
                % Add neurons
                for ni = 1:length(obj.m_layers{li})
                    obj.m_layers{li}{ni} = c_neuron(n_neuronOutputs, ni);
                end
                
                % Add bias neuron - Note: bias in outputlayer not used
                obj.m_layers{li}{end}.m_output = 1;
            end
        end
        
        function obj = feedForward(obj, input)
            % Check input
            assert(length(input) == length(obj.m_layers{1})-1);
            
            % Assign inputs to neuron value (=neuron output)
            for i = 1:length(input)
                obj.m_layers{1}{i}.m_output = input(i);
            end
            
            % Propagate
            for li = 2:length(obj.m_layers)
                for ni = 1:length(obj.m_layers{li})
                    obj.m_layers{li}{ni}.feedForward(obj.m_layers{li-1});
                end
            end
            
        end
        
        function obj = backProp(obj, target)
            % Check input
            assert(length(target) == length(obj.m_layers{end}));
            
            % Calculate network error
            obj.m_error = 0;
            for ni = 1:length(obj.m_layers{end})
                obj.m_error = obj.m_error + (target(ni) - obj.m_layers{end}{ni}.m_output) ^ 2;
            end
            obj.m_error = sqrt(obj.m_error / length(obj.m_layers{end}));
            
            % Calculate output layer gradients
            for ni = 1:length(obj.m_layers{end})
                obj.m_layers{end}{ni}.calcOutputGradients(target(ni));
            end
            
            % Calculate hidden layer gradients
            for li = length(obj.m_layers) - 1:-1:2
                for ni = 1:length(obj.m_layers{li})
                    obj.m_layers{li}{ni}.calcHiddenGradients(obj.m_layers{li+1});
                end
            end
            
            % Update weights
            for li = length(obj.m_layers):-1:2
                for ni = 1:length(obj.m_layers{li})
                    obj.m_layers{li}{ni}.updateInputWeights(obj.m_layers{li-1});
                end
            end
        end
    end    
end
