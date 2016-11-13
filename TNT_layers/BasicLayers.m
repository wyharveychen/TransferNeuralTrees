classdef BasicLayers < AbstractLayers
    properties
        %
        weights; %[pre,]
        BasicNeuron;
        BasicNeuronDiff;
        
        %cache parameter of updating
        in;
        out; %better to be modified to out
        grad;
        
        optim;
    end
    properties(Constant)
        %struct info
        type = 'Basic';
        is_final_layer = 0;
    end
    
    methods
        function layer = BasicLayers(indim,outdim)
            layer.weights = [(rand(indim, outdim) - 0.5); zeros(1,outdim)];
            layer.BasicNeuron = @BasicLayers.Tanh;
            layer.BasicNeuronDiff = @BasicLayers.TanhDiff;
            layer.optim = Optimizer();
        end
        function out = Forward(layer,in) %in: input, out: layer output (next layer input)
            in_num = size(in,2);
            layer.in = [in;ones(1, in_num)];
            out = layer.BasicNeuron(layer.weights' * layer.in);
            layer.out = out;
        end
        function prev_derr = Backward(layer,derr) %derr: derr/ds
            %in --*w--> s --activate--> out
            in_num = size(derr,2);
            %In common methemitical definition, we'll use derr/ds of next
            %layer to calculate derr/ds in this layer, that is:
            %prev_derr = (next_layer.weights * derr).* layer.BasicNeuronDiff(layer.out);
            %However, this require w in next layer, which is not modulized
            %Thus, we first do the multiplication of w in this layer to get derr/din  == prev derr/dout of and
            %pass backward
            prev_derr = (layer.weights * (derr.* layer.BasicNeuronDiff(layer.out)) );
            %derr is now next_derr/ds * next_w = derr/dout, to become derr/s in this layer, only need to multiply by dout/ds
            layer.grad = (layer.in*(derr.* layer.BasicNeuronDiff(layer.out))')/in_num;
            prev_derr = prev_derr(1:end-1,:);
        end
        function Update(layer,lambda)
            layer.weights = layer.optim.WeightUpdate(layer.weights,layer.grad,lambda);
        end
        function ClearCache(layer)
            layer.in = [];
            layer.out = [];
            layer.grad = [];
        end
    end
    methods(Static)
        %% for math
        function y = Tanh(x)
            y = tanh(x);
        end
        function y = TanhDiff(x)
            y = (1-x.^2);
        end
        function y = ReLU(x)
            y = max(x, 0);
        end
        function y = diffReLU(z)
            y = z > 0;
        end
        function y = Linear(x)
            y = x;
        end
        function y = diffLinear(z)
            y = 1;
        end
        function y = Bent(x)
            y = x.*(x>0) + 0.1*x.*(x<0);
        end
        function y = diffBent(z)
            y = (z>0) + 0.1*(z<0) + 0.55*(z==0);
        end
    end
end
