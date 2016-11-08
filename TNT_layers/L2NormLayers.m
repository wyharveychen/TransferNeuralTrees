classdef L2NormLayers <handle
    properties  
        type = 'L2_norm';
        weights; %[pre,]
        is_final_layer = 0;
        
        %cache parameter of updating
        x_norm;         
        next_in;
    end
    
    methods
        function layer = L2NormLayers()
        end
        function next_in = Forward(layer,in) %in: input, next_in: layer output (next layer input)       
             next_in = layer.L2Norm(in);
             layer.x_norm = sqrt(sum(in.^2,1));
             layer.next_in = next_in;
        end
        function prev_derr = Backward(layer,derr) %derr: derr/ds
             prev_derr = (derr.* layer.L2NormDiff(layer.next_in,layer.x_norm));         
        end
        function Update(layer,lr)               
        end
        function ClearCache(layer)
            layer.x_norm = [];
        end

    end
    methods(Static)
        function y = L2Norm(x)
            y = normc(x);
        end      
        function y = L2NormDiff(z,norm_x)
            y = bsxfun(@rdivide,(1-z.^2),norm_x);
        end
    end
end