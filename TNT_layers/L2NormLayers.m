classdef L2NormLayers <handle
    properties  
        type = 'L2_norm';
        weights; %[pre,]
        is_final_layer = 0;
        
        %cache parameter of updating
        x_norm;         
        out;
    end
    
    methods
        function layer = L2NormLayers()
        end
        function out = Forward(layer,in) %in: input, next_in: layer output (next layer input)       
             out = layer.L2Norm(in);
             layer.x_norm = sqrt(sum(in.^2,1));
             layer.out = out;
        end
        function prev_derr = Backward(layer,derr) %derr: derr/ds
             prev_derr = (derr.* layer.L2NormDiff(layer.out,layer.x_norm));         
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