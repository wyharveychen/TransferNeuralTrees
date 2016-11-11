classdef SoftmaxLayers <handle
    properties          
        %cache parameter of updating
        out;
        yid;
        
        label_list;
        label_map;
    end
    properties(Constant)
        %struct info
        type = 'Softmax';
        is_final_layer = 1;     
    end
    
    methods
        function layer = SoftmaxLayers(label_list)
             layer.label_list = label_list; 
             layer.label_map = containers.Map(label_list,1:length(label_list));                 
        end
        function out = Forward(layer,in) %in: input, out: layer output 
             out = layer.SoftMax(in);
             layer.out = out;
        end
        function prev_derr = Backward(layer,y_gt) %derr: derr/ds
             layer.yid = layer.LabelEmbedding(y_gt);
             prev_derr = (layer.prob - layer.yid);             
        end
        function Update(layer)               
        end
        
        function out = LabelEmbedding(layer,y)
             yid = cell2mat(values(layer.label_map,mat2cell(y', ones(size(y)))))';
             out = full(sparse(yid,1:length(yid),ones(length(yid),1)));
        end
        function cost = TotalError(layer)                                   
            cost = 0.5*mean(sum((layer.yid - layer.out).^2,1));
        end
        function predicted_label = Predict(layer)
            [~,max_id] = max(layer.out,[],1);
            predicted_label = layer.label_list(max_id);
        end
        function ClearCache(layer)
            layer.out = [];
            layer.yid = [];
        end

    end
    methods(Static)
        function y = SoftMax(x)
            z = exp(bsxfun(@minus, x, max(x)));
            y = bsxfun(@rdivide, z, sum(z));
        end      
    end
end