classdef AbstractLayers < matlab.mixin.Copyable
    
    % Constant abstract of matlab has bug, so we don't use it
    % issue: https://www.mathworks.com/matlabcentral/newsreader/view_thread/266378
    %properties(Abstract, Constant)
    %    %struct info
    %    type;
    %    is_final_layer;     
    %end
   
    methods (Abstract)
        out = Forward(layer,in)        
        prev_derr = Backward(layer,derr)        
        Update(layer,learn_rate)
        ClearCache(layer)
    end
    
end