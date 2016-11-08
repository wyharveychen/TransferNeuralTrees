classdef NN_layers < matlab.mixin.Copyable
    properties  
        parameters;
        isFinalLayer;
    end
    
    methods
        function out = Forward(in)
        end
        function [prev_derr,grad] = Backward(derr)
        end
        function Update(grad)        
        end
    end
    
end