classdef Util < handle
   methods(Static)
       function value = Initialize(arg,field,default)
            if(isfield(arg,field))
                value = eval(sprintf('arg.%s;',field));
            else
                value = default;
            end            
        end
   end 
end
