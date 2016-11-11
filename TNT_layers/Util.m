classdef Util < handle
    methods(Static)
        function value = Initialize(arg,field,default)
            if(isfield(arg,field))
                value = eval(sprintf('arg.%s;',field));
            else
                value = default;
            end            
        end

        function cell_output = CellFunwConst(varargin)
            func = varargin{1};
            varargin(1) = []; %delete function variable;
            assert(isa(func,'function_handle'),'first argument in cellfun is not a function');

            arg_iscell_s = cellfun(@(varg) isa(varg,'cell'), varargin);

            %example:
            %'@(arg1,arg2,arg4...) func(arg1,arg2,varargin{3},arg4),varargin{1},varargin{2},varargin{4} "UniformOutput", 0'

            anmfunc_argin  = [];
            func_argin = [];
            cellfun_argin = [];
            for id = 1:length(arg_iscell_s)
                if(arg_iscell_s(id))
                    anmfunc_argin  = [ anmfunc_argin, sprintf('arg%d,',id)];
                    func_argin  = [ func_argin, sprintf('arg%d,',id)];
                    cellfun_argin = [cellfun_argin, sprintf('varargin{%d},',id)];
                else
                    func_argin  = [ func_argin, sprintf('varargin{%d},',id)];
                end
            end
            anmfunc_argin(end) = [];
            func_argin(end) = [];
            cellfun_argin(end) =[]; 
                        
            quotemark = sprintf('\''');
            uni_arg = strrep('"UniformOutput", 0', '"', quotemark);
            
            cmd = sprintf('cellfun(@(%s) func(%s), %s, %s)', anmfunc_argin,func_argin,cellfun_argin,uni_arg);
            cell_output = eval(cmd);
        end
        
        function sub_cell_arr= CellSubset(cell_arr,col_id)
            sub_cell_arr = cellfun(@(M) M(:,col_id), cell_arr, 'UniformOutput', 0);
        end
        
        function t_stamp = TimeStamp()
            t = clock;
            t_stamp = sprintf('%04d%02d%02d%02d%02d',t(1),t(2),t(3),t(4),t(5));
        end
    end 
end
