function [out,varargout] = label_embedding(y,label_list,label_map)
    nout = max(nargout,1) - 1;
    if(nargin<2)
        label_list = unique(y);
        label_map = containers.Map(label_list,1:length(label_list));                 
    end
    yid = cell2mat(values(label_map,mat2cell(y', ones(size(y)))))';

    out = full(sparse(yid,1:length(yid),ones(length(yid),1)));
    if(size(out,1) ~= length(label_list))
        out(length(label_list),1) = 0; %for filling missing num
    end
    if(nout > 0)
        varargout{1} = label_list;
        varargout{2} = label_map;
    end
end