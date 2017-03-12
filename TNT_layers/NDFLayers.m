classdef NDFLayers < AbstractLayers
    properties            
        %structure parameter
        tree_num = 20;
        tree_depth = 7;                
        tree_dim_ratio = 0.2; %the ratio of total dimension are as tree input
        indim;
        outdim;
      
        %weights/indexs
        inidx; theta; route; pi;
        
        %cache parameter of updating        
        in; d; mu; p; out;
        yid; P; derr; prev_derr; prev_derr_merged;
        grad; G; D; m;
                          
        %for prediction
        label_list; label_map;
        p_norm;
           
        optim;
    end
    properties(Constant)
        %struct info
        type = 'NDF';
        is_final_layer = 1;     
        
        %update parameter                  
        treethres_alpha = 5;%5              
        leafprob_bias = 0.001;
        %pi_decay_rate = 1; %this is to handle the problem that how to update w_pi with batch input

    end
    
    methods
        %% For Initialize
        function layer = NDFLayers(indim,label_list)
            layer.indim = indim;             
            %initialize dimension of tree input                         
            
            randseedmap  =  NDFLayers.InitInidx(indim, layer.tree_dim_ratio,layer.tree_num);
            layer.inidx  =  reshape(randseedmap,[ceil(layer.tree_dim_ratio*indim),layer.tree_num]);
            
            layer.theta  = cellfun( @(i) (rand(layer.tree_dim_ratio*indim+1, 2^layer.tree_depth-1 ) - 0.5) ,  cell(layer.tree_num,1),'UniformOutput',0 );
            layer.route  = cellfun( @(i) NDFLayers.TreeWeightTable(layer.tree_depth)                       ,  cell(layer.tree_num,1),'UniformOutput',0 );  

            layer.InitializePILayer(label_list);

            layer.optim = Optimizer();
        end
                
        function InitializePILayer(layer,label_list)
            layer.label_list = label_list; 
            layer.outdim = length(layer.label_list);
            layer.label_map  = containers.Map(layer.label_list,1:layer.outdim);
            layer.pi         = cellfun( @(i) ones(2^layer.tree_depth, layer.outdim)/layer.outdim, cell(layer.tree_num,1) ,'UniformOutput',0);               
        end
   
        %% For Forwarding        
        function out = Forward(layer,in) %in: input, out: layer output (next layer input)             
             layer.in = NDFLayers.DimensionSelection(in, layer.inidx);  
             layer.d  = cellfun(@NDFLayers.ThetaLayer, layer.in,   layer.theta ,'UniformOutput',0);  
             layer.mu = cellfun(@NDFLayers.RouteLayer, layer.d ,   layer.route ,'UniformOutput',0);
             layer.p  = cellfun(@NDFLayers.PILayer   , layer.mu,   layer.pi    ,'UniformOutput',0);             
             out  = mean(cat(3,layer.p{:}),3);
             layer.out = out;
        end
    
        %% For Backward
        function prev_derr_merged = Backward(layer,gt_y) %derr: derr/ds               
            if(~isempty(gt_y))
                %prediction loss
                layer.yid  =  layer.LabelMapping(gt_y);
                
                layer.P         = Util.CellFunwConst(@NDFLayers.PILayerDiff,    layer.mu,   layer.yid,  layer.pi);
                layer.derr      = Util.CellFunwConst(@NDFLayers.RouteLayerDiff, layer.d,    layer.P,    layer.route);
                layer.prev_derr = Util.CellFunwConst(@NDFLayers.ThetaLayerDiff, layer.derr, layer.theta);
                
                prev_derr_merged = NDFLayers.DimensionMerge(layer.prev_derr, layer.inidx, layer.indim);             
                
                layer.grad      = Util.CellFunwConst(@NDFLayers.GradientDeri, layer.derr, layer.in);
            else
                %embedding loss
                layer.p_norm  = mean(layer.out,2);
                layer.grad    = 0;
                prev_derr_merged = 0;
                for yid = 1:length(layer.label_list)              
                    layer.P    =  cellfun(@(mu,pi) NDFLayers.PILayerDiff(mu, yid*ones(1,size(layer.out,2)) ,pi),                    layer.mu  ,                 layer.pi      ,'UniformOutput',0);
                    layer.derr =  cellfun(@(d,P,w_r) NDFLayers.RouteLayerEmlossDiff(d,P,w_r, layer.out(yid,:) ,layer.tree_num),    layer.d   ,     layer.P,    layer.route   ,'UniformOutput',0);
                    layer.prev_derr = cellfun(@NDFLayers.ThetaLayerDiff,                                                            layer.derr,                 layer.theta   ,'UniformOutput',0);
                                        
                    layer.grad = Util.CellFunwConst(@(g,derr,in) (g + NDFLayers.GradientDeri(derr,in) ), layer.grad, layer.derr, layer.in);
                    prev_derr_merged = prev_derr_merged + NDFLayers.DimensionMerge(layer.prev_derr, layer.inidx, layer.indim);
                end 
            end    
        end         

        %% For Update
        function Update(layer, lambda) 
            layer.theta = layer.optim.WeightUpdate(layer.theta,layer.grad,lambda);           
            layer.pi    = cellfun(@(p,w_pi) NDFLayers.UpdatePILayer(p,w_pi,layer.yid, NDFLayers.leafprob_bias ), layer.P, layer.pi,'UniformOutput',0);
        end

        function UpdatePIOnly(layer,ground_truth,bias)
             if(~exist('bias','var'))
                 bias = NDFLayers.leafprob_bias;
             end
             layer.yid  =  layer.LabelMapping(ground_truth);
             yid_expand  = cell(layer.tree_num,1);
             yid_expand(:) = {layer.yid};
             layer.P    =  cellfun(@NDFLayers.PILayerDiff       , layer.mu  ,yid_expand , layer.pi      ,'UniformOutput',0);
             layer.pi    = cellfun(@(p,w_pi) NDFLayers.UpdatePILayer(p,w_pi,layer.yid,bias ), layer.P, layer.pi,'UniformOutput',0);             
        end
        
        function UpdatePIDirectly(layer,w_pi)            
             layer.pi    = w_pi;
        end
        function UpdatePIDirectlyOneTree(layer,w_pi,tree_id)            
             layer.pi{tree_id}    = w_pi;
        end
        
        %% For Report                
        function cost = TotalError(layer)                         
            out_error = cellfun(@(p) Util.SparseSelect2D(p,layer.yid,1:length(layer.yid)), layer.p,'UniformOutput',0);
            out_error = cellfun(@(err) sum(log(err)),out_error,'UniformOutput',0);
            cost = -sum(cat(2,out_error{:}));           
        end
        
        function predicted_label = Predict(layer)
            [~,max_id] = max(layer.out,[],1);
            predicted_label = layer.label_list(max_id);
        end

        function RecordForestNorm(layer,gt_y)
            %need to do prediction first
            layer.yid  =  layer.LabelMapping(gt_y);           
            layer.p_norm = sum(layer.out,2)./ hist(layer.yid, 1:length(layer.label_list))';
            layer.p_norm = layer.p_norm/sum(layer.p_norm);
        end
                
        function variance = ForestVariance(layer)
            p_all = cellfun(@(p) p(:), layer.p,'UniformOutput',0); 
            variance = sum(reshape(var(cat(2,p_all{:}),0,2), size(layer.p{1})),1);            
        end
            
        function loss = EmbeddingLoss(layer)
            loss = -sum((bsxfun(@rdivide,(layer.out).^2, layer.p_norm)),1);
        end
                
        %% For Utility
        function yid = LabelMapping(layer,y)
             yid = cell2mat(values(layer.label_map,mat2cell(y', ones(size(y)))))';
        end
                
        function ClearCache(layer)
            layer.in = []; layer.d = []; layer.mu = []; layer.p = []; layer.out = [];
            layer.yid = []; layer.P = []; layer.derr = []; layer.prev_derr = []; layer.prev_derr_merged = [];
            layer.grad = [];     
        end
                                
    end
    
    methods(Static)
        %% For initialization
        function randseedmap = InitInidx(indim, inratio,tree_num)
            randseedmap = [];
            for i = 1:ceil(tree_num*inratio)
                randseedmap = [randseedmap, randperm(indim)];
            end
        end
        function w = TreeWeightTable(depth)
            w = [];
            for i = 1:depth
                w = [w;kron(eye(2^(i-1)),[ones(1,2^(depth-i)) , -ones(1,2^(depth-i))])];                
            end               
        end
        
        %% For Forward propagation
        function selected_input = DimensionSelection(input,selected_dims)
            [indim,num] = size(selected_dims);            
            selected_input = mat2cell(input(selected_dims(:),:), indim*ones(num,1));
            selected_input = cellfun(@(i) [i;ones(1,size(input,2))],  selected_input ,'UniformOutput',0 );
        end 
        function d = ThetaLayer(x,w_th)            
            route_bias = 1e-6;           
            d = min(max( NDFLayers.Sigmoid(w_th'*x,NDFLayers.treethres_alpha),route_bias),1-route_bias);
        end
        function mu = RouteLayer(d,w_r)                                
            mu = exp((w_r'>0)*log(d) + (w_r'<0)*log(1-d));           
        end
        function p = PILayer(mu,w_pi)            
            p = w_pi'*mu;              
        end
        
        %% For Back propagation 
        function P = PILayerDiff(mu,out,w_pi)
            P = mu.* w_pi(:,out); 
        end        
        function derr = RouteLayerDiff(d,P,w_r)    
            P = bsxfun(@rdivide,P,sum(P,1)); %sum(P,1) = Pt[y|x,theta,pi] in eq 10
            A = [(w_r~=0)*P;P];
            derr =(d.*A(3:2:end,:) - (1-d).*A(2:2:end,:));         
        end
        function derr = RouteLayerEmlossDiff(d,P,w_r,out,tree_num)
            f_norm = mean(out,2); %out = forest prediction = Pf[y~|x,theta,pi],f_norm = Pf[y~|theta,pi] 
            P =  2/tree_num *bsxfun(@times,P,out)/f_norm;
            A = [(w_r~=0)*P;P];
            derr =(d.*A(3:2:end,:) - (1-d).*A(2:2:end,:));             
        end       
        function prev_derr = ThetaLayerDiff(derr,w_th)
            prev_derr = w_th * derr; %derr/dx(input) of this layer
        end        
        function prev_derr_merged = DimensionMerge(prev_derr,selected_dims,dim)    
            selected_dims = selected_dims(:);
            prev_derr = cellfun(@(D) D(1:end-1,:),prev_derr,'UniformOutput',0 );
            prev_derr_all = cat(1,prev_derr{:});
            prev_derr_merged = zeros(dim,size(prev_derr_all,2));
            for i = 1:length(selected_dims)
                idx = selected_dims(i);
                prev_derr_merged(idx,:) = prev_derr_merged(idx,:)+prev_derr_all(i,:);
            end
        end        
        function grad = GradientDeri(derr,in) 
            grad = (in * derr') / size(in, 2);
        end
     
        %% For Update
        function w_pi = UpdatePILayer(P,prev_w_pi,out,bias)
            
            w_pi = zeros(size(prev_w_pi));
            P = bsxfun(@rdivide,P,sum(P,1));
            for i = unique(out)                
                w_pi(:,i) = sum(P(:,out == i),2);               
            end
            %w_pi = w_pi+ NDFLayers.pi_decay_rate*prev_w_pi +NDFLayers.leafprob_bias*ones(size(w_pi)); %prev
            w_pi = w_pi+ bias*ones(size(w_pi)); %prev
            w_pi = bsxfun(@rdivide,w_pi,sum(w_pi,2));          
        end           
        
        %% For math
        function y = Sigmoid(x,a)          
            if nargin < 2
                a = 1;
            end
            y = (1+exp(-a*x)).^(-1);
		end
		function y = SigmoidDiff(x,a)	            
            if nargin < 2
                a = 1;
            end
			y = a*x .* (1-x);         
        end

 
    end
end