classdef NDFLayers < matlab.mixin.Copyable
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
        
        %for optimization
        learn_rate;
        learn_rate_Adam = 5e-2; 
        learn_rate_RMSprop = 5e-3; 
        learn_rate_adadelta = 5; 
        learn_rate_adagrad = 5e-2; 
        learn_rate_noadagrad = 1; 
        ada_decay_rate = 0.9;
            
        optimizer;
    end
    properties(Constant)
        %struct info
        type = 'NDF';
        is_final_layer = 1;     
        
        %update parameter                  
        treethres_alpha = 5;%5              
        leafprob_bias = 0.001;%0.001
        pi_decay_rate = 1; %this is to handle the problem that how to update w_pi with batch input
        regrate_max = 0.1;%0.1
        regrate_lambda = 0.04;%0.04
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
            layer.G      = cellfun( @(i) (zeros(layer.tree_dim_ratio*indim+1, 2^layer.tree_depth-1)      ) ,  cell(layer.tree_num,1),'UniformOutput',0 );
            layer.D      = layer.G; 
            layer.m      = layer.G;
            
            layer.InitializePILayer(label_list);
            
            layer.SetOpimizer('RMSprop');
        end
                
        function InitializePILayer(layer,label_list)
            layer.label_list = label_list; 
            layer.outdim = length(layer.label_list);
            layer.label_map  = containers.Map(layer.label_list,1:layer.outdim);
            layer.pi         = cellfun( @(i) ones(2^layer.tree_depth, layer.outdim)/layer.outdim, cell(layer.tree_num,1) ,'UniformOutput',0);               
        end
        
        function SetOpimizer(layer,optimizer)
            layer.optimizer = optimizer;
            if(strcmp(layer.optimizer,'Adam'))
                layer.learn_rate = layer.learn_rate_Adam;
            elseif(strcmp(layer.optimizer,'RMSprop'))
                layer.learn_rate = layer.learn_rate_RMSprop;
            elseif(strcmp(layer.optimizer,'adadelta'))
                layer.learn_rate = layer.learn_rate_adadelta;
            elseif(strcmp(layer.optimizer,'adagrad'))
                layer.learn_rate = layer.learn_rate_adagrad;
            else    
                layer.learn_rate = layer.learn_rate_noadagrad;           
            end
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
        %{
        function out = ForwardOneTree(layer,in,tree_id) %in: input, out: layer output (next layer input)             
             layer.in{tree_id} = NDFLayers.DimensionSelectionOneTree(in, layer.inidx(:,tree_id));  
             layer.d{tree_id}  = NDFLayers.ThetaLayer(layer.in{tree_id},   layer.theta{tree_id} );  
             layer.mu{tree_id} = NDFLayers.RouteLayer( layer.d{tree_id} ,   layer.route{tree_id});
             layer.p{tree_id}  = NDFLayers.PILayer( layer.mu{tree_id},   layer.pi{tree_id}   );             
             out  = mean(cat(3,layer.p{:}),3);
             layer.out = out;
        end
        %}
        %% For Backward
        function prev_derr_merged = Backward(layer,gt_y) %derr: derr/ds                          
            if(~isempty(gt_y))
                %prediction loss
                layer.yid  =  layer.LabelMapping(gt_y);
                yid_expand  = cell(layer.tree_num,1);
                yid_expand(:) = {layer.yid};
                layer.P    =  cellfun(@NDFLayers.PILayerDiff       , layer.mu  ,yid_expand , layer.pi      ,'UniformOutput',0);
                layer.derr =  cellfun(@NDFLayers.RouteLayerDiff    , layer.d   ,layer.P    , layer.route   ,'UniformOutput',0);
                layer.prev_derr = cellfun(@NDFLayers.ThetaLayerDiff, layer.derr,             layer.theta   ,'UniformOutput',0);
                prev_derr_merged = NDFLayers.DimensionMerge(layer.prev_derr, layer.inidx, layer.indim);             
                layer.grad =  cellfun(@NDFLayers.GradientDeri, layer.derr, layer.in,'UniformOutput',0);     
            else
                %embedding loss
                layer.p_norm  = mean(layer.out,2);
                for yid = 1:length(layer.label_list)              
                    layer.P    =  cellfun(@(mu,pi) NDFLayers.PILayerDiff(mu, yid*ones(1,size(layer.out,2)) ,pi),                    layer.mu  ,                 layer.pi      ,'UniformOutput',0);
                    layer.derr =  cellfun(@(d,P,w_r) NDFLayers.RouteLayerEmlossDiff(d,P,w_r, layer.p_norm(yid) ,layer.tree_num),    layer.d   ,     layer.P,    layer.route   ,'UniformOutput',0);
                    layer.prev_derr = cellfun(@NDFLayers.ThetaLayerDiff,                                                            layer.derr,                 layer.theta   ,'UniformOutput',0);
                    if(yid == 1)
                        layer.grad =  cellfun(@NDFLayers.GradientDeri, layer.derr, layer.in,'UniformOutput',0);                   
                        prev_derr_merged = NDFLayers.DimensionMerge(layer.prev_derr, layer.inidx, layer.indim);
                    else
                        temp_grad  =  cellfun(@NDFLayers.GradientDeri, layer.derr, layer.in,'UniformOutput',0);
                        layer.grad =  cellfun(@plus, temp_grad, layer.grad ,'UniformOutput',0);
                        prev_derr_merged = prev_derr_merged + NDFLayers.DimensionMerge(layer.prev_derr, layer.inidx, layer.indim);
                    end
                end 
            end    
        end         
        %{        
        function prev_derr_merged = BackwardOneTree(layer,gt_y,tree_id)
             layer.yid  =  layer.LabelMapping(gt_y);
             layer.P{tree_id} = NDFLayers.PILayerDiff(layer.mu{tree_id},layer.yid ,layer.pi{tree_id});
             layer.derr{tree_id} = NDFLayers.RouteLayerDiff(layer.d{tree_id},layer.P{tree_id} ,layer.route{tree_id});
             layer.prev_derr{tree_id} = NDFLayers.ThetaLayerDiff(layer.derr{tree_id}, layer.theta{tree_id});
             prev_derr_merged = NDFLayers.DimensionMergeOneTree(layer.prev_derr{tree_id},layer.inidx(:,tree_id),layer.indim);
             layer.grad{tree_id} = NDFLayers.GradientDeri(layer.derr{tree_id}, layer.in{tree_id});
        end
        %}
        %% For Update
        function Update(layer, learn_rate_w)
            if( ~exist('learn_rate_w','var'))
                learn_rate = layer.learn_rate;
            else
                learn_rate = layer.learn_rate * learn_rate_w;
            end
            
            if(strcmp(layer.optimizer,'Adam'))
                adam_beta1 = 0.9;
                adam_beta2 = 0.999;
                layer.G = cellfun(@(g,G) (adam_beta2*G + (1-adam_beta2)*g.^2),layer.grad ,layer.G,'UniformOutput',0 ); 
                layer.m = cellfun(@(g,m) (adam_beta1*m + (1-adam_beta1)*m),layer.grad, layer.m,'UniformOutput',0);
                layer.theta = cellfun(@(w_th,m,G) (w_th - learn_rate*(m/(1-adam_beta1))./(sqrt(G/(1-adam_beta2))+1e-8)), layer.theta, layer.m, layer.G,'UniformOutput',0);    
            elseif(strcmp(layer.optimizer,'RMSprop'))
                layer.G = cellfun(@(g,G) (layer.ada_decay_rate*G + (1-layer.ada_decay_rate)*g.^2),layer.grad ,layer.G,'UniformOutput',0 ); 
                layer.theta = cellfun(@(w_th,grad,G) (w_th - learn_rate*grad./(sqrt(G)+1e-8)), layer.theta, layer.grad, layer.G,'UniformOutput',0);    
            elseif(strcmp(layer.optimizer,'adadelta'))
                layer.G = cellfun(@(g,G) (layer.ada_decay_rate*G + (1-layer.ada_decay_rate)*g.^2),layer.grad ,layer.G,'UniformOutput',0 ); 
                    X = cellfun(@(g,G,D) (-sqrt(D+0.1)./sqrt(G+0.1).*g),layer.grad ,layer.G, layer.D,'UniformOutput',0 );
                layer.D = cellfun(@(x,D) (layer.ada_decay_rate*D + (1-layer.ada_decay_rate)*x.^2),X ,layer.D,'UniformOutput',0 ); 
                layer.theta = cellfun(@(w_th,x) (w_th + learn_rate.*x), layer.theta, X,'UniformOutput',0);              
            elseif(strcmp(layer.optimizer,'adagrad'))
                layer.G = cellfun(@(g,G) (G + g.^2),layer.grad ,layer.G,'UniformOutput',0 ); 
                layer.theta = cellfun(@(w_th,grad,G) (w_th - learn_rate*grad./(sqrt(G)+1e-8)), layer.theta, layer.grad, layer.G,'UniformOutput',0);             
            else
                layer.theta = cellfun(@(w_th,grad) (w_th - learn_rate*grad), layer.theta, layer.grad,'UniformOutput',0);             
            end
            layer.pi    = cellfun(@(p,w_pi) NDFLayers.UpdatePILayer(p,w_pi,layer.yid, NDFLayers.leafprob_bias ), layer.P, layer.pi,'UniformOutput',0);
        end
        %{
        function UpdateOneTree(layer,tree_id)
            if(strcmp(layer.optimizer,'adadelta'))
                layer.G{tree_id} = layer.ada_decay_rate*layer.G{tree_id} + (1-layer.ada_decay_rate)*layer.grad{tree_id}.^2;
                    x = -sqrt(layer.D{tree_id}+0.1)./sqrt(layer.G{tree_id}+0.1).*layer.grad{tree_id};
                layer.D{tree_id} = layer.ada_decay_rate*layer.D{tree_id} + (1-layer.ada_decay_rate)*x.^2;
                layer.theta = layer.theta{tree_id} + layer.learn_rate.*x;     
            elseif(strcmp(layer.optimizer,'adagrad'))
                layer.G{tree_id} = layer.G{tree_id} + layer.grad{tree_id}.^2;
                layer.theta{tree_id} = layer.theta{tree_id} - layer.learn_rate*layer.grad{tree_id}./(sqrt(layer.G{tree_id} )+1e-8);          
            else
                layer.theta{tree_id} = layer.theta{tree_id} - layer.learn_rate*layer.grad{tree_id};          
            end
            layer.pi{tree_id}  =  NDFLayers.UpdatePILayer(layer.P{tree_id},layer.pi{tree_id},layer.yid, NDFLayers.leafprob_bias);
        end
        %}
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
            out_error = cellfun(@(p) NDFLayers.SparseSelect2D(p,layer.yid,1:length(layer.yid)), layer.p,'UniformOutput',0);
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
            %layer.p_norm = mean(layer.out,2);
        end
                
        function variance = ForestVariance(layer)
            p_all = cellfun(@(p) p(:), layer.p,'UniformOutput',0); 
            variance = sum(reshape(var(cat(2,p_all{:}),0,2), size(layer.p{1})),1);
            
        end
            
        function loss = EmbeddingLoss(layer)
            %loss = -sum((layer.out).^2,1);
            loss = -sum((bsxfun(@rdivide,(layer.out).^2, layer.p_norm)),1);
        end
        
        function loss = EmbeddingLossPartial(layer)
            %loss = -sum((layer.out).^2,1);
            k = 5; %top k tree
            tree_p = cat(3,layer.p{:});                 %combine cellfun
            maxp_in_tree = squeeze(max(tree_p,[],1));   %highest prob in each tree
            [~,sort_p_tid] = sort(maxp_in_tree,2,'descend'); %sort tree highest prob
            top_p_tid = sort_p_tid(:,1:k);          %select top k tree for each index
            data_num = size(top_p_tid,1);
            loss = zeros(1, data_num); 
            for id = 1:data_num 
                loss(id) = -sum(mean(squeeze(tree_p(:,id,top_p_tid(id,1:k))),2).^2);                
            end
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
                                
        %% For Visualization
        function ShowProjectedData(layer, gt_y, marker)
            if(~exist('marker'))
                marker = 'o';
            end
            layer.yid  =  layer.LabelMapping(gt_y);
               
            %scatter(x,y,marker size,color, marker type, [filled]);
            %defalut colormap: hsv, interval: 5 
            cmap = hsv;
            interval = 5;
            if(ismember(marker,'*+x.'))
                scatter(layer.in{3}(1,:),layer.in{3}(2,:),20,  cmap(interval*layer.yid,:),marker);
            else
                scatter(layer.in{3}(1,:),layer.in{3}(2,:),20,  cmap(interval*layer.yid,:),marker, 'filled');
            end
            
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
        function selected_input = DimensionSelectionOneTree(input,selected_dims)
            selected_input = input(selected_dims,:);
            selected_input = [selected_input;ones(1,size(input,2))];
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
            P = bsxfun(@rdivide,P,sum(P,1));
            A = [(w_r~=0)*P;P];
            derr =(d.*A(3:2:end,:) - (1-d).*A(2:2:end,:)); %derr/ds of this layer            
        end
        function derr = RouteLayerEmlossDiff(d,P,w_r,f_norm,tree_num)
            P =  2/tree_num *bsxfun(@times,P,sum(P,1))/f_norm;
            A = [(w_r~=0)*P;P];
            derr =(d.*A(3:2:end,:) - (1-d).*A(2:2:end,:)); %derr/ds of this layer            
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
        function prev_derr_merged = DimensionMergeOneTree(prev_derr,selected_dims,dim)    
            selected_dims = selected_dims(:);
            prev_derr = prev_derr(1:end-1,:);
            prev_derr_merged = zeros(dim,size(prev_derr,2));
            prev_derr_merged(selected_dims,:) = prev_derr;     
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
        
        %% For Utility
        function content = SparseSelect2D(data,x,y)
            l = size(data,1);
            content = data((y-1)*l+x);
        end
 
    end
end