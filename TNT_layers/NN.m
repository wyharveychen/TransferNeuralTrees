classdef NN < handle
   properties       
       layers;
       layer_num;
       inputs;
       derrs;
       final_cost;
       is_iteratively_update = 0;              
   end
   methods
        function nn = NN(layers)
            nn.layers = layers;
            nn.layer_num =  length(nn.layers);

        end
        function Train(nn,train_data,train_label,opt)  
            batch_num        = Util.Initialize(opt,'batch_num',1);
			epoch_num        = Util.Initialize(opt,'epoch_num', 50);
            input_layer      = Util.Initialize(opt,'input_layer',1);
            output_layer     = Util.Initialize(opt,'output_layer',nn.layer_num);
            updatestop_layer = Util.Initialize(opt,'updatestop_layer',output_layer);
            converge_acc     = Util.Initialize(opt,'converge_acc',1.01);% default: no converge
            
            has_label        = ~isempty(train_label);
            lr_weight        = Util.Initialize(opt,'lr_weight',1);
            init_epoch       = Util.Initialize(opt,'init_epoch',1);

                      
            batch_size = floor(size(train_data, 2)/ batch_num);

            ndf_pi_allbatch = cell(1,batch_num);            
            if(nn.is_iteratively_update && strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                ndf_pi_allbatch_treewise = cell(nn.layers{nn.layer_num}.tree_num,batch_num);            
            end
            for epoch = init_epoch:( init_epoch + epoch_num -1)
                batch_perm_id = reshape(randperm(size(train_data, 2), batch_num * batch_size),batch_num,batch_size);                
                total_cost = 0;
                if(~has_label && strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                    emloss = 0; %debug only, not quite good to directly count NDF specific attribute in a general NN.m
                end
                train_acc = 0;
                for batch = 1:batch_num                    
                    
                    nn.inputs{1} =  train_data(:,batch_perm_id(batch,:));
                    for l_id = input_layer:output_layer 
                        nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
                    end
                    
                    if(has_label)
                        nn.derrs{nn.layer_num+1} = train_label(batch_perm_id(batch,:));
                    else
                        nn.derrs{nn.layer_num+1} = [];
                    end
                    
                    for l_id = output_layer:-1:input_layer
                        nn.derrs{l_id} = nn.layers{l_id}.Backward(nn.derrs{l_id+1});
                            
                    end
                    
                    for l_id = input_layer:updatestop_layer
                        nn.layers{l_id}.Update(lr_weight);
                    end
                    
                    %total_cost = total_cost+ nn.layers{nn.layer_num}.TotalError();
                    %% Fully update PI                   
                    if( strcmp(nn.layers{nn.layer_num}.type, 'NDF') && updatestop_layer>=nn.layer_num)
                        ndf_pi_allbatch{batch} = nn.layers{nn.layer_num}.pi;     
                        average_pi = ndf_pi_allbatch{1};                        
                        if(batch_num ~= 1)
                            for i = 2:batch_num
                                if(isempty(ndf_pi_allbatch{i}))
                                    i = i-1;
                                    break;
                                end
                                average_pi = cellfun(@plus,average_pi,  ndf_pi_allbatch{i},'UniformOutput',0);
                            end                           
                            average_pi = cellfun(@(pi) pi/i,average_pi,'UniformOutput',0);
                        end
                        nn.layers{nn.layer_num}.UpdatePIDirectly(average_pi);
                    end              
                    
                    if(has_label)
                        total_cost = total_cost+ nn.layers{nn.layer_num}.TotalError();
                        train_acc = train_acc+ mean(nn.layers{nn.layer_num}.Predict() == nn.derrs{nn.layer_num+1});                  
                    elseif(strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                        emloss = emloss + sum(nn.layers{nn.layer_num}.EmbeddingLoss());
                    end
                end
                %predicted_label = Predict(nn,train_data);
                if(has_label)
                    nn.final_cost = total_cost;               
                    train_acc = train_acc/batch_num;
                    fprintf('Epoch %d, Training Cost: %d, Train Acc: %d\n',epoch, total_cost, train_acc);               
                    if(converge_acc<=train_acc)
                        break;
                    end
                elseif(strcmp(nn.layers{nn.layer_num}.type, 'NDF'))
                    fprintf('Emloss: %d\n',emloss);               
                end
            end
        end
        function predicted_label = Predict(nn,test_data)
            nn.inputs{1} =  test_data;
            for l_id = 1:nn.layer_num
                nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
            end
            predicted_label = nn.layers{nn.layer_num}.Predict();
        end
        function ClearCache(nn)
            for l_id = 1:nn.layer_num
                nn.layers{l_id}.ClearCache();
            end
        end
        function UpdateUnseenClass(nn,in,y)
            nn.inputs{1} =  in;
            nn.layers{nn.layer_num}.InitializePILayer(unique(y));
            for l_id = 1:nn.layer_num
                nn.inputs{l_id+1} = nn.layers{l_id}.Forward(nn.inputs{l_id});
            end
            %nn.layers{nn.layer_num}.Backward(y);
                %nn.layers{nn.layer_num}.UpdatePIOnly(y);            
            nn.layers{nn.layer_num}.UpdatePIOnly(y,1/100/length(unique(y)));                        
        end
   end
   
end