function test_acc = TNTforHDA(source, labeled_target, unlabeled_target)
    %% Initialization
    source_dim            = size(source.data ,1);    
    target_dim            = size(labeled_target.data,1);
    label_list            = unique(source.label);
    common_dim            = 100;

    source_mapping        = {BasicLayers(source_dim,common_dim)};
    target_mapping        = {BasicLayers(target_dim,common_dim)};
        
    %In case you are intereseted in deeper structure, you can try following code,
    %while it has sometimes better but unstable performance.
    
    %target_mapping        = {BasicLayers(target_dim,16*common_dim),L2NormLayers(), ...
    %                         BasicLayers(16*common_dim,4*common_dim),L2NormLayers(), ... 
    %                         BasicLayers(4*common_dim,common_dim)};  
    
    classifier            = {NDFLayers(common_dim,label_list)};

    source_path = NN([source_mapping,classifier]);    
    target_path = NN([target_mapping,classifier]);
 
    Optimizer.Method('RMSprop');
    %% Preprocessing
    source.data = normc(source.data);
    labeled_target.data = normc(labeled_target.data);
    unlabeled_target.data = normc(unlabeled_target.data);


    %% Training with source data
    source_path.Train(source.data,source.label,struct('epoch_num',50,'batch_num',5,'converge_acc',0.99));
    acc(1) = mean(source_path.Predict(source.data) == source.label);        
    fprintf('Source train acc = %d\n',acc(1));
    clf; hold on;
    source_path.ShowProjectedData(length(source_mapping), source.label,'+');
    source.projected_data = source_path.GetProjectedData(length(source_mapping));


    %% Training with target data    
    for i = 1:100
        target_path.Train(labeled_target.data,labeled_target.label,     struct('epoch_num',1,'updatestop_layer',length(target_mapping),'init_epoch',i));
        lambda  = 2/(1+exp(-0.01 *i))-1; %just gradually ascend from 0 to ~ 1, 
        target_path.Train([labeled_target.data,unlabeled_target.data],[],  struct('epoch_num',1,'updatestop_layer',length(target_mapping), 'lr_rate', lambda));
        
        acc(3) = mean(target_path.Predict(unlabeled_target.data) == unlabeled_target.label);        
        fprintf('Target test acc = %d\n',acc(3));
    end
    acc(2) = mean(target_path.Predict(labeled_target.data) == labeled_target.label);        
    fprintf('Target train acc = %d\n',acc(2));
    target_path.ShowProjectedData(length(target_mapping), labeled_target.label,'o');
    labeled_target.projected_data = target_path.GetProjectedData(length(target_mapping));

    %% Test unlabeled target data
    acc(3) = mean(target_path.Predict(unlabeled_target.data) == unlabeled_target.label);        
    fprintf('Target test acc = %d\n',acc(3));
    test_acc = acc(3);    
    
    target_path.ShowProjectedData(length(target_mapping), unlabeled_target.label,'.');
    unlabeled_target.projected_data = target_path.GetProjectedData(length(target_mapping));
     
    %% Record file
    %remove field no need to save
    source = rmfield(source,'data'); labeled_target = rmfield(labeled_target,'data'); unlabeled_target = rmfield(unlabeled_target,'data');    
    data_name = sprintf('./record_data/TNTforHDA_%sto%s_%s',source.dataset,labeled_target.dataset,Util.TimeStamp());
    save(data_name, 'source', 'labeled_target', 'unlabeled_target');
end