function test_acc = TNTforHDA(source, labeled_target, unlabeled_target)
    %% Initialization
    source_dim            = size(source.data ,1);    
    target_dim            = size(labeled_target.data,1);
    label_list            = unique(source.label);
    common_dim            = 100;

    source_mapping        = BasicLayers(source_dim,common_dim);
    
    target_mapping        = BasicLayers(target_dim,common_dim);  
    classifier            = NDFLayers(common_dim,label_list);

    source_path = NN({source_mapping,classifier});
    %source_path = NN({source_mapping,L2NormLayers(), BasicLayers(common_dim,common_dim),classifier});

    
    target_path = NN({target_mapping,classifier});
    %target_path = NN({target_mapping,L2NormLayers(), BasicLayers(common_dim,common_dim),classifier});

    
    %% Preprocessing
    source.data = normc(source.data);
    labeled_target.data = normc(labeled_target.data);
    unlabeled_target.data = normc(unlabeled_target.data);


    %% Training with source data
    source_path.Train(source.data,source.label,struct('epoch_num',20,'batch_num',5,'converge_acc',0.99));
    acc(1) = mean(source_path.Predict(source.data) == source.label);        
    fprintf('Source train acc = %d\n',acc(1));
    clf; hold on;
    classifier.ShowProjectedData(source.label,'+');
    %% Training with target data
    for i = 1:100
        target_path.Train(labeled_target.data,labeled_target.label,     struct('epoch_num',1,'updatestop_layer',1,'init_epoch',i));
        target_path.Train([labeled_target.data,unlabeled_target.data],[],  struct('epoch_num',1,'updatestop_layer',1, 'lr_rate', i/1000));
        acc(3) = mean(target_path.Predict(unlabeled_target.data) == unlabeled_target.label);        
        fprintf('Target test acc = %d\n',acc(3));
    end
    acc(2) = mean(target_path.Predict(labeled_target.data) == labeled_target.label);        
    fprintf('Target train acc = %d\n',acc(2));
    classifier.ShowProjectedData(labeled_target.label,'o');

    %% Test unlabeled target data
    acc(3) = mean(target_path.Predict(unlabeled_target.data) == unlabeled_target.label);        
    fprintf('Target test acc = %d\n',acc(3));
    test_acc = acc(3);    
    classifier.ShowProjectedData(unlabeled_target.label,'.');
end