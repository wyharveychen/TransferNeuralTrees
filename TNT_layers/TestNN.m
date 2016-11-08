load /home/chuna/Zeroshot/TNT/testdata/USPS.mat data label;

train_size = 1000;
train_data = data(:,1:train_size);
train_label = label(:,1:train_size);
test_data = data(:,(train_size+1):end);
test_label = label(:,(train_size+1):end);

in_dim = size(train_data,1);
in_num = size(train_data,2);
out_dim = length(unique(train_label));

label_list = unique(train_label);

simple_nn = NN({BasicLayers(in_dim,100),L2NormLayers(),BasicLayers(100,out_dim),SoftmaxLayers(label_list)});
simple_nn.Train(train_data,train_label,struct('epoch_num',50,'batch_num',4));
predicted_label = simple_nn.Predict(test_data);
fprintf('acc: %d\n',mean(predicted_label == test_label));

%{
ndf_nn = NN({BasicLayers(in_dim,100), L2NormLayers(),BasicLayers(100,100),NDFLayers(100,label_list)});
ndf_nn.Train(train_data,train_label,struct('epoch_num',50,'batch_num',5,'converge_acc',1));
predicted_label = ndf_nn.Predict(test_data);
fprintf('acc: %d\n',mean(predicted_label == test_label));
%}
%{
layer_num = length(layers);
inputs = {};
derrs = {};
grads = {};

for k = 1:50   
    inputs{1} =  train_data;
    for l_id = 1:length(layers)
        inputs{l_id+1} = layers{l_id}.Forward(inputs{l_id});
    end
    derrs{layer_num+1} = train_label;
    for l_id = layer_num:-1:1
        derrs{l_id} = layers{l_id}.Backward(derrs{l_id+1});
        layers{l_id}.Update();
    end
       
    disp(layers{layer_num}.TotalError());
end
%}