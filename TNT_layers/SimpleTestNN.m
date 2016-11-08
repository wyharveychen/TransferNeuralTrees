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

simple_nn = NN({BasicLayers(in_dim,out_dim),SoftmaxLayers(label_list)});
simple_nn.Train(train_data,train_label,struct('epoch_num',50,'batch_num',4));
predicted_label = simple_nn.Predict(test_data);
fprintf('acc: %d\n',mean(predicted_label == test_label));