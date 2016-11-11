addpath('./TNT_layers'); %function path

dataset_s = {'WEBCAM','CALTECH','AMAZON'};
feature_s = {'DECAF','SURF'};
file_path = './datasets';

S_dataset_s = {'AMAZON','CALTECH','WEBCAM','AMAZON','AMAZON', 'CALTECH','CALTECH','WEBCAM','WEBCAM' };
T_dataset_s = {'AMAZON','CALTECH','WEBCAM','WEBCAM','CALTECH','AMAZON', 'WEBCAM', 'AMAZON','CALTECH'};

acc = [];
for iter = 1
    for d_i = 3

        S_dataset = S_dataset_s{d_i};    
        T_dataset = T_dataset_s{d_i};    
        S_feature = feature_s{1};
        T_feature = feature_s{2};

        S_filename = sprintf('%s/%s_%s.mat',file_path,S_dataset,S_feature);
        S = load(S_filename,'data','label');
        S.dataset = S_dataset;

        T_filename = sprintf('%s/%s_%s.mat',file_path,T_dataset,T_feature);
        T = load(T_filename,'data','label');    

        T_class = unique(T.label);

        %fix id 
        %For convience, our experiment (including other method) is based on
        %one specific partition.
        L_id = 1:30;
        U_id = 31:length(T.label);

         
        %random id 
        %one can use these code instead to compare randomized id
%         L_id = [];
%         U_id = [];   
%         for y = T_class        
%             yid = find( T.label == y);
%             yid = yid(randperm(length(yid)));
%             L_id = [L_id, yid(1:3)];
%             U_id = [U_id, yid(4:end)];                     
%         end
        
        L.data = T.data(:,L_id);
        L.label = T.label(L_id);
        L.dataset = T_dataset;
        U.data = T.data(:,U_id);
        U.label = T.label(U_id);
        U.dataset = T_dataset;

        acc(iter,d_i) = TNTforHDA(S,L,U);    
    end
end