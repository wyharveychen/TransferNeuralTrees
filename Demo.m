addpath('./TNT_layers'); %function path
file_path = './datasets';

S_dataset = 'AMAZON_DECAF';
T_dataset = 'WEBCAM_SURF';

S_filename = sprintf('%s/%s.mat',file_path,S_dataset);
S = load(S_filename,'data','label');
S.dataset = S_dataset;


T_filename = sprintf('%s/%s.mat',file_path,T_dataset);
T = load(T_filename,'data','label'); 
L.dataset = T_dataset;
U.dataset = T_dataset;

%fix id 
%For convienice, our experiment (including the ones for compared method) is based on
%one fixed partition (still randomly selected).
L_id = 1:30;
U_id = 31:length(T.label);
         
%random id 
%one can use these code instead to compare randomized id
% T_class = unique(T.label);
% L_id = [];
% U_id = [];   
% for y = T_class        
%     yid = find( T.label == y);
%     yid = yid(randperm(length(yid)));
%     L_id = [L_id, yid(1:3)];
%     U_id = [U_id, yid(4:end)];                     
% end

L.data = T.data(:,L_id);
L.label = T.label(L_id);
U.data = T.data(:,U_id);
U.label = T.label(U_id);

acc = TNTforHDA(S,L,U);    
