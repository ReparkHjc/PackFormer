clear;close all;clc;
%% 
load('Oxford_Battery_Degradation_Dataset_1.mat');
Testdata = Cell1;
names = fieldnames(Testdata);
value1 = 3.55;
value2 = 3.75;
feature1 = [];
lable1 = [];
%%
for name = 1:length(names)
    index = names(name);
    data_name_C1ch_t = strcat('Cell1.',index,'.C1ch.t');
    data_name_C1ch_v = strcat('Cell1.',index,'.C1ch.v');
    data_name_C1ch_q = strcat('Cell1.',index,'.C1ch.q');
    
    Data_C1ch_t = eval(data_name_C1ch_t{1});
    Data_C1ch_v = eval(data_name_C1ch_v{1});
    Data_C1ch_q = eval(data_name_C1ch_q{1});
    
    data = [Data_C1ch_v Data_C1ch_t];
    
    index1 = find(data(:,1)>=value1 & data(:,1)<=value2);
    data1 = data(index1,:);
    Seq_min = min(data1(:,2));
    Seq_max = max(data1(:,2));
    
    data2 = Seq_max-Seq_min;
    feature1 = [feature1;data2];
    
    Seq_max_C1ch_q=max(Data_C1ch_q(:,1));
    lable1 =[lable1;Seq_max_C1ch_q];

end
%% 
 clear data data1 data2 Data_C1ch_t Data_C1ch_v Data_C1ch_q Testdata value1;
 clear data_name_C1ch_t data_name_C1ch_v data_name_C1ch_q  value2;
 clear Seq_max_C1ch_q Seq_min Seq_max index names name index1;

