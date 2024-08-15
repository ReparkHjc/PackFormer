clear;close all;clc;
%% 
data=load('Oxford_Battery_Degradation_Dataset_1.mat');
names = fieldnames(data);

%% 获取电池最少循环次数
len_num=[];
for name=1:length(names)
    data_name = strcat('data','.',names{name});
    num=length(fieldnames(eval(data_name)));
    len_num=[len_num,num];
end

%% 处理那几块电池并获取目标电池充放电最少次数
num=4:6;
% num=[1:3,7:8];
target_batter={};
for i=1:length(num)
    target_batter{i,1}={names{num(i)}};
    i=i+1;
end
target_num=[];
for i=1:length(num)
    data_name = strcat('data','.',names{num(i)});
    num_cycle=length(fieldnames(eval(data_name)));
    target_num=[target_num,num_cycle];
end
min_cycle_num=min(target_num);
%% 获取特征
v_min=[];
v_max=[];
target=[];
target1=[];
target2=[];
Q=[];
Q1=[];

for batter_index=1:length(num)
    data_name = strcat('data','.',names{num(batter_index)});
    data_cycle=eval(data_name);
    data_name=fieldnames(data_cycle);
    target2=[];
    Q1=[];
    for batter_cycle=1:min_cycle_num-1
        target_feature_v = eval(strcat('data_cycle','.',data_name{batter_cycle},'.','C1ch','.','v'));
        target_feature_t = eval(strcat('data_cycle','.',data_name{batter_cycle},'.','C1ch','.','t'));
        target_feature_q = eval(strcat('data_cycle','.',data_name{batter_cycle},'.','C1ch','.','q'));
        target_feature_T = eval(strcat('data_cycle','.',data_name{batter_cycle},'.','C1ch','.','T'));
        
        v_min=[v_min,min(target_feature_v)];
        v_max=[v_max,max(target_feature_v)];
        
        [seq,seq_len,position]=getpostion(target_feature_v,[2.75,4.195]);
        target_feature_t=target_feature_t(position(1):position(2));
        target_feature_q=target_feature_q(position(1):position(2));
        target_feature_T=target_feature_T(position(1):position(2));
        %插值出196个点  data_cycle.cyc0000.C1dc.q
        target_feature_t=filter_row(target_feature_t,position);
        target_feature_q=filter_row(target_feature_q,position);
        target_feature_T=filter_row(target_feature_T,position);
        
        target1=[target_feature_t;target_feature_q;target_feature_T];
        target2=[target2,target1];
        
        %容量
        a=strcat('data_cycle','.',data_name{batter_cycle},'.','C1dc','.','q')
        target_Q = eval(strcat('data_cycle','.',data_name{batter_cycle},'.','C1dc','.','q'));
        target_Q=max(abs(target_Q));
        Q1=[Q1;target_Q];
    end
    Q=[Q,Q1];
    target=[target;target2];
end
target=target';
Q_max=max(Q)
SOH=Q./Q_max
SOH=(SOH(:,1)+SOH(:,2)+SOH(:,3))/3;
% SOH=(SOH(:,1)+SOH(:,2)+SOH(:,3)+SOH(:,4)+SOH(:,5))/5;
%% 保存数据

csvwrite("SOH.csv",SOH)
csvwrite('data.csv',target)

%% 获得目标电压序列
function [seq,seq_len,position]=getpostion(seq,target_t)
seq_len=length(seq);
c=1;d=0;
for j=1:seq_len-1
    if (seq(j)-target_t(1)) <0
        if (seq(j+1)-target_t(1)) >0
            c=j;
            break;
        end
    end
end
for j=1:seq_len-1
    if (seq(j)-target_t(2)) <0
        if (seq(j+1)-target_t(2)) >0
            d=j+1;
            break;
        end
    end
end
seq=seq(c:d);
position=[c,d];
seq_len=length(seq);
end
%% 插值196个点
function filter_seq=filter_row(seq,position)
    i=1:(position(2)-position(1))/179:position(2);
    j=1:length(seq);
    filter_seq=interp1(j,seq,i,'spline');
    seq_max=max(filter_seq);seq_min=min(filter_seq);
    filter_seq=(filter_seq-seq_min)/(seq_max-seq_min);
end