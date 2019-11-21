clear
clc;
tic
t1=toc;
DataSample = csvread('data/communitycrime/balancecrimecommunity.csv'); 
selected_sample=500;
training = DataSample(1:selected_sample,1:100); 
[r,c]=size(training);
average=zeros(c,1);
sigma=zeros(c,1);
temp=0;
for i=1:c
    average(i,1)=mean(training(:,i));
    for j=1:r
        temp = temp+(training(j,i)-average(i,1))^2;
        sigma(i,1)=sqrt(temp/r);
    end
end
for i=1:r
    parfor j=1:c
        training(i,j)=(training(i,j)-average(j,1))/sigma(j,1); 
    end
end
normal_data = training;
DataSample1 = csvread('data/communitycrime/balancecrimecommunity.csv');
normal_data(:,1)=DataSample1(1:selected_sample,1);
train_x = normal_data(1:selected_sample,2:100);
train_label = DataSample1(1:selected_sample,102);
train_Sensitive=DataSample1(1:selected_sample,1);
covariance=0.1;
n=r;
featureNum=99;
total =600; 
u=0;
sigma=1;
R = normrnd(u,sigma,featureNum,total); 
[new_fair,new_unfair] = GaussianRandomNoisefunction(total,train_x,train_Sensitive,R,featureNum,covariance);
W_fair = new_fair;
W_unfair=new_unfair;
[fa fb]=size(W_fair);
[fc, fd]=size(W_unfair);
fair_prediction = zeros(r,fb);
unfair_prediction = zeros(r,fd);
prediction_fair_sigal = zeros(r,1);
prediction_unfair_sigal = zeros(r,1);
for i=1:fb
    parfor j=1:r
        prediction_fair_sigal(j,1)=train_x(j,:)*W_fair(:,i);
    end
    fair_prediction(:,i) = prediction_fair_sigal;
end
for i=1:fd
    parfor j=1:r
        prediction_unfair_sigal(j,1)=train_x(j,:)*W_unfair(:,i);
    end
    unfair_prediction(:,i) = prediction_unfair_sigal;
end
sample  = [fair_prediction unfair_prediction];
sample_size = fb+fd;
model_num=20; 
infer_sum=zeros(r,model_num);
for i=1:model_num
    num_fair = randi([1,40],1,1); 
    num_unfair = 40-num_fair;
    set_fair = randperm(fb,fb);
    fair_index = set_fair(1,1:num_fair);
    set_unfair = randperm(fd,fd);
    unfair_index = set_unfair(1,1:num_unfair);
    fair_yhat = zeros(selected_sample,num_fair);
    unfair_yhat = zeros(selected_sample,num_unfair);
    parfor j=1:num_fair
        fair_yhat(:,j) = fair_prediction(:,fair_index(1,j));
    end
    parfor k=1:num_unfair
        unfair_yhat(:,k) = unfair_prediction(:,unfair_index(1,k));
    end
    A = zeros(r,num_fair);
    B = zeros(r,num_unfair);
    for p=1:num_fair
        parfor k=1:r
            A(k,p) = fair_yhat(k,p)-mean(fair_yhat(:,p));
        end
    end
    A = transpose(A);
    for j=1:num_unfair
        parfor p=1:r
            B(p,j)=unfair_yhat(p,j)-mean(unfair_yhat(:,j));
        end
    end
    B = transpose(B);
    infer_s = optimvar('infer_s',n,'Type','integer','LowerBound',0,'UpperBound',1);
    infer_prob = optimproblem('Objective',1);
    for g=1:num_fair
        fair1 = A(g,:)*infer_s;
        con1(g)= fair1 <=(n-1)*covariance;
        con2(g) = fair1 >= 0;
    end
    infer_prob.Constraints.cov1=con1;
    infer_prob.Constraints.cov2=con2;
    for j=1:num_unfair
        unfair1 = B(j,:)*infer_s;
        con3(j)=unfair1 >=(n-1)*covariance;
    end
    infer_prob.Constraints.cov3=con3;
    problem = prob2struct(infer_prob);
    [sol,fval,exitflag,output] = intlinprog(problem);
    if(~isempty(sol))
        infer_sum(:,i) = sol;
    end
    clearvars fair unfair con1 con2 con3
end
count=0;
for i=1:r
    count = sum(infer_sum(i,:));
    avg = count/model_num;
    if(avg>=0.5)
        result(i,1)=1;
    else
        result(i,1)=0;
    end
end
error_count=0;
for r=1:n
    if(result(r,1)~= train_Sensitive(r,1))
        error_count = error_count+1;
    end
end
error_rate = error_count/n;
writetable(table(result),'infer_s.txt','Delimiter','\t');
writetable(table(error_rate),'error_rate.txt','Delimiter','\t');
t2=toc;
display(strcat('Excution time is:',num2str(t2),'s'));