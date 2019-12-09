clear
clc;
tic
t1=toc;
parpool('local',6);
DataSample = csvread('data/communitycrime/crimecommunity.csv'); 
%--------------------- normalize data---------------------------
selected_sample=500;
training = DataSample(1:selected_sample,1:100); %training samples
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
    for j=1:c
        training(i,j)=(training(i,j)-average(j,1))/sigma(j,1); %1493*99
    end
end
normal_data = training;
DataSample1 = csvread('data/communitycrime/balancecrimecommunity.csv');
normal_data(:,1)=DataSample1(1:selected_sample,1);
train_x = normal_data(1:selected_sample,2:100);
train_label = DataSample1(1:selected_sample,102);
train_Sensitive=DataSample1(1:selected_sample,1);
%-------------------------------fair vectors-------------------------------
covariance=0.1;
iteration=20; 
n=r;
error_rate = zeros(iteration,1);
infer_sum = zeros(n,iteration);
times=0;
featureNum=99;
for loop=1:iteration
    total=600; 
    u=0;
    sigma=1;
    R = normrnd(u,sigma,featureNum,total);
    W_fair=zeros(featureNum,total);
    W_unfair=zeros(featureNum,total);
    for i=1:total
     [W_fair(:,i),W_unfair(:,i)]=NewThirdParty(train_Sensitive,train_x*R(:,i),R(:,i),covariance);
    end
    W_fair(:,all(W_fair==0)) = [];
    W_unfair(:,all(W_unfair==0)) = [];
    [fa fb]=size(W_fair);
    [fc,fd]=size(W_unfair);
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
    A = zeros(r,fb);
    B = zeros(r,fd);
    for i=1:fb
        parfor k=1:r
            A(k,i) = fair_prediction(k,i)-mean(fair_prediction(:,i));
        end
    end
    A = transpose(A);
    for j=1:fd
        parfor p=1:r
            B(p,j)=unfair_prediction(p,j)-mean(unfair_prediction(:,j));
        end
    end
    B = transpose(B);
    infer_s = optimvar('infer_s',n,'Type','integer','LowerBound',0,'UpperBound',1);
    infer_prob = optimproblem('Objective',1);
    if(fb~=0)
        for i=1:fb
            fair1 = A(i,:)*infer_s;
            con1(i)= fair1 <=(n-1)*covariance;
            con2(i) = fair1 >= 0;
        end
        infer_prob.Constraints.cov1=con1;
        infer_prob.Constraints.cov2=con2;
    end
    if(fd~=0)
        for j=1:fd
            unfair1 = B(j,:)*infer_s;
            con3(j)=unfair1 >=(n-1)*covariance;
        end
        infer_prob.Constraints.cov3=con3;
    end
    problem = prob2struct(infer_prob);
    [sol,fval,exitflag,output] = intlinprog(problem);
    if(~isempty(sol))
        infer_sum(:,loop) = sol;
    end
    error_count=0;
    for r=1:n
        if(infer_sum(r,loop)~= train_Sensitive(r,1))
            error_count = error_count+1;
            
        end
    end
    error_rate(loop,1) = error_count/n;
    clearvars fair unfair con1 con2 con3
end
writetable(table(infer_sum),'infer_s.txt','Delimiter','\t');
writetable(table(error_rate),'error_rate.txt','Delimiter','\t');
t2=toc;
display(strcat('Excution time is:',num2str(t2),'s'));
