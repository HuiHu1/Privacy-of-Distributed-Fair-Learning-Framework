clear
clc;
tic
t1=toc;
DataSample = csvread('data/communitycrime/crimecommunity.csv');
randomset = csvread('data/communitycrime/crimecommunity_index.csv');
training = DataSample(1:500,2:100); %training samples
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
        training(i,j)=(training(i,j)-average(j,1))/sigma(j,1);
    end
end
testing = DataSample(1494:1993,2:100);
[row,column]=size(testing);
temp2=0;
for i=1:row
    for j=1:column
        testing(i,j)=(testing(i,j)-average(j,1))/sigma(j,1);
    end
end
%---------------------------------------------------------------------------
iter=20;
featureNum=99;
for i=1:iter
    data1(i,1)=30;
    SP(i,1)=30;
    DI(i,1)=30;
    data_minority(i,1)=30;
    data_majority(i,1)=30;
end
for loop=1:iter
    normal_data = DataSample(:,2:100);
    trainset = randomset(1:500,loop);%1493*1
    testset = randomset(1494:1993,loop);%500*1
    train_x = normal_data(trainset(:,1),:);
    train_label = DataSample(trainset(:,1),102);
    train_Sensitive = DataSample(trainset(:,1),1);
    test_x = normal_data(testset(:,1),:);
    test_label = DataSample(testset(:,1),102);
    total=600;
    u=0;
    sigma=1;
    covariance=0.1;
    R =normrnd(u,sigma,featureNum,total);
    %     [new_fair,new_unfair] = GaussianRandomNoisefunction(total,train_x,train_Sensitive,R,featureNum,covariance);
    %     W_fair = new_fair;
    %     [new_fair,new_unfair] = GaussiannoiseintohatY(total,train_x,train_Sensitive,R,featureNum,covariance);
    %      W_fair = new_fair;
    W_fair=zeros(featureNum,total);
    for i=1:total
        W_fair(:,i)=NewThirdParty(train_x*R(:,i),R(:,i),covariance,loop,DataSample,randomset);
    end
    W_fair(:,all(W_fair==0)) = [];
    [row column]=size(W_fair);
    sigma2=0.1;
    [x1,y1]=size(train_x);
    pr_back_0=zeros(x1,1);
    pr_back_1=zeros(x1,1);
    al = zeros(iter,column);
    for parameter=-3:3
        lamda=10^(parameter);
        alpha_initial=10*normrnd(u,sigma2,column,1);
        for j=1:x1
            pr_train(j,1) = exp(train_x(j,:)*(W_fair*alpha_initial))/(1+exp(train_x(j,:)*(W_fair*alpha_initial)));
            pr_back_0(j,1)=pr_train(j,1);
            pr_back_1(j,1)=1-pr_back_0(j,1);
            if(pr_train(j,1)<0.5)
                pr_train(j,1)=1;
            else
                pr_train(j,1)=0;
            end
        end
        counter=0;
        [x2,y2]=size(test_x);
        for i=1:x2
            pr_test(i) = exp(test_x(i,:)*(W_fair*alpha_initial))/(1+exp(test_x(i,:)*(W_fair*alpha_initial)));
            if(pr_test(i)<=0.5)
                pr_test(i)=1;
            else
                pr_test(i)=0;
            end
            if(pr_test(i)~=test_label(i,1))
                counter = counter+1;
            end
        end
        error_temp=counter/x2;
        if(data1(loop,1)>error_temp)
            data1(loop,1)=error_temp;
        end
        error_temp=0;
        [error_min,error_ma,Fair_SP,Fair_DI]=Calcualte(pr_test,test_label,loop,DataSample,randomset);
        if(data_minority(loop,1)>error_min)
            data_minority(loop,1)=error_min;
        end
        if(data_majority(loop,1)>error_ma)
            data_majority(loop,1)=error_ma;
        end
        SP(loop,1)=Fair_SP;
        if(SP(loop,1)==0)
            DI(loop,1)=0;
        else
            DI(loop,1)=Fair_DI;
        end
        %--------------------------update alpha-----------------------------
        M = zeros(x1,x1);
        for i=1:x1
            for j=1:x1
                if(i==j)
                    M(i,j)=pr_back_0(i,1)*pr_back_1(j,1);
                end
            end
        end
        first_partial = transpose(W_fair)*transpose(train_x)*(train_label-pr_back_1)+2*lamda*transpose(W_fair)*W_fair*alpha_initial;
        second_partial = transpose(W_fair)*transpose(train_x)*M*train_x*W_fair+2*lamda*transpose(W_fair)*W_fair;
        learning_rate=0.001;
        alpha_temp = alpha_initial-learning_rate*inv(second_partial)*first_partial;
        alpha_initial = alpha_temp;
        alpha_temp=0;
        al(loop,:) =  alpha_initial;
    end
end
writetable(table(data1),'prediction_error.txt','Delimiter','\t');
writetable(table(SP),'Group fairness.txt','Delimiter','\t');
t2=toc;
display(strcat('parfor并行计算时间：',num2str(t2),'秒'));