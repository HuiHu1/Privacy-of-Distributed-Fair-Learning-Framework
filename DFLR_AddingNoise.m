clear
clc;
tic
t1=toc;
DataSample = csvread('data/communitycrime/crimecommunity.csv');
randomset = csvread('data/communitycrime/crimecommunity_index.csv');
training = DataSample(1:500,2:100);
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
testing = DataSample(1494:1993,2:100); 
[row,column]=size(testing);
temp2=0;
for i=1:row
    for j=1:column
        testing(i,j)=(testing(i,j)-average(j,1))/sigma(j,1); %500*99
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
    %normal_data = [training;testing];
    trainset = randomset(1:500,loop);
    testset = randomset(1494:1993,loop);
    train_x = normal_data(trainset(:,1),:);
    train_label = DataSample(trainset(:,1),102);
    train_Sensitive=DataSample(trainset(:,1),1);
    test_x = normal_data(testset(:,1),:);
    test_label = DataSample(testset(:,1),102);
    total=600;
    u=0;
    sigma=1;
    covariance=0.1;
    R =normrnd(u,sigma,featureNum,total);
    %--------------------------------first randomnoise----------------
    %     [new_fair,new_unfair] = GaussianRandomNoisefunction(total,train_x,train_Sensitive,R,featureNum,covariance);
    %      W_fair = new_fair;
    %-------------------------second mechanism--------------
    %     [new_fair,new_unfair] = GaussiannoiseintohatY(total,train_x,train_Sensitive,R,featureNum,covariance);
    %      W_fair = new_fair;
    %-------------------------------no noise------------------------
    W_fair=zeros(featureNum,total);
    for i=1:total
        W_fair(:,i)=NewThirdParty(train_x*R(:,i),R(:,i),covariance,loop,DataSample,randomset);
    end
    W_fair(:,all(W_fair==0)) = [];
    %----------------------------------------------------------------------
    [a b]=size(W_fair);
    if(b~=0)
        dimension=b;
        W_new =W_fair;
        for parameter=-3:3
            alpha =10^(parameter);
            cofficient = pinv(transpose(W_new)*transpose(train_x)*train_x*W_new+alpha*eye(dimension,dimension))*transpose(W_new)*transpose(train_x)*train_label;
            prediction_label = test_x*(W_new*cofficient);
            [prow pcolumn]=size(prediction_label);
            for i = 1:prow
                if(abs(prediction_label(i,1))>=0.3)
                    prediction_label(i,1)=1;
                else
                    prediction_label(i,1)=0;
                end
            end
            [trow,tcolumn]=size(test_x);
            count =0;
            for j=1:trow
                if(abs(prediction_label(j,1))~=test_label(j,1))
                    count =count+1;
                end
            end
            error_temp = count/trow;
            if(data1(loop,1) > error_temp)
                data1(loop,1) = error_temp;
            end
            error_temp=0;
            [sub_error_min,sub_error_ma,Fair_SP,Fair_DI] = Calcualte(prediction_label,test_label,loop,DataSample,randomset);
            data_minority(loop,1)=sub_error_min;
            data_majority(loop,1)=sub_error_ma;
            if(SP(loop,1)>Fair_SP)
                SP(loop,1)=Fair_SP;
            end
            if(SP(loop,1)==0)
                DI(loop,1)=0;
            else
                if(DI(loop,1)>Fair_DI)
                    DI(loop,1)=Fair_DI;
                end
            end
        end
    end
end
writetable(table(data1),'prediction_error.txt','Delimiter','\t');
writetable(table(SP),'Group fairness.txt','Delimiter','\t');
t2=toc;
display(strcat('parfor并行计算时间：',num2str(t2),'秒'));