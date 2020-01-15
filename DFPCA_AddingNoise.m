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
        training(i,j)=(training(i,j)-average(j,1))/sigma(j,1); 
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
    trainset = randomset(1:500,loop);
    testset = randomset(1494:1993,loop);
    train_x = normal_data(trainset(:,1),:);
    train_label = DataSample(trainset(:,1),102);
    train_Sensitive = DataSample(trainset(:,1),1);
    test_x = normal_data(testset(:,1),:);
    test_label = DataSample(testset(:,1),102);
    total=600;
    u=0;
    sigma=10;
    covariance=0.1;
    R =normrnd(u,sigma,featureNum,total);
    %     [new_fair,new_unfair] = GaussianRandomNoisefunction(total,train_x,train_Sensitive,R,featureNum,covariance);
    %     W_fair = new_fair;
    %     [new_fair,new_unfair] = GaussiannoiseintohatY(total,train_x,train_Sensitive,R,featureNum,covariance);
    %     W_fair = new_fair;
    W_fair=zeros(featureNum,total);
    for i=1:total
        W_fair(:,i)=NewThirdParty(train_x*R(:,i),R(:,i),covariance,loop,DataSample,randomset);
    end
    W_fair(:,all(W_fair==0)) = [];
    [a b]=size(W_fair);
    W_new =W_fair;
    cov_x = cov(train_x);
    W_projection = zeros(featureNum,b);
    beta=zeros(b,b);
    Q = transpose(W_new)*W_new;
    matrix = pinv(Q)*(transpose(W_new)*cov_x*W_new);
    [eigenvector,eigenvalue]=eig(matrix);
    [neweigenvalue,Idex] = sort(diag(eigenvalue),'descend');
    neweigenvalue=diag(neweigenvalue);
    neweigenvector=eigenvector(:,Idex);
    for i=1:b
        beta(:,i)=neweigenvector(:,i);
        W1_optimal = W_new*beta(:,i); %the first projec
        W_projection(:,i) = W1_optimal;
    end
    for parameter=-5:5
        alpha =10^(parameter);
        cofficient = pinv(transpose(W_projection)*transpose(train_x)*train_x*W_projection+alpha*eye(b,b))*transpose(W_projection)*transpose(train_x)*train_label;
        predict_label = test_x*(W_projection*cofficient);
        [prow pcolumn]=size(predict_label);
        for i = 1:prow
            if(abs(real(predict_label(i,1)))>=0.5)
                predict_label(i,1)=1;
            else
                predict_label(i,1)=0;
            end
        end
        [trow,tcolumn]=size(test_x);
        count =0;
        for j=1:trow
            if(predict_label(j,1)~=test_label(j,1))
                count =count+1;
            end
        end
        error_temp = count/trow;
        if(error_temp<data1(loop,1))
            data1(loop,1) = error_temp;
        end
        error_temp=0;
        [sub_error_min,sub_error_ma,Fair_SP,Fair_DI] = Calcualte(predict_label,test_label,loop,DataSample,randomset);
        data_minority(loop,1)=sub_error_min;
        data_majority(loop,1)=sub_error_ma;
        SP(loop,1)=Fair_SP;
        if(SP(loop,1)==0)
            DI(loop,1)=0;
        else
            DI(loop,1)=Fair_DI;
        end
    end
end
writetable(table(data1),'prediction_error.txt','Delimiter','\t');
writetable(table(SP),'Group fairness.txt','Delimiter','\t');
t2=toc;
display(strcat('parfor²¢ÐÐ¼ÆËãÊ±¼ä£º',num2str(t2),'Ãë'));
