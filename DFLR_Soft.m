%---------------------fix H------------------------
clear
clc;
tic
t1=toc;
DataSample = csvread('data/communitycrime/crimecommunity.csv');
randomset = csvread('data/communitycrime/crimecommunity_index.csv');
covariance=0.1;
iter=20;
featureNum=99;
total=300; 
u=0;
sigma=1;
R = normrnd(u,sigma,featureNum,total);
for i=1:iter
    data1(i,1)=30;
    SP(i,1)=30;
    DI(i,1)=30;
    data_minority(i,1)=30;
    data_majority(i,1)=30;
end
hypo = zeros(iter,1);
for loop=1:iter
    normal_data = DataSample(:,2:100);
    trainset = randomset(1:500,loop);
    testset = randomset(1494:1993,loop);
    train_x = normal_data(trainset(:,1),:);
    train_label = DataSample(trainset(:,1),102);
    train_Sensitive=DataSample(trainset(:,1),1);
    test_x = normal_data(testset(:,1),:);
    test_label = DataSample(testset(:,1),102);
     %---------------------------------soft_threshold------------------
    [x1,y1]=size(train_x);
    Pre_Hat= zeros(x1,total);
    for i=1:total
        Pre_Hat(:,i)= train_x*R(:,i);
    end
    Probabiity= Softhresholdparty(Pre_Hat,loop,DataSample,randomset); % store the probability returned
    C = cumsum(Probabiity);
    dimension=total;
    i = fix(rand*floor(C(end)))+1;
    index = find(C >= i,1);
    f = zeros(dimension,1);
    for i=1:dimension
        rand_number =  min(C) + (C(end)-min(C)).*rand(1,1);
        for j=1:dimension
            if(C(j,1)>=rand_number)
                f(i,1)=j;
                break;
            end
        end
    end
%     for i=1:dimension
%        f(i,1) = 1+sum(C(end)*rand>C);
%     end
    [fair_index,m1,n1] = unique(f);
    [x1,y1]=size(fair_index);
    W_fair=zeros(featureNum,x1);
    for i=1:x1
       W_fair(:,i) = R(:,fair_index(i,1)); 
    end
    [a b]=size(W_fair);
    hypo(loop,1)=b;
    if(b~=0)
        dimension=b;
        W_new =W_fair;
        for parameter=-3:3
            alpha =10^(parameter);
            cofficient = pinv(transpose(W_new)*transpose(train_x)*train_x*W_new+alpha*eye(dimension,dimension))*transpose(W_new)*transpose(train_x)*train_label;
            prediction_label = test_x*(W_new*cofficient);
            [prow pcolumn]=size(prediction_label);
            for i = 1:prow
                if(abs(prediction_label(i,1))>=0.5)
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
%-------------------------deviation----------------------------
diviation_SP = std(SP);
diviation_Error = std(data1);
t2=toc;
display(strcat('parfor²¢ÐÐ¼ÆËãÊ±¼ä£º',num2str(t2),'Ãë'));
