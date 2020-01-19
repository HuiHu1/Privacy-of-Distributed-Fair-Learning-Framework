clear
clc;
tic
t1=toc;
DataSample = csvread('data/communitycrime/crimecommunity.csv'); 
randomset = csvread('data/communitycrime/crimecommunity_index.csv');
%R = csvread('data/fix_h.csv');
featureNum=99;
total=300; 
u=0;
sigma=1;
R = normrnd(u,sigma,featureNum,total);
%-----------------------------------------------------------------------
covariance=0.1;
iteration=9; 
select = 500;
error_rate = zeros(iteration,1);
infer_sum = zeros(select,iteration);
for loop=1:iteration
    trainset = randomset(1:select,loop);
    train_x = DataSample(trainset(:,1),2:100);
    train_label = DataSample(trainset(:,1),102);
    train_Sensitive=DataSample(trainset(:,1),1);
%     W_fair=zeros(featureNum,total);
%     W_unfair=zeros(featureNum,total);
%     for i=1:total
%      [W_fair(:,i),W_unfair(:,i)]=NewThirdParty(train_Sensitive,train_x*R(:,i),R(:,i),covariance);
%     end
%     W_fair(:,all(W_fair==0)) = [];
%     W_unfair(:,all(W_unfair==0)) = [];
%---------------------------------soft_threshold------------------
    [x1,y1]=size(train_x);
    Pre_Hat= zeros(x1,total);
    for i=1:total
        Pre_Hat(:,i)= train_x*R(:,i);
    end
    Probabiity= Softhresholdparty(Pre_Hat,loop,DataSample,randomset); % store the probability returned
    C = cumsum(Probabiity);
    dimension=total;
    f = zeros(dimension,1);
    for i=1:dimension
       f(i,1) = 1+sum(C(end)*rand>C);
    end
    [fair_index,m1,n1] = unique(f,'first');
    [x1,y1]=size(fair_index);
    W_fair=zeros(featureNum,x1);
    for i=1:x1
       W_fair(:,i) = R(:,fair_index(i,1)); 
    end
    W_unfair=zeros(featureNum,total-x1);
    unfair_index_temp = zeros(total,1);
    for j=1:total
        if (ismember(j,fair_index)~=1) 
            unfair_index_temp(j,1)=j;
        end
    end
    unfair_index = unfair_index_temp(find(unfair_index_temp));
    for j=1:(total-x1)
        W_unfair(:,j) = R(:,unfair_index(j,1)); 
    end
    %-----------------------------------------------------------------
    [fa fb]=size(W_fair);
    [fc, fd]=size(W_unfair);
    r = select;
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
    n = select;
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