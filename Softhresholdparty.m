function [pro] = Softhresholdparty(Pre_Hat,loop,DataSample,randomset)
trainset = randomset(1:500,loop);
train_Sensitive=DataSample(trainset(:,1),1);
[x1,y1]= size(Pre_Hat); 
cov_value = zeros(y1,1);
pro = zeros(y1,1);
for i=1:y1
    cov_matrix=cov(train_Sensitive,Pre_Hat(:,i));
    cov_value(i,1) = cov_matrix(1,2);
end
for i=1:y1
    pro(i,1) = 1/(1+exp(cov_value(i,1)));
end
end

