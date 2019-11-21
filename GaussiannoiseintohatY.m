%----------------------------Adding gaussian noise into hat_Y----------
function [new_fair,new_unfair] = GaussiannoiseintohatY(total,train_x,train_Sensitive,R,featureNum,covariance)
[row,column]=size(train_Sensitive);
u1=0; 
sigma1=0.1; 
R1 = normrnd(u1,sigma1,row,total);
for i=1:total
    Y_hat(:,i)=train_x*R(:,i);
    temp0 = cov(Y_hat(:,i),train_Sensitive);
    covariance_before(i,1)= temp0(1,2);
    Tuta_Y_hat(:,i)=Y_hat(:,i)+R1(:,i);
    temp1 = cov(Tuta_Y_hat(:,i),train_Sensitive);
    covariance_after(i,1) =  temp1(1,2);
end
new_fair = zeros(featureNum,total);
new_unfair = zeros(featureNum,total);
for i=1:total
    if(covariance_after(i,1)<=covariance && covariance_after(i,1)>0)
        new_fair(:,i) = R(:,i); 
    end
    if(covariance_after(i,1)>covariance)
        new_unfair(:,i) = R(:,i);
    end
end
new_fair(:,all(new_fair==0)) = [];
new_unfair(:,all(new_unfair==0)) = [];
end