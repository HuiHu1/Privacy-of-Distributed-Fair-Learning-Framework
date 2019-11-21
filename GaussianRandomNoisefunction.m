%--------------------Adding gaussian noise into covariance value---------
function [new_fair,new_unfair] = GaussianRandomNoisefunction(total,train_x,train_Sensitive,R,featureNum,covariance)
u1=0; 
sigma1=1; % control the noise
R1 = normrnd(u1,sigma1,total,1);
for i=1:total
    Y_hat(:,i)=train_x*R(:,i);
    temp0 = cov(Y_hat(:,i),train_Sensitive);
    covariance_before(i,1)= temp0(1,2);
    covariance_after(i,1) = covariance_before(i,1)+R1(i,1);
end
new_fair = zeros(featureNum,total);
new_unfair = zeros(featureNum,total);
for i=1:total
    if(covariance_after(i,1)>0 && covariance_after(i,1)<=covariance)
        new_fair(:,i) = R(:,i); 
    end
    if(covariance_after(i,1)>covariance)
        new_unfair(:,i) = R(:,i);
    end
end
new_fair(:,all(new_fair==0)) = [];
new_unfair(:,all(new_unfair==0)) = [];
end