function [W_fair,W_unfair] = NewThirdParty(sensitive,non_sensitive,W_old,covariance)
threshold=covariance; 
covariance=cov(sensitive,non_sensitive);
[a,b]=size(W_old);
W_fair =zeros(a,1);
W_unfair =zeros(a,1);
if(covariance(1,2)>0 && covariance(1,2)<=threshold)
    W_fair = W_old;
end
if(covariance(1,2) > threshold) 
    W_unfair = W_old; 
end
% if(abs(covariance(1,2))<=threshold)
%     W_fair = W_old;
% else
%     W_unfair = W_old; 
% end
end
