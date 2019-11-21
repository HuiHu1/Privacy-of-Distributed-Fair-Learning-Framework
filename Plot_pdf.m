[p1,x1] = hist(covariance_before,100); 
[p2,x2] = hist(R1,100); 
[p3,x3] = hist(covariance_after,100); 
plot(x1,p1,'b-',x2,p2,'r--',x3,p3,'k-.'); %PDF
legend('Original Covariance','Gaussian Noise','Covariance after adding noise');
title('p.d.f');