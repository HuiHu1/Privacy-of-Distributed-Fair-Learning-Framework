error1 = [0.3320
0.2580
0.2500
0.2860
0.2800
0.2720];
error2 = [0.3120
0.1820
0.2141
0.1700
0.2360
0.2230];
error3 =[0.2740
0.1500
0.1540
0.1446
0.1623
0.1752];
checkpoints =[1 10 20 30 40 50];
plot(checkpoints,error1,'-',checkpoints,error2,'-',checkpoints,error3,'-');
legend('Infer Error(H_i=40)','Infer Error(H_i=60)','Infer Error(H_i=100)')
xlabel('k')
ylabel('Error Rate')
ylim([0.14,0.36]);

% plot(checkpoints,error_rate_40,'-',checkpoints,error_rate_60,'--',checkpoints,error_rate_100,':');
% % title('Performance versus H')
% legend('Infer error(H_i=40)','Infer error(H_i=60)','Infer error(H_i=100)')
% xlabel('Number of base models (k)')
% ylabel('error rate')
% ylim([0.14,0.5]);
% set(gca, 'YTick')
% str=[repmat(9,1) num2str(error_rate)];
% plot(checkpoints,error_rate,'-o')
% text(checkpoints,error_rate,cellstr(str))