clc
clear
SP=[0.1416	
0.2666	
0.254	
0.3193	
0.4124	
0.3898	
0.3883	
0.3899	
0.366	
0.3926	
];	
Classifier_error=[0.2400	
0.2736	
0.264	
0.36	
0.386	
0.402	
0.4145	
0.436	
0.4176	
0.4132	
];	
Infer_error=[0.2770	
0.1531	
0.1834	
0.1766	
0.1748	
0.1931	
0.2418	
0.3376	
0.4089	
0.4557];	
checkpoints =[0.1:0.1:1];
% plot(checkpoints,SP,'b*-',checkpoints,Classifier_error,'R+-',checkpoints,Infer_error,'go-');
% title('Performance versus H')
% legend('SP','Classifier error','Infer error')
% xlabel('Number of hypertheses (H)')
% ylabel('Values')
% ylim([0,0.5]);
yyaxis left
plot(checkpoints,SP,'-');
ylim([0.14 0.42])
ylabel('SP')
xlabel('\rho')

yyaxis right
plot(checkpoints,Classifier_error,'--',checkpoints,Infer_error,':');
ylim([0.15 0.46])
ylabel('Error')
legend('SP','Classifier error','Infer error')
%plot(checkpoints,data1,'b*-',checkpoints,SP,'R+-',checkpoints,DI,'go-');
%title('Group fairness and Error vs.Number of fair hypertheses(DFGR)')
