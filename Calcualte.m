function [sub_error_min,sub_error_ma,Fair_SP,Fair_DI] = Calcualte(prediction_label,test_label,loop,DataSample,randomset)
testset = randomset(1494:1993,loop);
test_Sensitive=DataSample(testset(:,1),1);
class_Minority=size(find(test_Sensitive(:,1)==1)); 
class_Majority=size(find(test_Sensitive(:,1)==0)); 
sub_min_number=0;
sub_ma_number=0;
for q=1:length(test_label)
    if(prediction_label(q,1)~=test_label(q,1)&&(test_Sensitive(q,1)==1))
        sub_min_number = sub_min_number+1;
    end
    if(prediction_label(q,1)~=test_label(q,1)&&(test_Sensitive(q,1)==0))
        sub_ma_number = sub_ma_number+1;
    end
end
sub_error_min=sub_min_number/class_Minority(1,1);
sub_error_ma=sub_ma_number/class_Majority(1,1);
Minority = 0;
Majority = 0;
for g=1:length(test_label)
    if((test_Sensitive(g,1)==1) && prediction_label(g,1)==1)
        Minority = Minority+1;
    end
    if((test_Sensitive(g,1)==0) && prediction_label(g,1)==1)
        Majority = Majority+1;
    end
end
ratio_Minority =  Minority/class_Minority(1,1);
ratio_Majority =  Majority/class_Majority(1,1);
Fair_SP=abs(ratio_Minority - ratio_Majority); %SP
if(ratio_Minority<=ratio_Majority)
    Fair_DI=abs(1 - (ratio_Minority)/ratio_Majority);
else
    Fair_DI=abs(1 - (ratio_Majority)/ratio_Minority);
end