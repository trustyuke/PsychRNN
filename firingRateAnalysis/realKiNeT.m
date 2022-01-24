% the kinet analysis of real pmd data
clear all; close all; clc
% for linux work station 
% temp = load("/home/tianwang/code/behaviorRNN/PsychRNNArchive/stateActivity/spikes.mat").trials;
% checker = load("~/Desktop/14October2013_Data.mat").forGPFA.dat;
% for Tian's PC
temp = load("C:\Users\tianwang/Downloads/spikes.mat").trials;
checker = load("C:\Users\tianwang/Downloads/14October2013_Data.mat").forGPFA.dat;;

checker = struct2table(checker);
RT = checker.RT;
RTR = round(RT, -1);

[a, b, c] = size(temp);


%% 

% rt = 300:100:800;
% right = checker.choice == 2;
% left = checker.choice == 1;
% 
% leftTraj = NaN(a,length(rt)-1,b);
% rightTraj = NaN(a,length(rt)-1,b);
% leftRT = NaN(length(rt)-1,1);
% rightRT = NaN(length(rt)-1,1);
% 
% for ii  = 1 : length(rt) - 1
%     selectedTrials = (rt(ii) < RTR & RTR < rt(ii + 1));
% 
%     leftSelect = selectedTrials & left;
%     rightSelect = selectedTrials & right;
%     leftTrajAve = mean(temp(:,:,leftSelect), 3);
%     rightTrajAve = mean(temp(:,:,rightSelect), 3);
%     
%     leftTraj(:,ii,:) = leftTrajAve;
%     rightTraj(:,ii,:) = rightTrajAve;
%     
%     % left and right average RT of each RT bin, should add 50 to be the
%     % real time from start
%     leftAveRT = round(mean(RTR(leftSelect))) + 60;
%     rightAveRT = round(mean(RTR(rightSelect))) + 60;
%     
%     leftRT(ii) = leftAveRT;
%     rightRT(ii) = rightAveRT;
%     
% end
% 
% 
% [a, b, c] = size(leftTraj);
% 
% test = reshape(leftTraj, [a, b*c])';
% 
% [coeff, score, latent] = pca(test);
% orthF = [];
% for thi = 1 : c
%     orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
% end
% 
% 
% addpath("./KiNeT-master");
% KiNeT(orthF(1:10, :,:),1);

%%
test = reshape(temp, [a, b*c])';

[coeff, score, latent] = pca(test);
orthF = [];
for thi = 1 : c
    orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
end

%%

rt = 300:100:800;
right = checker.choice == 2;
left = checker.choice == 1;

leftTraj = NaN(a,length(rt)-1,b);
rightTraj = NaN(a,length(rt)-1,b);
leftRT = NaN(length(rt)-1,1);
rightRT = NaN(length(rt)-1,1);

for ii  = 1 : length(rt) - 1
    selectedTrials = (rt(ii) < RTR & RTR < rt(ii + 1));

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = mean(orthF(:,:,leftSelect), 3);
    rightTrajAve = mean(orthF(:,:,rightSelect), 3);
    
    leftTraj(:,ii,:) = leftTrajAve;
    rightTraj(:,ii,:) = rightTrajAve;
    
    % left and right average RT of each RT bin, should add 50 to be the
    % real time from start
    leftAveRT = round(mean(RTR(leftSelect))) + 60;
    rightAveRT = round(mean(RTR(rightSelect))) + 60;
    
    leftRT(ii) = leftAveRT;
    rightRT(ii) = rightAveRT;
    
end



%% KiNeT analysis

addpath("./KiNeT-master");
KiNeT(leftTraj(:,:,:),1);
