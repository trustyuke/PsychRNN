clear all; close all; clc
% On linux work station (for checkerPmd)

% vanilla RNN
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/temp.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdBasic2InputNoise0.75.csv");

% RNN with g0 & gSlope additive
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainA.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain3Additive.csv");

% RNN with g0 additive
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainAg0.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain3g0.csv");


% RNN with multiplicative gain
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainM.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain4Multiply.csv");
% 


% On Tian's PC (for checkerPmd)

% vanilla RNN
% temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\temp.mat").temp;
% checker = readtable("D:/BU/chandLab/PsychRNN/resultData/checkerPmdBasic2InputNoise0.75.csv");

% RNN with g0 & gSlope additive
% temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\gainA.mat").temp;
% checker = readtable("D:/BU/chandLab/PsychRNN/resultData/checkerPmdGain3Additive.csv");

% RNN with g0 additive
% temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\gainAg0.mat").temp;
% checker = readtable("D:\BU\ChandLab\PsychRNN\resultData\checkerPmdGain3g0.csv");


% RNN with multiplicative gain
temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\gainM.mat").temp;
checker = readtable("D:/BU/chandLab/PsychRNN/resultData/checkerPmdGain4Multiply.csv");

%% get r from x
% % vanilla
% temp = max(temp, 0);

% additive
% for id = 1 : size(temp, 3)
%     tempGain = checker.g0(id);
%     temp(:,:,id) = temp(:, :,id) + tempGain;
% end
% temp = max(temp, 0);

% multiplicative
for id = 1 : size(temp, 3)
    tempGain = checker.g0(id);
    temp(:,:,id) = temp(:,:,id).*tempGain;
end
temp = max(temp, 0);

% only choose trials with 95% RT
sortRT = sort(checker.decision_time);
disp("95% RT threshold is: " + num2str(sortRT(5000*0.95)))
rtThresh = checker.decision_time <= sortRT(5000*0.95);
checker = checker(rtThresh, :);
temp = temp(:,:,rtThresh);

[a, b, c] = size(temp);



%% look at 1 coh bin 

bin1 = (checker.coherence_bin == 0.5 | checker.coherence_bin == -0.5);

checker = checker(bin1, :);
temp = temp(:,:,bin1);
%% align data to checkerboard onset (target onset)

RT = checker.decision_time;
targetOn = checker.target_onset;
checkerOn = checker.checker_onset;
RTR = round(RT, -1);
targetOnR = round(targetOn,-1);
checkerOnR = round(checkerOn + targetOn, -1);

% state activity alignes to checkerboard onset, with 500ms before and 2000
% ms after
alignState = [];
for ii = 1 : c
    zeroPt = checkerOnR(ii)./10 + 1;
    alignState(:,:,ii) = temp(:,zeroPt - 50:zeroPt + 100, ii);
end

[a, b, c] = size(alignState);


%% directly generated 7 conditions and then do pca

% rt = 100:50:450;
rt =[100:100:800]
% rt = [100:100:800 1200]
right = checker.decision == 1;
left = checker.decision == 0;

leftTraj = NaN(a,length(rt)-1,b);
rightTraj = NaN(a,length(rt)-1,b);
leftRT = NaN(length(rt)-1,1);
rightRT = NaN(length(rt)-1,1);

aveGain0 = [];
aveGainS = [];
for ii  = 1 : length(rt) - 1
    selectedTrials = (rt(ii) < RTR & RTR < rt(ii + 1));
    sum(selectedTrials)
    
    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = mean(alignState(:,:,leftSelect), 3);
    rightTrajAve = mean(alignState(:,:,rightSelect), 3);
    
    aveGain0(ii) = mean(checker.g0(leftSelect));
    aveGainS(ii) = mean(checker.gSlope(leftSelect));
%     
    leftTraj(:,ii,:) = leftTrajAve;
    rightTraj(:,ii,:) = rightTrajAve;
    
    % left and right average RT of each RT bin, should add 50 to be the
    % real time from start
    leftAveRT = round(mean(RTR(leftSelect))./10) + 50;
    rightAveRT = round(mean(RTR(rightSelect))./10) + 50;
    
    leftRT(ii) = leftAveRT;
    rightRT(ii) = rightAveRT;
    
end

%% reshape leftTraj & rightTraj and then do PCA

[a, b, c] = size(rightTraj);

test = reshape(rightTraj, [a, b*c])';

[coeff, score, latent] = pca(test);
orthF = [];
for thi = 1 : c
    orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
end


addpath("./KiNeT-master");
KiNeT(orthF(1:5, :,:),1);

cc = jet(size(orthF, 2));

figure()
for ii = 1 : size(orthF, 2)
    plot(squeeze(orthF(1,ii,:)), 'color', cc(ii,:));  
    hold on
end


%% plot raw ave firing rate for all RT bins
meanSpike = squeeze(mean(leftTraj, 1));
cc = jet(size(meanSpike, 1));

figure()
for ii = 1 : size(meanSpike, 1)
    plot(meanSpike(ii,:)', 'color', cc(ii,:));  
    hold on
end
%% reshape data and do pca
test = reshape(alignState, [a, b*c])';

[coeff, score, latent] = pca(test);
orthF = [];
for thi = 1 : c
    orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
end

%% based on RT
% the trials definitely have more trials with fastere RT, so the long RT
% trajs are messay

% total data: 500ms before checkerboard onset to 2000ms after checkerboard
% onset. So max RT that can be plotted is 2000ms
rt = 100:100:800;
right = checker.decision == 1;
left = checker.decision == 0;

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
    leftAveRT = round(mean(RTR(leftSelect))./10) + 50;
    rightAveRT = round(mean(RTR(rightSelect))./10) + 50;
    
    leftRT(ii) = leftAveRT;
    rightRT(ii) = rightAveRT;
    
end

%% KiNeT analysis

addpath("./KiNeT-master");
KiNeT(leftTraj,1);
