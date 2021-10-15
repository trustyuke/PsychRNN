clear all; close all; clc
% for linux work station 
% temp = load("/home/tianwang/code/behaviorRNN/PsychRNNArchive/stateActivity/temp.mat").temp;
% checker = readtable("/home/tianwang/code/behaviorRNN/PsychRNN/resultData/basic2InputNoise0.5.csv");

% for Tian's PC
temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\temp.mat").temp;
checker = readtable("D:/BU/chandLab/PsychRNN/resultData/basic2InputNoise0.5.csv");

% for checkerPmd
% temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\state.mat").state;
% checker = readtable("D:/BU/chandLab/PsychRNN/resultData/checkerPmdInputNoise0.25recNoise0.5.csv");


[a, b, c] = size(temp);

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
    alignState(:,:,ii) = temp(:,zeroPt - 50:zeroPt + 200, ii);
end

[a, b, c] = size(alignState);

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
rt = 0:200:1800;
right = checker.decision == 1;
left = checker.decision == 0;

leftTraj = NaN(a,9,b);
rightTraj = NaN(a,9,b);
leftRT = NaN(9,1);
rightRT = NaN(9,1);

for ii  = 1 : length(rt) - 1
    selectedTrials = (rt(ii) < RTR & RTR < rt(ii + 1));

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = sum(orthF(:,:,leftSelect), 3);
    rightTrajAve = sum(orthF(:,:,rightSelect), 3);
    
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
