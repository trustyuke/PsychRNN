clear all; close all; clc
% for linux work station 
% temp = load("/home/tianwang/code/behaviorRNN/PsychRNNArchive/stateActivity/temp.mat").temp;
% checker = readtable("/home/tianwang/code/behaviorRNN/PsychRNN/resultData/basic2InputNoise0.5.csv");

% for Tian's PC
temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\temp.mat").temp;
checker = readtable("D:/BU/chandLab/PsychRNN/resultData/basic2InputNoise0.5.csv");

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

%% plot first trial all state activity
plot(temp(:,:,1)')
%%

right = checker.decision == 1;
left = checker.decision == 0;
leftTrajAve = sum(orthF(:,:,left), 3);
rightTrajAve = sum(orthF(:,:,right), 3);

%% general average of left and right
figure(); 
plot3(leftTrajAve(1,:), leftTrajAve(2,:),leftTrajAve(3,:));
hold on
plot3(leftTrajAve(1,1), leftTrajAve(2,1),leftTrajAve(3,1), 'ro', 'markersize', 10);

plot3(rightTrajAve(1,:), rightTrajAve(2,:),rightTrajAve(3,:));
plot3(rightTrajAve(1,1), rightTrajAve(2,1),rightTrajAve(3,1), 'ro', 'markersize', 10);

legend('left', 'right')

%% based on coherence: combined same coh trials

figure();
coh = unique(checker.coherence_bin);
for ii  = 1 : floor(length(coh)/2)
    selectedTrials = (checker.coherence_bin == coh(ii) | checker.coherence_bin == -coh(ii));

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = sum(orthF(:,:,leftSelect), 3);
    rightTrajAve = sum(orthF(:,:,rightSelect), 3);

    plot3(leftTrajAve(1,:), leftTrajAve(2,:),leftTrajAve(3,:), 'color', [0,1,0].*ii.*0.1, 'linestyle', '--');
    hold on
    plot3(rightTrajAve(1,:), rightTrajAve(2,:),rightTrajAve(3,:), 'color', [0,0,1].*ii.*0.1);
    pause()
end

%% based on coherence: not combining same coh trials


cc = jet(19);
figure();
coh = unique(checker.coherence_bin);
for ii  = 1 : length(coh)
    selectedTrials = checker.coherence_bin == coh(ii);

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = sum(orthF(:,:,leftSelect), 3);
    rightTrajAve = sum(orthF(:,:,rightSelect), 3);
    
    leftAveRT = round(mean(RTR(leftSelect))./10) + 50;
    rightAveRT = round(mean(RTR(rightSelect))./10) + 50;
%     plot3(leftTrajAve(1,:), leftTrajAve(2,:),leftTrajAve(3,:), 'color', [0,1,0].*ii.*0.05, 'linestyle', '--');
%     hold on
%     plot3(rightTrajAve(1,:), rightTrajAve(2,:),rightTrajAve(3,:), 'color', [0,0,1].*ii.*0.05);
%     
    plot3(leftTrajAve(1,1:leftAveRT), leftTrajAve(2,1:leftAveRT),leftTrajAve(3,1:leftAveRT), 'color', cc(ii,:), 'linestyle', '--');
    hold on
    % mark the checkerboard onset
    plot3(leftTrajAve(1,50), leftTrajAve(2,50),leftTrajAve(3,50), 'color', cc(ii,:), 'marker', '.', 'markersize', 10);

    plot3(rightTrajAve(1,1:rightAveRT), rightTrajAve(2,1:rightAveRT),rightTrajAve(3,1:rightAveRT), 'color', cc(ii,:));
    hold on
    % mark the checkerboard onset
    plot3(rightTrajAve(1,50), rightTrajAve(2,50),rightTrajAve(3,50), 'color', cc(ii,:), 'marker', '.', 'markersize', 10);
    
%     pause()
end

%% based on RT

RT = checker.decision_time;

rt = 0:300:3000;

% blue to red as RT increases
% left: --; right: -
figure();
for ii  = 1 : length(rt) - 1
    selectedTrials = (rt(ii) < RT & RT < rt(ii + 1));

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = sum(orthF(:,:,leftSelect), 3);
    rightTrajAve = sum(orthF(:,:,rightSelect), 3);

    plot3(leftTrajAve(1,:), leftTrajAve(2,:),leftTrajAve(3,:), 'color', [ii*0.1,0,1-ii*0.1], 'linestyle', '--');
    hold on
    plot3(rightTrajAve(1,:), rightTrajAve(2,:),rightTrajAve(3,:), 'color', [ii*0.1,0,1-ii*0.1]);
    pause()
end

