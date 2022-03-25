clear all; close all; clc


temp = load("D:\BU\ChandLab\DLPFCRNN\PsychRNN\temp.mat").temp;
checker = readtable("D:\BU\ChandLab\DLPFCRNN\PsychRNN\checkerDLPFCTest.csv");

%% only choose trials with 95% RT
sortRT = sort(checker.decision_time);
disp("95% RT threshold is: " + num2str(sortRT(5000*0.95)))
% rtThresh = checker.decision_time <= sortRT(5000*0.95);
rtThresh = checker.decision_time >= 0 & checker.decision_time < sortRT(size(checker,1)*0.95) & checker.decision ~= 0;
checker = checker(rtThresh, :);
temp = temp(:,:,rtThresh);

[a, b, c] = size(temp);

%% align data to target onset

% reaction time; targetOn time and checkerOn time
RT = checker.decision_time;
targetOn = checker.target_onset;
checkerOn = checker.checker_onset;

% real RT, targetOn and checkerOn round to 10's digit
RTR = round(RT, -1);
targetOnR = round(targetOn,-1);
checkerOnR = round(checkerOn + targetOn, -1);

% left & right trials
right = checker.decision == 1;
left = checker.decision == -1;

% state activity alignes to checkerboard onset, with 500ms before and 2000
% ms after
alignState = [];
for ii = 1 : c
    zeroPt = targetOnR(ii)./10 + 1;
    alignState(:,:,ii) = temp(:,zeroPt - 25:zeroPt + 200, ii);
end

[a, b, c] = size(alignState);

%% 

RL = left & checker.chosen_color == 1;
RR = right & checker.chosen_color == 1;
GL = left & checker.chosen_color == -1;
GR = right & checker.chosen_color == -1;

firingRateAverage(:,1,1,:) = mean(alignState(:,:,RL),3);
firingRateAverage(:,1,2,:) = mean(alignState(:,:,RR),3);
firingRateAverage(:,2,1,:) = mean(alignState(:,:,GL),3);
firingRateAverage(:,2,2,:) = mean(alignState(:,:,GR),3);

for ii = 1:size(firingRateAverage, 1)
    temp = squeeze(firingRateAverage(ii,:,:,:));
    temp2 = [];
    for jj = 1:2
        for kk = 1:2
            temp2 = [temp2 squeeze(temp(jj,kk,:))'];
        end
    end
    processedFR(ii,:)= temp2;
end

test = processedFR';
[coeff, score, latent] = pca(test);
m = 4;
t = size(firingRateAverage,4);
orthF = [];
for thi = 1:m
    orthF(:,:,thi) = (score((1:t) + (thi-1)*t,:))';
end

traj = orthF([1,2,3], :, :);
RLTraj = traj(:,:,1);
RRTraj = traj(:,:,2);
GLTraj = traj(:,:,3);
GRTraj = traj(:,:,4);

% %% reshape data and do pca
% test = reshape(alignState, [a, b*c])';
% 
% [coeff, score, latent] = pca(test);
% orthF = [];
% for thi = 1 : c
%     orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
% end
% 
% %% plot trajectory based on color and direction 
% 
% RL = left & checker.chosen_color == 1;
% RR = right & checker.chosen_color == 1;
% GL = left & checker.chosen_color == -1;
% GR = right & checker.chosen_color == -1;
% 
% traj = orthF([1,2,3], :, :);
% 
% RLTraj = mean(traj(:,:,RL),3);
% RRTraj = mean(traj(:,:,RR),3);
% GLTraj = mean(traj(:,:,GL),3);
% GRTraj = mean(traj(:,:,GR),3);

%%
figure; 
plot3(RLTraj(1,:), RLTraj(2,:), RLTraj(3,:), 'r');
hold on
plot3(RRTraj(1,:), RRTraj(2,:), RRTraj(3,:), 'r--');
hold on
plot3(GLTraj(1,:), GLTraj(2,:), GLTraj(3,:), 'g');
plot3(GRTraj(1,:), GRTraj(2,:), GRTraj(3,:), 'g--');

plot3(RLTraj(1,25), RLTraj(2,25), RLTraj(3,25), 'k.', 'markersize', 50);

che = round(mean(checkerOnR)/10);

plot3(RLTraj(1,che), RLTraj(2,che), RLTraj(3,che), 'm.', 'markersize', 30);
plot3(RRTraj(1,che), RRTraj(2,che), RRTraj(3,che), 'm.', 'markersize', 30);
plot3(GLTraj(1,che), GLTraj(2,che), GLTraj(3,che), 'm.', 'markersize', 30);
plot3(GRTraj(1,che), GRTraj(2,che), GRTraj(3,che), 'm.', 'markersize', 30);
