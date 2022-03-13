clear all; close all; clc
% addpath('/net/derived/tianwang/LabCode');

% On linux work station (for checkerPmd)

% vanilla RNN
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/temp.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdBasic2InputNoise0.75.csv");

% RNN with g0 & gSlope additive
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainA.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain3Additive.csv");

% RNN with g0 additive
temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainAg0.mat").temp;
checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain3g0.csv");


% RNN with multiplicative gain
% temp = load("/net/derived/tianwang/psychRNNArchive/stateActivity/gainM.mat").temp;
% checker = readtable("~/code/behaviorRNN/PsychRNN/resultData/checkerPmdGain4Multiply.csv");



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

temp = load("D:\BU\ChandLab\PsychRNN\temp.mat").temp;
checker = readtable("D:\BU\ChandLab\PsychRNN\\gainInput.csv");

% RNN with multiplicative gain
% temp = load("D:\BU\ChandLab\PsychRNNArchive\stateActivity\gainM.mat").temp;
% checker = readtable("D:/BU/chandLab/PsychRNN/resultData/checkerPmdGain4Multiply.csv");


%% get r from x
<<<<<<< Updated upstream
% % vanilla
% temp = max(temp, 0);

% additive
gainThresh = round(checker.decision_time + checker.target_onset + checker.checker_onset, -1)/10;
for id = 1 : size(temp, 3)
    tempGain = checker.g0(id);
    for idx = 1:gainThresh(id)
        temp(:,idx,id) = temp(:, idx,id) + tempGain;
    end
end
temp = max(temp, 0);

% multiplicative
% for id = 1 : size(temp, 3)
%     tempGain = checker.g0(id);
%     temp(:,:,id) = temp(:,:,id).*tempGain;
% end
% temp = max(temp, 0);

=======
>>>>>>> Stashed changes

% only choose trials with 95% RT
sortRT = sort(checker.decision_time);
disp("95% RT threshold is: " + num2str(sortRT(5000*0.95)))
% rtThresh = checker.decision_time <= sortRT(5000*0.95);
rtThresh = checker.decision_time >= 100;

checker = checker(rtThresh, :);
temp = temp(:,:,rtThresh);

[a, b, c] = size(temp);

%% align data to checkerboard onset (target onset)
RT = checker.decision_time;
targetOn = checker.target_onset;
checkerOn = checker.checker_onset;
targetOnR = round(targetOn,-1);
checkerOnR = round(checkerOn + targetOn, -1);

left = checker.decision == 0;
right = checker.decision == 1;
coh = checker.coherence;
% state activity alignes to checkerboard onset, with 500ms before and 1000
% ms after
alignState = [];
for ii = 1 : c
    zeroPt = checkerOnR(ii)./10 + 1;
    alignState(:,:,ii) = temp(:,zeroPt - 50:zeroPt + 100, ii);
end

[a, b, c] = size(alignState);

%%

trials1 = alignState(:,:,left);
trials2 = alignState(:,:,right);

% decoder to predict RT on choice 1 (without using predictRT function)
r2 = zeros(size(trials1,2), 1);

train_x = trials1;
train_y = RT(left);

for ii = 1 : size(train_x,2)
%     t1 = [squeeze(train_x(:,ii,:))', coh(left)];
    t1 = [squeeze(train_x(:,ii,:))'];
    
    md1 = fitrlinear(t1, train_y, 'learner', 'leastsquares');

    label = predict(md1, t1);
    R = corrcoef(label, train_y);
    R2 = R(1,2).^2;
    r2(ii) = R2;
end

% 
% tic
% % shuffled r2 of choice 1    
% shuffled_r2 = zeros(100, size(trials1,2));
% 
% for sIdx = 1 : 100
% 
%     R = randperm(size(trials1,3));
%     train_x = trials1;
%     temp = RT(left);
%     train_yS = temp(R);
%     
%     for ii = 1 : size(train_x,2)
% 
% %         t1 = [squeeze(train_x(:,ii,:))', coh(left)];
%         t1 = [squeeze(train_x(:,ii,:))'];
%         md1 = fitrlinear(t1, train_yS, 'learner', 'leastsquares');
% 
%         label = predict(md1, t1);
%         R = corrcoef(label, train_yS);
%         R2 = R(1,2).^2;
%         shuffled_r2(sIdx, ii) = R2;    
%     end 
% 
% end
    
% % calculate bound accuarcy
% bounds = zeros(2, size(trials1,2));
% percentile = 100/size(shuffled_r2,1);
% bounds(1,:) = prctile(shuffled_r2, percentile, 1);
% bounds(2,:) = prctile(shuffled_r2, 100 - percentile, 1);
% 
% toc

%% plot regression

figure; hold on

t = linspace(-500,1000,151);

ylimit = 0.8;
xpatch = [-500 -500 0 0];
ypatch = [ylimit 0 0 ylimit];
p1 = patch(xpatch, ypatch, 'cyan');
p1.FaceAlpha = 0.2;
p1.EdgeAlpha = 0;

% plot(t, bounds', '--', 'linewidth', 5);
plot(t, r2, 'linewidth', 5, 'color', [236 112  22]./255)
plot([0,0], [ylimit,0], 'color', [0.5 0.5 0.5], 'linestyle', '--', 'linewidth',5)
title('Regression on RT', 'fontsize', 30)


% cosmetic code
hLimits = [-500,1000];
hTickLocations = -500:300:1000;
hLabOffset = 0.05;
hAxisOffset =  -0.011;
hLabel = "Time: ms"; 

vLimits = [0,ylimit];
vTickLocations = [0 ylimit/2 ylimit];
vLabOffset = 150;
vAxisOffset = -520;
vLabel = "R^{2}"; 

plotAxis = [1 1];

[hp,vp] = getAxesP(hLimits,...
    hTickLocations,...
    hLabOffset,...
    hAxisOffset,...
    hLabel,...
    vLimits,...
    vTickLocations,...
    vLabOffset,...
    vAxisOffset,...
    vLabel, plotAxis);

set(gcf, 'Color', 'w');
axis off; 
axis square;
axis tight;


save('./resultData/boundAr.mat', 'bounds');
save('./resultData/r2Ar.mat', 'r2');
print('-painters','-depsc',['./resultFigure/', 'RTAr','.eps'], '-r300');

%%

% decoder to predict RT on choice 2

 [r2, shuffled_r2, bounds] = predictRT(trials2, RT(right), coh(right));

%% decoder to predict RT on choice 2 (without using predictRT function)


figure;
plot(bounds', '--');
hold on
plot(r2)
xlabel('Bin number')
ylabel('Variance explained')
title('Choice 2 Variance explained of binned spike counts')
xline(50, 'color', [0.5 0.5 0.5], 'linestyle', '--')
xpatch = [0 0 50 50];
ypatch = [0 1 1 0];
p1 = patch(xpatch, ypatch, 'cyan');
p1.FaceAlpha = 0.2;
p1.EdgeAlpha = 0;
xlim([1,250])
ylim([-0.02,1])


%% pre
 r2 = zeros(size(trials2,2), 1);

train_x = trials2;
train_y = RT(right);

for ii = 1 : size(train_x,2)
    t1 = [squeeze(train_x(:,ii,:))', coh(right)];
    md1 = fitrlinear(t1, train_y, 'learner', 'leastsquares');

    label = predict(md1, t1);
    R = corrcoef(label, train_y);
    R2 = R(1,2).^2;
    r2(ii) = R2;
end



% shuffled r2 of choice2    
shuffled_r2 = zeros(100, size(trials2,2));

for sIdx = 1 : 100

    R = randperm(size(trials2,3));
    train_x = trials2;
    temp = RT(right);
    train_yS = temp(R);
    
    for ii = 1 : size(train_x,2)

        t1 = [squeeze(train_x(:,ii,:))', coh(right)];
        md1 = fitrlinear(t1, train_yS, 'learner', 'leastsquares');

        label = predict(md1, t1);
        R = corrcoef(label, train_yS);
        R2 = R(1,2).^2;
        shuffled_r2(sIdx, ii) = R2;    
    end 

end
    
% calculate bound accuarcy
bounds = zeros(2, size(trials1,2));
percentile = 100/size(shuffled_r2,1);
bounds(1,:) = prctile(shuffled_r2, percentile, 1);
bounds(2,:) = prctile(shuffled_r2, 100 - percentile, 1);
