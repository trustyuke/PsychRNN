test = load("temp2.mat").temp2;
temp = load("temp.mat").temp;
checker = readtable("/home/tianwang/code/behaviorRNN/PsychRNN/resultData/basic2InputNoise0.5.csv");
[a, b, c] = size(temp);

test = reshape(temp, [a, b*c])';

[coeff, score, latent] = pca(test);
orthF = [];
for thi = 1 : c
    orthF(:,:,thi) = (score( (1:b) + (thi-1)*b, :))';
end

%% 
plot(test(1:b,:))

%% 

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

figure();
coh = unique(checker.coherence_bin);
for ii  = 1 : length(coh)
    selectedTrials = checker.coherence_bin == coh(ii);

    leftSelect = selectedTrials & left;
    rightSelect = selectedTrials & right;
    leftTrajAve = sum(orthF(:,:,leftSelect), 3);
    rightTrajAve = sum(orthF(:,:,rightSelect), 3);

    plot3(leftTrajAve(1,:), leftTrajAve(2,:),leftTrajAve(3,:), 'color', [0,1,0].*ii.*0.05, 'linestyle', '--');
    hold on
    plot3(rightTrajAve(1,:), rightTrajAve(2,:),rightTrajAve(3,:), 'color', [0,0,1].*ii.*0.05);
    pause()
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

