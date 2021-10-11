function [r2, shuffled_r2, bounds] = predictRT(trials, RT, coh)

% input:
%     trials: the stateActivity data: #units * #timestep * #trials
%     coh: coherence of the trials: #trials * 1
%     RT: reaction time: # trials * 1 
%     
% output:
%     r2: R^2 of the regression
%     shuffled_r2: all R^2 after 100 times shuffle
%     bounds: 1 and 99 percentile of shuffled_r2


% decoder to predict RT on choice 1
r2 = zeros(size(trials,2), 1);

train_x = trials;
train_y = RT;

for ii = 1 : size(train_x,2)
    t1 = [squeeze(train_x(:,ii,:))', coh];
    md1 = fitrlinear(t1, train_y, 'learner', 'leastsquares');

    label = predict(md1, t1);
    R = corrcoef(label, train_y);
    R2 = R(1,2).^2;
    r2(ii) = R2;
end


tic
% shuffled r2 of choice 1    
shuffled_r2 = zeros(100, size(trials,2));

for sIdx = 1 : 100

    R = randperm(size(trials,3));
    train_x = trials;
    temp = RT;
    train_yS = temp(R);
    
    for ii = 1 : size(train_x,2)

        t1 = [squeeze(train_x(:,ii,:))', coh];
        md1 = fitrlinear(t1, train_yS, 'learner', 'leastsquares');

        label = predict(md1, t1);
        R = corrcoef(label, train_yS);
        R2 = R(1,2).^2;
        shuffled_r2(sIdx, ii) = R2;    
    end 

end
bounds = zeros(2, size(trials,2)); 
percentile = 100/size(shuffled_r2,1);

% calculate bound accuarcy
bounds(1,:) = prctile(shuffled_r2, percentile, 1);
bounds(2,:) = prctile(shuffled_r2, 100 - percentile, 1);

end

