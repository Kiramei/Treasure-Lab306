close all;
clear all;
clc;
load('digits4000.mat')

numPC=300;
is1vs9=0;
isPCA=0;
isHOG=1;
isSVM=1;
isSTD=1;
% testX1 = digits_vec(:,testset(1,:)); % get test data (trial 1)
% testX2 = digits_vec(:,testset(2,:)); % get test data (trial 2)
% testXimg = reshape(testX, [28 28 1 2000]); % turn into an image sequence
% montage(uint8(testXimg), 'size', [40 50]); % view as a montage
trial_num=2;
for i=1:trial_num
    TrainData=digits_vec(:,trainset(i,:))';
    trainLabel=digits_labels(1,trainset(i,:))';
    TestData=digits_vec(:,testset(i,:))';
    testLabel=digits_labels(1,testset(i,:))';
    % 主要代码部分
    % Preprocess
    if isPCA
        % 主成分分析降维
        [coeff, ~] = pca(TrainData);
        TrainData = TrainData * coeff(:, 1:numPC);
        TestData = TestData * coeff(:, 1:numPC);
    end
    
    if isHOG
        fprintf('You have chosen to use HOG. Now processing...\n')
        TrainDataImg = reshape(TrainData, [size(TrainData, 1) 28 28]);
        TestDataImg = reshape(TestData, [size(TestData, 1) 28 28]);
        
        TrainDataHOG = zeros(size(TrainData, 1), 36);
        TestDataHOG = zeros(size(TestData, 1), 36);
        
        for j = 1:size(TrainData, 1)
            to_hog = squeeze(TrainDataImg(j, :, :));
            corners = detectFASTFeatures(im2gray(to_hog));
            strongest = selectStrongest(corners,3);
            TrainDataHOG(j, :) = extractHOGFeatures(to_hog,'CellSize',[14 14]);
        end
        
        for j = 1:size(TestData, 1)
            to_hog = squeeze(TestDataImg(j, :, :));
            corners = detectFASTFeatures(im2gray(to_hog));
            strongest = selectStrongest(corners,3);
            TestDataHOG(j, :) = extractHOGFeatures(to_hog,'CellSize',[14 14]);
        end
        
        TrainData = TrainDataHOG;
        TestData = TestDataHOG;
        fprintf('HOG feature extract finished.\n')
    end
    
    if isSTD
        % Data Standardization
        TrainData = zscore(TrainData);
        TestData = zscore(TestData);
    end
    
    % BiTest
    if is1vs9
        % 1-vs-9 BiTest
        trainLabel(trainLabel ~= 1) = -1;
        testLabel(testLabel ~= 1) = -1;
    end
    % t = templateSVM('KernelFunction','linear');
    fprintf('Training SVM model...\n');
    tic
    mdl = fitcecoc(TrainData,trainLabel);
    consume_for_train = toc;
    fprintf('The Training Process has fininshed.\n')
    
    fprintf('Testing SVM model...\n');
    tic
    result = predict(mdl,TestData);
    consume_for_test = toc;
    fprintf('The Training Process has fininshed.\n')
    
    % accuracy = sum(result == testLabel) / numel(testLabel) * 100;
    
    % accuracy(i) = sum(result == testLabel) / numel(testLabel) * 100;
    % tp = sum(result == 1 & testLabel == 1);
    % fp = sum(result == 1 & testLabel ~= 1);
    % fn = sum(result ~= 1 & testLabel == 1);
    % f1_score(i) = 2 * tp / (2 * tp + fp + fn);
    
    accuracy(i) = sum(result == testLabel) / numel(testLabel) * 100;
    precision = sum(result == testLabel & result == 1) / sum(result == 1) * 100;
    recall = sum(result == testLabel & result == 1) / sum(testLabel== 1) * 100;
    f1_score(i) = 2 * precision * recall / (precision + recall);
    
    fprintf('--------------------------\n');
    fprintf('Training Time : %d sec.\n',consume_for_train);
    fprintf('Testing Time  : %d sec.\n',consume_for_test);
    fprintf('Accuracy      : %0.2f %%.\n' ,accuracy(i));
    fprintf('Precision     : %0.2f .\n' ,precision);
    fprintf('Recall        : %0.2f .\n' ,recall);
    fprintf('F1 Score      : %0.2f .\n' ,f1_score(i));
    fprintf('--------------------------\n');
    
    if i == 1
        fprintf('Now reverse the train and test data!\n')
    else
        fprintf('All task has finished!\n')
    end
    
    % Iterate for svm, however its computation
    % consumption is respectively high
    %
    % mdls = [];
    % for y=1:9
    %     label_1=ones(200*y,1);
    %     label_2=-ones(1800,1);
    %     label=[label_1;label_2];
    %     st=round(200*(y-1))+1;
    %     r = TrainData(st:2000,:);
    %     Mdl=fitcsvm(r,label);
    %     mdls(end+1) = Mdl;
    % end
end
avg_accuracy = mean(accuracy);
avg_f1_score = mean(f1_score);
fprintf('Average Accuracy: %.2f%%\n', avg_accuracy);
fprintf('Average F1 Score: %.2f\n', avg_f1_score);