load Dataset\Umist\umist_resized.mat
data_X = X';
data_y = label;


y_axis_data = [];
for r = 10:10:160
    x_value = [];
    y_value = [];
    fprintf('当降维到%d时\n', r);
    k = 17;
    [train_face, train_label, test_face, test_label] = Loaddata(data_X, data_y, k);
    [data_train_new, data_mean, V_r] = PCA(train_face, r);
    num_train = size(data_train_new, 1);
    num_test = size(test_face, 1);
    temp_face = test_face - repmat(data_mean, num_test, 1);
    data_test_new = temp_face * V_r;
    true_num = 0;
    for i = 1:num_test
        testFace = data_test_new(i, :);
        diffMat = data_train_new - repmat(testFace, num_train, 1);
        sqDiffMat = diffMat .^ 2;
        sqDistances = sum(sqDiffMat, 2);
        [~, indexMin] = min(sqDistances);
        if train_label(indexMin) == test_label(i)
            true_num = true_num + 1;
        end
    end
    accuracy = true_num / num_test;
    y_axis_data = [y_axis_data; accuracy * 100];
    fprintf('当每个人选择%d张照片进行训练时，The classify accuracy is: %.2f%%\n', k, accuracy * 100);
end

x_axis_data = 10:10:160;
plot(x_axis_data, y_axis_data, 'b*--', 'LineWidth', 1, 'MarkerSize', 5);
for i = 1:length(x_axis_data)
    text(x_axis_data(i), y_axis_data(i) + 0.3, sprintf('%.0f', y_axis_data(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 7.5);
end

legend('17');
xlabel('dimension');
ylabel('Recognition rate');

function [newdata, meanVal] = Centralization(dataMat)
    [rows, cols] = size(dataMat);
    meanVal = mean(dataMat, 1);
    newdata = dataMat - repmat(meanVal, rows, 1);
end

function [train_face, train_label, test_face, test_label] = Loaddata(data_X, data_y, k)
    imgsize = 32 * 32;
    train_face = zeros(k * 20, imgsize);
    train_label = zeros(k * 20, 1);
    test_face = zeros(575, imgsize);
    test_label = zeros(575, 1);
    rec = int32(1);
    j = int32(1);
    for i = 1:575
        peopleID = data_y(i);
        imgVector = data_X(i,:);
        if peopleID == rec 
            if j <= k 
                ind = fix((rec - 1) * 20 + j);
                train_face(ind, :) = imgVector;
                train_label(ind) = peopleID;
                j = j + 1;
            else
                j = 0;
                rec = rec + 1;
            end
        end
        test_face(i, :) = imgVector;
        test_label(i) = peopleID;
    end
end

function [final_data, meanVal, V_r] = PCA(data, r)
    dataMat = single(data);
    [A, meanVal] = Centralization(dataMat);
    covMat = A * A';
    [V, D] = eig(covMat);
    [D, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    V_r = V(:, 1:r);
    V_r = A' * V_r;
    for i = 1:r
        V_r(:, i) = V_r(:, i) / norm(V_r(:, i));
    end
    final_data = A * V_r;
end
