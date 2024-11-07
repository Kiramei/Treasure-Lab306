% 读取ORL56_46数据集
path = './Dataset/ORL/';
X = [];
y = [];

for i = 1:40
    for j = 1:10
        img = imread([path, sprintf('s%d/%d.pgm', i, j)]);
        X = [X; img(:)'];
        y = [y; i];
    end
end
X = double(X);
y = double(y);

% 对数据进行PCA降维
X_2d = pca(X, 'NumComponents', 2);
X_3d = pca(X, 'NumComponents', 3);

% 将数据集可视化在二维空间
figure;
colors = rand(40, 3);
for i = 1:10
    scatter(X_2d((i-1)*10+1:i*10, 1), X_2d((i-1)*10+1:i*10, 2), 30, colors(i, :), 'filled', 'DisplayName', ['Person ', num2str(i)]);
    hold on;
end
hold off;
legend('show');
title('2D Visualization of ORL56_46 Subsets');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

% 将数据集可视化在三维空间
figure;
ax = axes('Projection','orthographic');
for i = 1:10
    x_ = X_3d((i-1)*10+1:i*10, 1);
    y_ = X_3d((i-1)*10+1:i*10, 2);
    z_ = X_3d((i-1)*10+1:i*10, 3);

    scatter3(ax, x_, y_, z_, 30, colors(i, :), 'filled', 'DisplayName', ['Person ', num2str(i)]);
    hold on;
end
hold off;
legend('show');
title('3D Visualization of ORL56_46 Subsets');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');