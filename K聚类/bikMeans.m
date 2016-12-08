biK = 5;
biDataSet = Iris;
[row,col] = size(biDataSet);
% 存储质心矩阵
biCentSet = zeros(biK,col);
% 初始化设定cluster数量为1
numCluster = 1;
%第一列存储每个点被分配的质心，第二列存储点到质心的距离
biClusterAssume = zeros(row,2);
%初始化质心
biCentSet(1,:) = mean(biDataSet);
for i = 1:row
    biClusterAssume(i,1) = numCluster;
    biClusterAssume(i,2) = distEclud(biDataSet(i,:),biCentSet(1,:));
end
while numCluster < biK
    minSSE = 10000;
    %寻找对哪个cluster进行划分最好，也就是寻找SSE最小的那个cluster
    for j = 1:numCluster
        curCluster = biDataSet(find(biClusterAssume(:,1) == j),:);
        [spiltCentSet,spiltClusterAssume] = kMeans(curCluster,2);
        spiltSSE = sum(spiltClusterAssume(:,2));
        noSpiltSSE = sum(biClusterAssume(find(biClusterAssume(:,1)~=j),2));
        curSSE = spiltSSE + noSpiltSSE;
        fprintf('第%d个cluster被划分后的误差为：%f \n' , [j, curSSE])
        if (curSSE < minSSE)
            minSSE = curSSE;
            bestClusterToSpilt = j;
            bestClusterAssume = spiltClusterAssume;
            bestCentSet = spiltCentSet;
        end
    end

     %更新cluster的数目  
    numCluster = numCluster + 1;
    %必须先更新2的为新的类，再更新1的为原来的类，不然的话，当对第二个cluster进行划分的时候，就会被全部分为同一个类  
    bestClusterAssume(find(bestClusterAssume(:,1) == 2),1) = numCluster;
    bestClusterAssume(find(bestClusterAssume(:,1) == 1),1) = bestClusterToSpilt;

    
    %更新和添加质心坐标  第一行为更新质心，第二行为添加质心  
    biCentSet(bestClusterToSpilt,:) = bestCentSet(1,:);
    biCentSet(numCluster,:) = bestCentSet(2,:);

    % 更新被划分的cluster的每个点的质心分配以及误差 
    biClusterAssume(find(biClusterAssume(:,1) == bestClusterToSpilt),:) = bestClusterAssume;
end




