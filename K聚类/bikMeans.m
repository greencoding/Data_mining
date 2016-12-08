biK = 5;
biDataSet = Iris;
[row,col] = size(biDataSet);
% �洢���ľ���
biCentSet = zeros(biK,col);
% ��ʼ���趨cluster����Ϊ1
numCluster = 1;
%��һ�д洢ÿ���㱻��������ģ��ڶ��д洢�㵽���ĵľ���
biClusterAssume = zeros(row,2);
%��ʼ������
biCentSet(1,:) = mean(biDataSet);
for i = 1:row
    biClusterAssume(i,1) = numCluster;
    biClusterAssume(i,2) = distEclud(biDataSet(i,:),biCentSet(1,:));
end
while numCluster < biK
    minSSE = 10000;
    %Ѱ�Ҷ��ĸ�cluster���л�����ã�Ҳ����Ѱ��SSE��С���Ǹ�cluster
    for j = 1:numCluster
        curCluster = biDataSet(find(biClusterAssume(:,1) == j),:);
        [spiltCentSet,spiltClusterAssume] = kMeans(curCluster,2);
        spiltSSE = sum(spiltClusterAssume(:,2));
        noSpiltSSE = sum(biClusterAssume(find(biClusterAssume(:,1)~=j),2));
        curSSE = spiltSSE + noSpiltSSE;
        fprintf('��%d��cluster�����ֺ�����Ϊ��%f \n' , [j, curSSE])
        if (curSSE < minSSE)
            minSSE = curSSE;
            bestClusterToSpilt = j;
            bestClusterAssume = spiltClusterAssume;
            bestCentSet = spiltCentSet;
        end
    end

     %����cluster����Ŀ  
    numCluster = numCluster + 1;
    %�����ȸ���2��Ϊ�µ��࣬�ٸ���1��Ϊԭ�����࣬��Ȼ�Ļ������Եڶ���cluster���л��ֵ�ʱ�򣬾ͻᱻȫ����Ϊͬһ����  
    bestClusterAssume(find(bestClusterAssume(:,1) == 2),1) = numCluster;
    bestClusterAssume(find(bestClusterAssume(:,1) == 1),1) = bestClusterToSpilt;

    
    %���º������������  ��һ��Ϊ�������ģ��ڶ���Ϊ�������  
    biCentSet(bestClusterToSpilt,:) = bestCentSet(1,:);
    biCentSet(numCluster,:) = bestCentSet(2,:);

    % ���±����ֵ�cluster��ÿ��������ķ����Լ���� 
    biClusterAssume(find(biClusterAssume(:,1) == bestClusterToSpilt),:) = bestClusterAssume;
end




