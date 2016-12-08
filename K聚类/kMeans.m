function [centSet,clusterAssment] = kMeans(dataSet,K)

[row,col] = size(dataSet);
% �洢���ľ���
centSet = zeros(K,col);
% �����ʼ������
for i= 1:col
    minV = min(dataSet(:,i));
    if isempty(minV)
        break;
    end
    rangV = max(dataSet(:,i)) - minV;
    centSet(:,i) = bsxfun(@plus,minV,rangV*rand(K,1));
end

% ���ڴ洢ÿ���㱻�����cluster�Լ������ĵľ���
clusterAssment = zeros(row,2);
clusterChange = true;
while clusterChange
    clusterChange = false;
    % ����ÿ����Ӧ�ñ������cluster
    for i = 1:row
        %�ⲿ�ֿ��ܿ����Ż�      
        minDist = 10000;
        minIndex = 0;
        for j = 1:K
            distCal = distEclud(dataSet(i,:) , centSet(j,:));
            if (distCal < minDist)
                minDist = distCal;
                minIndex = j;
            end
        end
        if minIndex ~= clusterAssment(i,1)            
            clusterChange = true;
        end
        clusterAssment(i,1) = minIndex;
        clusterAssment(i,2) = minDist;
    end
    
    % ����ÿ��cluster ������ 
    for j = 1:K
        simpleCluster = find(clusterAssment(:,1) == j);
        centSet(j,:) = mean(dataSet(simpleCluster',:));
    end
end
end