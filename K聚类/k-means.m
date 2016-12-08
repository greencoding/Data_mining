[m n] = size(Iris);
u = zeros(5,n);
j = 1;
for i=1:5
    u(j,:)=Iris(i,:);
    j = j + 1;
end
k=5;
while 1
    pre_u = u;
    quan = zeros(m,k);
    for i=1:m
        c=[];
        for j=1:k
            c=[c norm(Iris(i,:) - u(j,:))];
        end
        [value index] = min(c);
        quan(i,index) = 1;
    end
    for i=1:k
        for j=1:n
            u(i,j)=sum(quan(:,i).*Iris(:,j))/sum(quan(:,i));
        end
    end
    if norm(u - pre_u)<0.1
        break;
    end
end
re =[];
for i=1:m
    tmp=[];
    for j=1:k
        tmp=[tmp norm(Iris(i,:)-u(j,:))];
    end
    [value index] = min(tmp);
    re = [re;Iris(i,:) index];
end
SSE = 0
for i=1:m
    SSE = SSE + norm(Iris(i,:) - u(re(i,5),:))
end