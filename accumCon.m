function [A,j] = accumCon(A,thr)
[U,S,V]=svd(A);
sum=0;
for i=1:256%算一下累计贡献率
    sum=sum+S(i,i);
end
sum2=0;
con=zeros(1,256);
for i=1:256%算一下累计贡献率
    sum2=sum2+S(i,i);
    con(i)=sum2/sum;%累计贡献率
end
%记录到哪一个奇异值贡献率超过0.9
j=0;
for i=1:256
    if con(i)>=thr
        j=i;
        break;
    end
end
A=U(:,1:j)*S(1:j,1:j)*V(:,1:j)';
end

