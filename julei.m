function [container] = julei(M,y)
k=1;
container=[];%第一个大行的
container(1,1)=1;
count=1;
loc=2;
y=4;
for i=2:size(M,1)
    if M(i,1)==y
       if i<k
           continue;
       end
       if M(i,2)==M(i,3) && ismember(M(i,2),container) && M(i,2)<384
           x=M(i,2)+1;
           temp=find(M(:,2)==M(:,3 ) & M(:,3)==x);
           k=temp(y);
           continue;
       end
       if M(i,2)==M(i,3)&&ismember(M(i,2),container)==false
           count=count+1;
           loc=1;
           container(count,loc)=M(i,2);
           loc=loc+1;
       end
       if  M(i,2)~=M(i,3)&&ismember(M(i,3),container)==false
           container(count,loc)=M(i,3);
           loc=loc+1;
       end
    end
end
end

