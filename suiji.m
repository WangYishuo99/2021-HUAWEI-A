function [V1] = suiji(A)
Q=[];
w=normrnd(0,1,size(A,2),1);
y=A*w;
q=y;
q_new=q/norm(q);
Q=horzcat(Q,q_new);
count1=0;
while norm(A-Q*Q'*A)>0.000001
    w=normrnd(0,1,size(A,2),1);
    y=A*w;
    q=(eye(size(A,1))-Q*Q')*y;
    q_new=q/norm(q);
    
    Q=horzcat(Q,q_new);
    count1=count1+1;
end
B=Q'*A;
[u,s,V1]=svd(B);

end

