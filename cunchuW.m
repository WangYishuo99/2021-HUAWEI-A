function [W_new11] = cunchuW(num1)
W_new11=zeros(64,2,4,384);
for i=1:4
    for j=1:384
        W_new11(:,:,i,j)=num1((i-1)*64+1:i*64,(j-1)*2+1:j*2);
    end
end
end

