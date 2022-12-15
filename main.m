%% 问题一的求解 矩阵之间的相关性(模方法)
clc;
clear;
errorAll=zeros(1,6);
V2All=zeros(64,2,384,4,6);%存储所有随机svd加模方法算出来的V
containerAll=zeros(384,10,6,4);%最后两个参数表示哪个数据集，哪一行
filepath=[pwd,'\A\H'];
for q=1:6
temp=load([filepath,'\Data',num2str(q),'_H','.mat']);
H=temp.H;

N=[];
for i=1:4%所有矩阵的二范数,相当于压缩了
    for j=1:384
        A=H(:,:,i,j);
        for k=1:64
            N(i,j,k)=norm(A(:,k));
        end
    end
end
Co=zeros(4,384,384);
for i=1:4
    for j=1:384
        for p=j:384
            temp=corrcoef(N(i,j,:),N(i,p,:));
            Co(i,j,p)=temp(1,2);
        end
    end
end
M=[];
l=1;
for i=1:4
    for j=1:384
        for k=j:384
            if Co(i,j,k)>=0.9986%筛选出来相关系数大于0.9986的矩阵关系，存储在M中
                M(l,1)=i;
                M(l,2)=j;
                M(l,3)=k;
                l=l+1;
            end
        end
    end
end

% 所有的算出来
V2=zeros(64,2,384,4);%V2是求出的所有矩阵V
for k=1:4
    container=julei(M,k);
    containerAll(1:size(container,1),1:size(container,2),q,k)=container;
    column=container(:,1);
    V1=zeros(64,2,size(column,1));
    for i=1:size(column)
        a=column(i);
        tmp=suiji(H(:,:,k,a));
        V1(:,:,i)=tmp(:,1:2);%求随机SVD
    end
    
    V3=zeros(64,2,384);
    for i=1:size(column)
        for j=1:size(container,2)
            if container(i,j)~=0
                V3(:,:,container(i,j))=V1(:,:,i);
            end
        end
    end
    
    V2(:,:,:,k)=V3;
end
V2All(:,:,:,:,q)=V2;
% 进行V的检验
temp1=load([filepath,'\Data',num2str(q),'_V','.mat']);
V=temp1.V;
P=zeros(2,4,384);
for i=1:4
    for j=1:384
        for l=1:2
            a=V(:,l,i,j);
            b=V2(:,l,j,i);
            P(l,i,j)=norm(a'*b)/norm(a)*norm(b);
        end
    end
end
errorAll(1,q)=min(min(min(P)));%V的建模精度
end

%% 第一题求解 写得到的V，写出.mat文件给Python程序运行
%写入excel文件V1.mat-V6.mat给python运行
V2final=zeros(256,768,6);
for k=1:6
    for j=1:384
        for i=1:4
            tmp=V2All(:,:,j,i,k);
            V2final((i-1)*64+1:i*64,(j-1)*2+1:j*2,k)=tmp;%此处的V用于下一步的对接
        end
    end
end
data1=V2final(:,:,1);
data2=V2final(:,:,2);
data3=V2final(:,:,3);
data4=V2final(:,:,4);
data5=V2final(:,:,5);
data6=V2final(:,:,6);
for i=1:6
    save([pwd,'\V',num2str(i)],['data',num2str(i)]);
end
%% 问题一的求解 再计算W 存储使用python计算的W的过程
%从python返回来的是data1.mat-data6.mat
dataAll=zeros(256,768,6);
for i=1:6
    tmp=load([pwd,'\T1\data',num2str(i),'.mat']);
    dataAll(:,:,i)=tmp.mat;
end

W_new=zeros(64,2,4,384,6);%用来存储所有的W数据
for i=1:6
    W_new(:,:,:,:,i)=cunchuW(dataAll(:,:,i));%升维处理
end
%% 第一题的解答 对python算出来的W进行检测
accuracy=zeros(6,1);%检测准确度
filepath=[pwd,'\A\W'];
for q=1:6
temp1=load([filepath,'\Data',num2str(q),'_W','.mat']);
W=temp1.W;
P=zeros(2,4,384);
for i=1:4
    for j=1:384
        for l=1:2
            a=W(:,l,i,j);
            b=W_new(:,l,i,j,q);
            P(l,i,j)=norm(a'*b)/norm(a)*norm(b);
        end
    end
end
accuracy(q,1)=min(min(min(P)));
end
%% 问题二的求解 优化方法3 (小分块压缩) 只能对H进行处理
errorAll=zeros(6,1);
countAll=zeros(6,1);
countSymmetry=0;%寻找对称
filepath=[pwd,'\A\H'];%测试时替换！！！
for q=1:6
temp=load([filepath,'\Data',num2str(q),'_H','.mat']);
H=temp.H;

%首先先把每个H进行分裂
H2=zeros(4,384,4,4,16);%H2是存储的所有小矩阵4*4
for i=1:4
    for j=1:384
        H1=H(:,:,i,j);
        for k=1:16
            H2(i,j,:,:,k)=H1(1:4,(k-1)*4+1:k*4);
        end
    end
end
%接着算每个小矩阵首元奇异值的贡献率
contribution=zeros(4,384,16);
location=[];
p=1;
for i=1:4
    for j=1:384
        for k=1:16
            H3=H2(i,j,:,:,k);
            tmp=reshape(H3,[4,4]);
            if isequal(tmp,tmp')|| isequal(tmp,tmp.')
                countSymmetry=countSymmetry+1;%寻找每个小矩阵是否有对称矩阵
            end
            [U3,S3,V3]=svd(tmp);
            contribution(i,j,k)=S3(1,1)/(S3(1,1)+S3(2,2)+S3(3,3)+S3(4,4));
            if contribution(i,j,k)>=0.749
                location(p,1)=i;
                location(p,2)=j;
                location(p,3)=k;
                p=p+1;
            end
        end
    end
end
countAll(q,1)=size(location,1);

H6=H;
for i=1:size(location,1)
    H4=H2(location(i,1),location(i,2),:,:,location(i,3));
    tmp=reshape(H4,[4,4]);
    [U,S,V]=svd(tmp);
    V=V';
    H5=U(1:4,1)*S(1,1)*V(1,1:4);
    H6(1:4,(location(i,3)-1)*4+1:location(i,3)*4, location(i,1),location(i,2))=H5;%H6是替换完的
end

sum1=0;
sum2=0;
for i=1:4
    for j=1:384
        a=H6(:,:,i,j);
        b=H(:,:,i,j);
        sum1=sum1+(norm(a-b,'fro'))^2;
        sum2=sum2+(norm(b,'fro'))^2;
    end
end
error=10*log10(sum1/sum2);
errorAll(q,1)=error;

end
spareSpace3H=7*countAll;%优化方法3节约的空间（H）
errorMinH=errorAll;%优化方法3的精度(H)
%% 问题二的求解 优化方法2 （大分块压缩） 先对W进行处理
errorAll=zeros(6,1);
thresAll=zeros(6,3);
filepath=[pwd,'\A\W'];
for k=1:6
temp=load([filepath,'\Data',num2str(k),'_W','.mat']);
W=temp.W;

W2D=zeros(256,768);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        W2D((i-1)*64+1:i*64,(j-1)*2+1:j*2)=W(:,:,i,j);
    end
end
%然后分三个256×256大块进行SVD分解
H2DD=zeros(256,768);%存储解压之后的值
threshold=zeros(1,3);%存储用了前多少列
for i=1:3
    [A,j]=accumCon(W2D(:,(i-1)*256+1:i*256),0.966);
    H2DD(:,(i-1)*256+1:i*256)=A;
    threshold(1,i)=j;
end
thresAll(k,:)=threshold;
sum1=0;
sum2=0;%计算误差
for i=1:4
    for j=1:384
        a=W2D((i-1)*64+1:i*64,(j-1)*2+1:j*2);
        b=H2DD((i-1)*64+1:i*64,(j-1)*2+1:j*2);
        sum1=sum1+(norm(a-b,'fro'))^2;
        sum2=sum2+(norm(b,'fro'))^2;
    end
end
error=10*log10(sum1/sum2);
errorAll(k,1)=error;
end
errorMedium=errorAll;%优化方法2的精度(W)
thres=zeros(6,1);
for i=1:6
    sum=0;
    for j=1:3
        sum=sum+thresAll(i,j);
    end
    thres(i,1)=sum;
end
thresAllMedium=thresAll;%优化方法2每个大块所需要取的奇异值数量（W）
spareSpace2=ones(6,1)*196608-513*thres;%优化方法2节省的空间（W）
%% 问题二的求解 优化方法2 （大分块压缩） 再对H进行处理
errorAll=zeros(6,1);
thresAll=zeros(6,6);
filepath=[pwd,'\A\H'];
for k=1:6
temp=load([filepath,'\Data',num2str(k),'_H','.mat']);
H=temp.H;

H2D=zeros(256,1536);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        H2D((i-1)*64+1:i*64,(j-1)*4+1:j*4)=H(:,:,i,j).';%要转置
    end
end
%然后分六个256×256大块进行SVD分解
H2DD=zeros(256,1536);%存储解压之后的值
threshold=zeros(1,6);%存储用了前多少列
for i=1:6
    [A,j]=accumCon(H2D(:,(i-1)*256+1:i*256),0.966);
    H2DD(:,(i-1)*256+1:i*256)=A;
    threshold(1,i)=j;
end
thresAll(k,:)=threshold;
sum1=0;
sum2=0;%计算误差
for i=1:4
    for j=1:384
        a=H2D((i-1)*64+1:i*64,(j-1)*4+1:j*4);
        b=H2DD((i-1)*64+1:i*64,(j-1)*4+1:j*4);
        sum1=sum1+(norm(a-b,'fro'))^2;
        sum2=sum2+(norm(b,'fro'))^2;
    end
end
error=10*log10(sum1/sum2);
errorAll(k,1)=error;
end
errorMediumH=errorAll;%优化方法2的精度(H)
thres=zeros(6,1);
for i=1:6
    sum=0;
    for j=1:6
        sum=sum+thresAll(i,j);
    end
    thres(i,1)=sum;
end
thresAllMediumH=thresAll;%优化方法2每个大块所需要取的奇异值数量（H）
spareSpace2H=ones(6,1)*393216-513*thres;%优化方法2节省的空间（H）
%% 问题二的求解 优化方法1 （不分块处理W！）
errorAll=zeros(6,1);
thresAll=zeros(6,1);
filepath=[pwd,'\A\W'];
for k=1:6
temp=load([filepath,'\Data',num2str(k),'_W','.mat']);
W=temp.W;

W2D=zeros(256,768);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        W2D((i-1)*64+1:i*64,(j-1)*2+1:j*2)=W(:,:,i,j);
    end
end
%然后直接进行SVD分解
% H2DD=zeros(256,768);%存储解压之后的值
[A,j]=accumCon(W2D,0.974);
threshold=j;%存储用了前多少列
thresAll(k,1)=threshold;
H2DD=A;

sum1=0;
sum2=0;%计算误差
for i=1:4
    for j=1:384
        a=W2D((i-1)*64+1:i*64,(j-1)*2+1:j*2);
        b=H2DD((i-1)*64+1:i*64,(j-1)*2+1:j*2);
        sum1=sum1+(norm(a-b,'fro'))^2;
        sum2=sum2+(norm(b,'fro'))^2;
    end
end
error=10*log10(sum1/sum2);
errorAll(k,1)=error;
end
spareSpace1=ones(6,1)*196608-1025*thresAll;%优化方法1节省的空间（W）
thresAllMax=thresAll;%优化方法1所需要取的奇异值数量（W）
errorMax=errorAll;%优化方法1的精度(W)
%% 问题二的求解 优化方法1 （不分块处理H！）
errorAll=zeros(6,1);
thresAll=zeros(6,1);
filepath=[pwd,'\A\H'];
for k=1:6
temp=load([filepath,'\Data',num2str(k),'_H','.mat']);
H=temp.H;

H2D=zeros(256,1536);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        H2D((i-1)*64+1:i*64,(j-1)*4+1:j*4)=H(:,:,i,j).';%要转置
    end
end
%直接进行SVD分解
H2DD=zeros(256,1536);%存储解压之后的值
[A,j]=accumCon(H2D,0.947);
H2DD=A;
threshold=j;

thresAll(k,1)=threshold;
sum1=0;
sum2=0;%计算误差
for i=1:4
    for j=1:384
        a=H2D((i-1)*64+1:i*64,(j-1)*4+1:j*4);
        b=H2DD((i-1)*64+1:i*64,(j-1)*4+1:j*4);
        sum1=sum1+(norm(a-b,'fro'))^2;
        sum2=sum2+(norm(b,'fro'))^2;
    end
end
error=10*log10(sum1/sum2);
errorAll(k,1)=error;
end
spareSpace1H=ones(6,1)*393216-1793*thresAll;%优化方法1节省的空间（H）
thresAllMaxH=thresAll;%优化方法1所需要取的奇异值数量（H）
errorMaxH=errorAll;%优化方法1的精度(H)
%% 问题三的求解 先压缩解压矩阵H
%大分块压缩 256*256
errorAll=zeros(6,1);
thresAll=zeros(6,6);
HAll=zeros(4,64,4,384,6);%存储解压缩后的H
filepath=[pwd,'\A\H'];
for k=1:6
temp=load([filepath,'\Data',num2str(k),'_H','.mat']);
H=temp.H;

H2D=zeros(256,1536);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        H2D((i-1)*64+1:i*64,(j-1)*4+1:j*4)=H(:,:,i,j).';%要转置
    end
end
%然后分六个256×256大块进行SVD分解
H2DD=zeros(256,1536);%存储解压之后的值
for i=1:6
    [A,j]=accumCon(H2D(:,(i-1)*256+1:i*256),0.966);
    H2DD(:,(i-1)*256+1:i*256)=A;
end
%接着对压缩后的H升维度
for i=1:4
    for j=1:384
        HAll(:,:,i,j,k)=H2DD((i-1)*64+1:i*64,(j-1)*4+1:j*4).';
    end
end
end

%% 第三题的求解 再进行随机SVD及模方法
errorAll=zeros(1,6);
V2All=zeros(64,2,384,4,6);%存储所有随机svd加模方法算出来的V
containerAll=zeros(384,10,6,4);%最后两个参数表示哪个数据集，哪一行
filepath=[pwd,'\A\H'];
for q=1:6
H=HAll(:,:,:,:,q);%使用解压后的矩阵H

N=[];
for i=1:4%所有矩阵的二范数,相当于压缩了
    for j=1:384
        A=H(:,:,i,j);
        for k=1:64
            N(i,j,k)=norm(A(:,k));
        end
    end
end
Co=zeros(4,384,384);
for i=1:4
    for j=1:384
        for p=j:384
            temp=corrcoef(N(i,j,:),N(i,p,:));
            Co(i,j,p)=temp(1,2);
        end
    end
end
M=[];
l=1;
for i=1:4
    for j=1:384
        for k=j:384
            if Co(i,j,k)>=0.9986%筛选出来相关系数大于0.9986的矩阵关系，存储在M中
                M(l,1)=i;
                M(l,2)=j;
                M(l,3)=k;
                l=l+1;
            end
        end
    end
end

% 所有的算出来
V2=zeros(64,2,384,4);%V2是求出的所有矩阵V
for k=1:4
    container=julei(M,k);
    containerAll(1:size(container,1),1:size(container,2),q,k)=container;
    column=container(:,1);
    V1=zeros(64,2,size(column,1));
    for i=1:size(column)
        a=column(i);
        tmp=suiji(H(:,:,k,a));
        V1(:,:,i)=tmp(:,1:2);%求随机SVD
    end
    
    V3=zeros(64,2,384);
    for i=1:size(column)
        for j=1:size(container,2)
            if container(i,j)~=0
                V3(:,:,container(i,j))=V1(:,:,i);
            end
        end
    end
    
    V2(:,:,:,k)=V3;
end
V2All(:,:,:,:,q)=V2;
% 进行V的检验
temp1=load([filepath,'\Data',num2str(q),'_V','.mat']);
V=temp1.V;
P=zeros(2,4,384);
for i=1:4
    for j=1:384
        for l=1:2
            a=V(:,l,i,j);
            b=V2(:,l,j,i);
            P(l,i,j)=norm(a'*b)/norm(a)*norm(b);
        end
    end
end
errorAll(1,q)=min(min(min(P)));
end
%% 第三题求解 写得到的V，把data1-data6给Python程序运行
%写入excel文件
V2final=zeros(256,768,6);
for k=1:6
    for j=1:384
        for i=1:4
            tmp=V2All(:,:,j,i,k);
            V2final((i-1)*64+1:i*64,(j-1)*2+1:j*2,k)=tmp;
        end
    end
end
data1=V2final(:,:,1);
data2=V2final(:,:,2);
data3=V2final(:,:,3);
data4=V2final(:,:,4);
data5=V2final(:,:,5);
data6=V2final(:,:,6);
for i=1:6
    save([pwd,'\V',num2str(i)],['data',num2str(i)]);
end
%% 第三题的求解 存储使用python计算的W的过程
dataAll=zeros(256,768,6);
for i=1:6
    tmp=load([pwd,'\T3\data',num2str(i),'.mat']);
    dataAll(:,:,i)=tmp.mat;
end
W_new=zeros(64,2,4,384,6);%用来存储所有的W数据
for i=1:6
    W_new(:,:,:,:,i)=cunchuW(dataAll(:,:,i));%升维处理
end
%% 第三题的求解 再对W进行降维压缩
WAll=zeros(64,2,4,384,6);%存储解压缩后的W
for k=1:6
W=W_new(:,:,:,:,k);

W2D=zeros(256,768);%先把四维矩阵展成二维矩阵
for i=1:4
    for j=1:384
        W2D((i-1)*64+1:i*64,(j-1)*2+1:j*2)=W(:,:,i,j);
    end
end
%然后分三个256×256大块进行SVD分解
H2DD=zeros(256,768);%存储解压之后的值
for i=1:3
    [A,j]=accumCon(W2D(:,(i-1)*256+1:i*256),0.966);
    H2DD(:,(i-1)*256+1:i*256)=A;
end
for i=1:4
    for j=1:384
        WAll(:,:,i,j,k)=H2DD((i-1)*64+1:i*64,(j-1)*2+1:j*2);
    end
end
end
%% 第三题的求解 对W的检验
accuracy=zeros(6,1);%检测准确度
filepath=[pwd,'\A\W'];
for q=1:6
temp1=load([filepath,'\Data',num2str(q),'_W','.mat']);
W=temp1.W;
P=zeros(2,4,384);
for i=1:4
    for j=1:384
        for l=1:2
            a=W(:,l,i,j);
            b=WAll(:,l,i,j,q);
            P(l,i,j)=norm(a'*b)/norm(a)*norm(b);
        end
    end
end
accuracy(q,1)=min(min(min(P)));
end