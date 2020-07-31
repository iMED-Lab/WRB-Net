function [ z,s ] = find_bwup( bw )
%FIND_BWUP 此处显示有关此函数的摘要
%   此处显示详细说明
    bwup=[];
   [m,n]= size(bw);
    for i=1:n
        col=bw(:,i);
        if find(col==1,1)
            s(i)=find(col==1,1);
        end
    end

    for i=1:length(s)
        bwup(:,i)=[s(i);i];
    end
    bwup=bwup';
    s1=find(bwup(:,1)>0);
    s2=bwup(:,1);
    s2(find(s2==0))=[];
    s=[s2,s1];
    
    
    s1=s(:,1);
    s2=s(:,2);
    z=zeros(m,n);
    for i=1:length(s)
        z(s1(i),s2(i))=1;
    end

end

