function [ z, s ] = find_bwdown( im)
    bwdown=[];s=[];
    [m,n]= size(im);
    for i=1:n
        col=im(:,i);
        if find(col==1,1)
            s(i)=find(col==1,1,'last');
        end
    end

    for i=1:length(s)
        bwdown(:,i)=[s(i);i];
    end
    bwdown=bwdown';
    s1=find(bwdown(:,1)>0);
    s2=bwdown(:,1);
    s2(find(s2==0))=[];
    s=[s2,s1];

    s1=s(:,1);
    s2=s(:,2);
    z=zeros(m,n);
    for i=1:length(s)
        z(s1(i),s2(i))=1;
    end
end

