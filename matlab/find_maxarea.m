function [ bw ] = find_maxarea( BW )
%FIND_MAXAREA 此处显示有关此函数的摘要
%   此处显示详细说明
    im = bwlabel(BW);
    stats = regionprops(im,'Area');
    [b,index]=sort([stats.Area],'descend');
    bw = ismember(im,index(1));

end

