function result = iris_seg_up(data_list)

    %This function is for get upper boundary of iris
    a = [data_list{:}];
    x = cell2mat(a);
    I = double(reshape(x,536,536));
    I = rot90(I,3);
    I=flipdim(I,2);%原图像的水平镜像
    level=graythresh(double(I));
    I=im2bw(I,level);

    I = imclose(I,ones(2));
    I = find_maxarea(I);
    contour = bwperim(I);


    iris_dn = find_bwdown(contour);
    iris_up = contour - iris_dn;
    iris_up = find_maxarea(iris_up);
    %figure;imshow(iris_up);

    result = iris_up;
    %imwrite(iris_up,'a.jpg');

end
        
        
