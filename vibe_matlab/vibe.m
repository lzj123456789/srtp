clear all;
%算法参数初始化，N表示2D欧式坐标图像中，每个像素点建立的背景采样个数；R表示欧式距离阀值；每个像素
%
matlabpool close
matlabpool open local 2
N=20;
R=50;
MIN=6;
Q=16;
xyloObj  = VideoReader('traffic.avi');
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
mov(1:nFrames) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);
% Preallocate movie structure.用于存放灰度视频帧
image(1:nFrames) = ...
     struct('cdata', zeros(vidHeight, vidWidth, 1, 'uint8'),...
            'colormap', []);
for k = 1 : nFrames
    mov(k).cdata = read(xyloObj, k);
    image(k).cdata = rgb2gray(mov(k).cdata );
%     imshow(image(k).cdata);
end
samples=zeros(vidHeight, vidWidth, N, 'uint8');
segMap = zeros(vidHeight, vidWidth, 1, 'uint8');
foregroundMatchCount = zeros(vidHeight, vidWidth, 1, 'uint8');
background=0;
foreground=255;
parfor k = 1 : N
    for x=1:vidHeight
        for y=1:vidWidth
          xNG=getRandomNeighbrXCoordinate( x,vidHeight);
          yNG=getRandomNeighbrYCoordinate( y,vidWidth);
          samples(x,y,k)=image(1).cdata(xNG,yNG);  
        end
    end
end
% imshow(samples(:,:,20));
for k = 1 : 200
    for x=1:vidHeight
        for y=1:vidWidth
            count=0;
            index=1;
            dist=0;
            while((count<MIN)&&(index<=N))
%                dist= pdist2(mov(k).cdata(x,y),samples(x,y,index),'Euclidean');
             dist=abs(double(mov(k).cdata(x,y))-double(samples(x,y,index)));
               if(dist<R)
                   count=count+1;
               end
               index=index+1;
            end
               if(count>=MIN)
                   foregroundMatchCount(x,y)=0;
                   segMap(x,y)=background;
               rand=randi([1 Q],1,1);
               if(rand==1)
                   rand=randi([1 N],1,1);
                   samples(x,y,rand)=image(k).cdata(x,y);
               end
               rand=randi([1 Q],1,1);
               if(rand==1)
                   xNG=getRandomNeighbrXCoordinate( x,vidHeight);
                   yNG=getRandomNeighbrYCoordinate( y,vidWidth);
                   rand=randi([1 N],1,1);
                   samples(xNG,yNG,rand)=image(k).cdata(x,y);
               end
               else
                   foregroundMatchCount(x,y)= foregroundMatchCount(x,y)+1;
                   segMap(x,y)=foreground;
                   if(foregroundMatchCount(x,y)>50)
                    rand=randi([1 Q],1,1);
                    if(rand==1)
                     rand=randi([1 N],1,1);
                     samples(x,y,rand)=image(k).cdata(x,y);
                    end 
                   end
               end
        end
    end
    figure(2);
    clf;
    subplot(2,2,1),imshow(samples(:,:,randi([1 N],1,1)));
    subplot(2,2,2),imshow(uint8(segMap));
    subplot(2,2,3),imshow(uint8(mov(k).cdata))
end
matlabpool close
