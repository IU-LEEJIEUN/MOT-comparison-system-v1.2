# import os
# import cv2
#
# # 定义图像文件夹路径和视频输出路径
# image_folder = './test_video/video/uav0000073_00600_v'
# video_name = './test_video/video/uav0000073_00600_v.mp4'
#
# # 获取图像文件列表并按文件名排序
# images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
# images.sort()
#
# # 获取第一张图像的宽度和高度作为输出视频的宽度和高度
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, _ = frame.shape
#
# # 定义输出视频编解码器和FPS
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 30.0
#
# # 创建输出视频对象
# out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
#
# # 遍历每张图像，将其加入输出视频中
# for image in images:
#     frame = cv2.imread(os.path.join(image_folder, image))
#     out.write(frame)
#
# # 释放输出视频对象和所有窗口
# out.release()
import os
import cv2

VIDEO_EXTENSION = '.mp4'

# 遍历指定路径下的所有文件夹
for foldername in os.listdir('./test_video/video/'):
    folderpath = os.path.join('./test_video/video/', foldername)
    if not os.path.isdir(folderpath):
        continue

    # 获取当前文件夹下所有图片的路径
    imgpaths = [os.path.join(folderpath, filename) for filename in os.listdir(folderpath) if filename.endswith('.jpg')]

    # 按文件名排序
    imgpaths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # 设置输出视频路径为当前文件夹的名字
    output_video_path = os.path.join('./test_video/video/', foldername + VIDEO_EXTENSION)

    # 读取第一张图片，获取图像尺寸
    img = cv2.imread(imgpaths[0])
    height, width, channels = img.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # 逐帧写入视频
    for imgpath in imgpaths:
        img = cv2.imread(imgpath)
        video_writer.write(img)

    # 释放资源
    video_writer.release()


