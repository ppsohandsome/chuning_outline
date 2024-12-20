# 是否保存图片(默认False)可改为True(保存在zoutput文件夹下)
image_save = False
# rtsp流地址，按照d25-d30顺序
urls = [
     #'zinput/@d25.mp4',
     #'zinput/@d26.mp4',
     #'zinput/@d27.mp4',
     #'zinput/@d28.mp4',
     #'zinput/@d29.mp4',
     #'zinput/@d30.mp4',
    # ...
     'rtsp://admin:abcd1234@192.168.1.74:554/h264/ch1/main/av_stream',
     'rtsp://admin:abcd1234@192.168.1.78:554/h264/ch1/main/av_stream',
     'rtsp://admin:abcd1234@192.168.1.96:554/h264/ch1/main/av_stream',
     'rtsp://admin:abcd1234@192.168.1.105:554/h264/ch1/main/av_stream',
     'rtsp://admin:abcd1234@192.168.1.128:554/h264/ch1/main/av_stream',
     'rtsp://admin:abcd1234@192.168.1.140:554/h264/ch1/main/av_stream',
    #'rtsp://admin:abcd1234@192.168.1.74/Streaming/Channels/103?transportmode=unicast',
    #'rtsp://admin:abcd1234@192.168.1.78/Streaming/Channels/103?transportmode=unicast',
    #'rtsp://admin:abcd1234@192.168.1.96/Streaming/Channels/103?transportmode=unicast',
    #'rtsp://admin:abcd1234@192.168.1.105/Streaming/Channels/103?transportmode=unicast',
    #'rtsp://admin:abcd1234@192.168.1.128/Streaming/Channels/103?transportmode=unicast',
    #'rtsp://admin:abcd1234@192.168.1.140/Streaming/Channels/103?transportmode=unicast',

]
# 各股道电子围栏设置  y1<y2
D25_line_y1 = 100
D25_line_y2 = 101

D26_line_y1 = 200
D26_line_y2 = 201

D27_line_y1 = 300
D27_line_y2 = 301

D28_line_y1 = 100
D28_line_y2 = 150

D29_line_y1 = 200
D29_line_y2 = 250

D30_line_y1 = 300
D30_line_y2 = 350
