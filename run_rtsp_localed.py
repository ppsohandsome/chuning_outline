import json
import cv2
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaProducer
import config
import detect_train_line_electronicfence
from datetime import datetime
import time
import threading
import ctypes
import traceback

# Kafka配置
producer = KafkaProducer(bootstrap_servers="192.168.1.13", max_request_size=1048576000)
topic_send = "train_located"

# 视频流配置
streams = [
    {"stream_name": "Stream 0", "rtsp_url": config.urls[0]},
    {"stream_name": "Stream 1", "rtsp_url": config.urls[1]},
    {"stream_name": "Stream 2", "rtsp_url": config.urls[2]},
    {"stream_name": "Stream 3", "rtsp_url": config.urls[3]},
    {"stream_name": "Stream 4", "rtsp_url": config.urls[4]},
    {"stream_name": "Stream 5", "rtsp_url": config.urls[5]},
    # Add more streams...
]

# 创建一个字典来存储每个流的线程对象
stream_threads = {}

# 处理帧的函数
# 处理帧的函数
def process_frame(rtsp_url, stream_name):
    max_retry_attempts = 5  # 设置最大重试次数
    retry_count = 0  # 用于追踪尝试次数
    cap = cv2.VideoCapture(rtsp_url)
    while retry_count < max_retry_attempts:  # 限制重试次数
        try:
            # 设置帧率为10帧/秒
            processing = False
            target_frame_rate = 5  # 例如，目标帧速率为每秒处理30帧
            while True:
                # 在处理帧之前等待以满足目标帧速率
                frame_interval = 1.0 / target_frame_rate
                time.sleep(frame_interval)
                # 只有在不在处理帧的情况下才获取下一帧
                if not processing:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"ret,RTSP 流 {stream_name} 断开连接。正在重新连接...")
                        # 关闭之前的连接
                        cap.release()
                        # 重新连接
                        try:
                            cap = cv2.VideoCapture(rtsp_url)  # 创建新的 cv2.VideoCapture 对象:wq
                            restart_thread(rtsp_url, stream_name)
                        except Exception as e:
                            print(f"重新启动线程时发生异常: {str(e)}")
                        break
                # 开始处理帧
                processing = True
                # 调用 detect_train_line_electronicfence 模块的函数
                x = detect_train_line_electronicfence.main(frame, stream_name)

                if x is None or x[0] not in [True, False]:
                    print(stream_name + " 的 x 为 None 或者不是 True/False。正在重启线程...")
                    # 关闭之前的连接
                    cap.release()
                    # 置 processing 为 False 以触发重新获取下一帧
                    processing = False
                    # 终止旧线程并重新启动新线程
                    restart_thread(rtsp_url, stream_name)
                    break

                print(stream_name, x[0], x[1], x[2], x[3], x[4])

                if not x[0]:
                    continue
                format = "%Y-%m-%d|%H:%M:%S"
                message = {"camera-id": x[1],
                           "train-located": x[2],
                           "trainnum": x[3],
                           "timestemp": datetime.now().strftime(format),
                           "base64img": x[5]}

                producer.send(topic_send, value=json.dumps(message).encode('utf-8'))
                producer.flush()
                processing = False

                # 添加时间间隔（例如，休眠1秒）
                time.sleep(1)  # 1秒钟的时间间隔

        except cv2.error as e:
            print(f"OpenCV error: {str(e)}")
            traceback.print_exc()  # 打印异常堆栈跟踪信息
            # 在这里执行资源释放和清理操作
            cap.release()
            retry_count += 1  # 增加尝试次数
            if retry_count < max_retry_attempts:
                print(f"尝试重新连接... (尝试次数: {retry_count})")
            else:
                print(f"达到最大尝试次数，无法继续。")
                break  # 达到最大尝试次数，退出循环
        except Exception as e:
            print(f"其他异常: {str(e)}")
            traceback.print_exc()  # 打印异常堆栈跟踪信息
            # 在这里执行资源释放和清理操作
            cap.release()
            retry_count += 1  # 增加尝试次数
            if retry_count < max_retry_attempts:
                print(f"尝试重新连接... (尝试次数: {retry_count})")
            else:
                print(f"达到最大尝试次数，无法继续。")
                break  # 达到最大尝试次数，退出循环

def terminate_thread(thread):
    """Terminate a thread."""
    if not thread.is_alive():
        return
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def restart_thread(rtsp_url, stream_name):
    print(f"正在重新启动线程：{stream_name}")
    # 终止旧线程
    if stream_name in stream_threads:
        old_thread = stream_threads[stream_name]
        terminate_thread(old_thread)  # 终止旧线程
        del stream_threads[stream_name]  # 从字典中移除旧线程对象

    # 创建一个新线程来处理视频流
    thread = threading.Thread(target=process_frame, args=(rtsp_url, stream_name))
    thread.daemon = True  # 将线程设置为守护线程
    thread.start()
    stream_threads[stream_name] = thread
    print(f"已重新启动线程：{stream_name}")

# 创建线程池并启动任务
with ThreadPoolExecutor(max_workers=len(streams)) as executor:
    for stream in streams:
        print(f"正在启动线程：{stream['stream_name']}")
        executor.submit(process_frame, stream["rtsp_url"], stream["stream_name"])

