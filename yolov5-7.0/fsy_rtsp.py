import ffmpeg
import logging
import subprocess
import sys
import time
import socket
import threading

# 设置日志配置
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

def start_mediamtx():
    """
    启动 MediaMTX RTSP 服务器
    """
    try:
        # 启动 RTSP 服务器的命令
        mediamtx_command = ['D:/install/ffmpeg/mediamtx']
        #mediamtx_command = ['D:/install/ffmpeg/mediamtx.exe']#, '--config', 'E:/yolov5-7.0/yolov5-7.0/mediamtx.yml','--rtspAddress', '0.0.0.0:8554']
        #mediamtx_command = ['mediamtx', '--config', 'mediamtx.yaml', '--rtsp-address', '0.0.0.0:8554']

        logging.info("启动 MediaMTX RTSP 服务器...")
        mediamtx_process = subprocess.Popen(mediamtx_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 等待 RTSP 服务器启动
        time.sleep(15)
        logging.info("MediaMTX RTSP 服务器已启动...")
        return mediamtx_process
    except Exception as e:
        logging.error(f"启动 MediaMTX 服务器时出错：{e}")
        return None


def read_stream_output(stream, log_func):
    """异步读取流的输出并记录到日志"""
    while True:
        output = stream.readline()
        if not output:
            break
        log_func(output.decode().strip())

def stream_mp4_to_rtsp(mp4_file, rtsp_url):
    """
    将 MP4 文件转换为 RTSP 流
    :param mp4_file: MP4 文件路径
    :param rtsp_url: RTSP 流的 URL
    """
    try:
        command = (
            ffmpeg
            .input(mp4_file,stream_loop=-1)
            .output(rtsp_url, format='rtsp', vcodec='libx264', acodec='aac', preset='ultrafast', tune='zerolatency', g=30)
            .global_args('-re')  # 以实时速度流式传输
            .global_args('-loglevel', 'debug')  # 添加日志级别
            .global_args('-max_interleave_delta', '0')  # 尝试减少 NAL 大小
        )
        logging.info(f"FFmpeg 推流命令: {command.get_args()}")
        process = command.run_async(pipe_stdout=True, pipe_stderr=True)

        # 创建并启动线程来读取标准输出和错误输出
        stdout_thread = threading.Thread(target=read_stream_output, args=(process.stdout, logging.debug))
        stderr_thread = threading.Thread(target=read_stream_output, args=(process.stderr, logging.error))

        stdout_thread.start()
        stderr_thread.start()

        # 等待进程完成
        process.wait()

        # 等待线程结束
        stdout_thread.join()
        stderr_thread.join()

        return process
    except FileNotFoundError:
        logging.error("FFmpeg 未找到，请确保已正确安装并添加到系统路径中。")
        return None
    except Exception as e:
        logging.error(f"转换过程中出现错误：{e}")
        return None

def check_port_in_use(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((ip, port))
    if result == 0:
        print(f"Port {port} is open and in use.")
    else:
        print(f"Port {port} is not in use.")
    sock.close()


if __name__ == "__main__":
    check_port_in_use('10.10.11.16', 8556)
    # 启动 MediaMTX RTSP 服务器
    mediamtx_process = start_mediamtx()
    if not mediamtx_process:
        logging.error("无法启动 MediaMTX 服务器")
        sys.exit(1)
    # 等待一段时间以确保 MediaMTX 启动
    time.sleep(5)  # 可以调整等待时间
    check_port_in_use('10.10.11.16', 8556)
    mp4_file_path = 'E:/test1.mp4'  # 替换为你的MP4文件路径
    rtsp_stream_url = 'rtsp://10.10.11.16:8556/live.sdp'  # RTSP 流的 URL
    # 启动 RTSP 流
    process = stream_mp4_to_rtsp(mp4_file_path, rtsp_stream_url)

    if process is not None:
        logging.info("RTSP 流已启动...")
        try:
            process.wait()  # 等待 FFmpeg 推流完成
        except KeyboardInterrupt:
            logging.info("流推送已手动中止")
    else:
        logging.error("无法启动 RTSP 流。")


    # 关闭 MediaMTX 服务器
    if mediamtx_process:
        mediamtx_process.terminate()
        logging.info("MediaMTX 服务器已停止。")

