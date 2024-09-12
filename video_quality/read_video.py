import cv2
import numpy as np
import os 
import subprocess as sp
import json


def get_video_time(filename):
    command = 'ffprobe -loglevel quiet -print_format json -show_format -show_streams -select_streams v {filename}'.format(\
            filename=filename)
    result = sp.Popen(command,shell=True,stdout=sp.PIPE, stderr=sp.STDOUT)#, env=FFPROBE_ENV)
    out = result.stdout.read()
    temp = str(out.decode('utf-8'))
    try:
        duration = json.loads(temp)['streams'][0]['duration']
        duration = eval(duration)
    except:
        duration = 0
    return duration


def transcode_video(src_file:str) -> str:
    """
    @description  :
        部分 ts 获取帧率失败,导致time_limit参数失效
        本函数完成 ts 无损转 mp4, 之后能够正常读取视频帧率
    ---------
    @param  :
        src_file(str): video local path
    -------
    @Returns  :
        out_file(str): video local path with correct fps
    -------
    """
    out_file = '{}_fix_fps.mp4'.format(os.path.basename(src_file))
    sp.run(['ffmpeg', '-y', '-v', 'quiet', '-i', src_file, '-vcodec', 'copy', out_file])#, env=FFMPEG_ENV)
    return out_file


def read_video_low_memory(video_file, time_limit=-1, patch_info=[-1, -1, -1, -1]):
    """
    @description  :
                read video data (low memory)
    ---------
    @param  :
        video_file(str):  video local path (must be)
        time_limit(int):  read video time
        patch_info(List(int)):  patch info, clip video
        short_side(int):    是否按照短边对视频进行resize
    -------
    @Returns  :
        color_video/gray_video/read_video_time/resolution_src
    -------
    """
    sample_frames, read_video_time, resolution_src = read_video_all_frames(video_file=video_file, time_limit=time_limit, patch_info=patch_info)
    color_video = np.stack([cv2.cvtColor(k, cv2.COLOR_BGR2RGB) for k in sample_frames], axis=0)
    # gray_video = np.stack([cv2.cvtColor(k, cv2.COLOR_BGR2GRAY) for k in sample_frames], axis=0)
    gray_video = None
    
    return color_video, gray_video, read_video_time, resolution_src


def read_video_all_frames(video_file, time_limit=-1, patch_info=[-1, -1, -1, -1]):
    delete_flag = False
    if patch_info == None:
        t, l, b, r = -1, -1, -1, -1
    else:
        t, l, b, r = patch_info[0], patch_info[1], patch_info[2], patch_info[3]
    try:
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)
        frame_num = cap.get(7)
        if frame_rate > 100:
            # 帧数异常
            cap.release()
            raise ValueError('wrong framerate')
    except ValueError:
        video_file = transcode_video(src_file=video_file)
        delete_flag = True
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5)
        frame_num = cap.get(7)
    if os.path.splitext(video_file)[-1] == '.ts' and get_video_time(video_file) != 0:
        video_time = get_video_time(video_file)
    else:
        video_time = frame_num / frame_rate
    if video_time < time_limit or time_limit == -1:
        read_video_time = video_time
    else:
        read_video_time = time_limit
    frame_limit = round(read_video_time * frame_rate)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resolution_src = f'{frame_height}x{frame_width}'
    
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if t != -1:
                frame = frame[l:r,t:b,:]
            all_frames.append(frame)
        else:
            break
        if len(all_frames) == frame_limit:
            break

    cap.release()
    if delete_flag and os.path.exists(video_file):
        os.remove(video_file)
    
    # 开始抽帧
    sample_num = int(int(read_video_time) * 16)
    frame_idx_sel = np.linspace(start=0, stop=len(all_frames)-1, num=sample_num, dtype=np.int16)
    sample_frames = [all_frames[k] for k in frame_idx_sel]

    return sample_frames, read_video_time, resolution_src
