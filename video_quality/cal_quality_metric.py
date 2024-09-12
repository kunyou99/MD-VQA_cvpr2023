from video_quality.sharpness import sharpness
from video_quality.colorfulness import colorfulness
from video_quality.blockinessII import blockinessII
from video_quality.noise import noise
from video_quality.exposure_mul import exposure_ds_mul_new_single_thread
import numpy as np
import torch
import cv2


## note: channel order of color_video: RGB
def cal_quality_metric_3_low_memory(color_video):

    quality_metric = []
    width, height, c = color_video[0].shape
    for i in range(len(color_video)):
        _color_video_frame = np.swapaxes(np.swapaxes(color_video[i], 0, 2), 1, 2)
        _color_video_NCHW_torch = torch.from_numpy(_color_video_frame).float()
        _gray_video_torch = torch.from_numpy(cv2.cvtColor(color_video[i], cv2.COLOR_RGB2GRAY)).float()
        
        sharpness_ = sharpness(img_gray=_gray_video_torch)
        blockiness_ = blockinessII(img_gray=_gray_video_torch)
        over_index, under_index = 0, 0
        colorfulness_ = colorfulness(img=_color_video_NCHW_torch)
        noise_ = noise(img=_color_video_NCHW_torch)
        
        res = [sharpness_, noise_, blockiness_, over_index, under_index, colorfulness_]
        quality_metric.append(res)

    frame_idx_sel = [k for k in range(len(color_video))]
    over_indexs, under_indexs = exposure_ds_mul_new_single_thread(color_video, frame_idx_sel)

    five_feature = torch.as_tensor(quality_metric, dtype=torch.float32)
    five_feature[:, 3] = torch.from_numpy(over_indexs)
    five_feature[:, 4] = torch.from_numpy(under_indexs)

    return five_feature


def check_blockiness_values_2(five_feature):
    for idx in range(five_feature.shape[0]):
        if torch.isnan(five_feature[idx, 1]) or torch.isinf(five_feature[idx, 1]):
            l_cnt, r_cnt = 0, 0
            t_sum = 0.0
            t_idx = idx
            while t_idx >= 0 and l_cnt < 2:
                if not (torch.isnan(five_feature[t_idx, 1]) or torch.isinf(five_feature[t_idx, 1])):
                    t_sum += five_feature[t_idx, 1]
                    l_cnt += 1
                t_idx -= 1
            t_idx = idx
            while t_idx < five_feature.shape[0] and r_cnt < 2:
                if not (torch.isnan(five_feature[t_idx, 1]) or torch.isinf(five_feature[t_idx, 1])):
                    t_sum += five_feature[t_idx, 1]
                    r_cnt += 1
                t_idx += 1
            try:
                five_feature[idx, 1] = t_sum / (l_cnt + r_cnt)
            except ZeroDivisionError:
                five_feature[idx, 1] = -245.8909

    return five_feature
    