import torch
import numpy as np
import os
import time
import json

from video_quality.read_video import read_video_low_memory
from video_quality.model import EfficientNetV2Multilevel, get_video_semantic_feature, R2plus1d_18Feature, generate_motion_features, MDVQA_cvpr
from video_quality.cal_quality_metric import  check_blockiness_values_2, cal_quality_metric_3_low_memory
# from mdvqa_utils.common import ErrorCode


class MDVQA:
    def __init__(self, use_cudnn: bool = False,
                 semantic_path='./data/efficientnet_v2_s-dd5fe13b.pth',
                 motion_path='./data/r2plus1d_18-91a641e6.pth',
                 mdvqa_path='./data/LSVQ_rp0.pth'):

        semantic_model = EfficientNetV2Multilevel(weights=semantic_path).to('cuda')
        motion_model = R2plus1d_18Feature(res3d_path=motion_path).to('cuda')

        mdvqa_model = MDVQA_cvpr()
        mdvqa_model.load_state_dict(torch.load(mdvqa_path, map_location='cpu'))
        mdvqa_model.to('cuda')
        mdvqa_model.eval()

        self.model = [semantic_model, motion_model, mdvqa_model]

        if use_cudnn:
            torch.backends.cudnn.benchmark = True  
        else:
            pass

    def __call__(self, video_path:str):
        """
        @description  :     MDVQA实例推理
        ---------
        @param  :
                    video_path:         输入必须为本地视频, 算法内部无下载动作                      
        -------
        @Returns  : MDVQA 结果 json (kv全部为str)
        -------
        """
        rt_all = {}
        _task_rt_start = time.monotonic()

        if not os.path.exists(video_path):
            raise ValueError(f'video file not exists, MD-VQA need local file, {video_path}')
        
        _read_video_rt_start = time.monotonic()
        print('start read video...')
        try:
            color_video, gray_video, video_time, resolution_src = read_video_low_memory(video_path)
        except Exception as e:
            print('read video error, {}'.format(e))
            raise ValueError('read video error, {}'.format(e))
        
        print('color_video: {}x{}, video_time: {}'.format(
            len(color_video), color_video[0].shape, video_time))
        rt_all['read_video_rt'] = str(round(time.monotonic() - _read_video_rt_start, 2))
        print('time of read video: {}'.format(rt_all['read_video_rt']))

        """
        传统特征提取
        """
        _trad_rt_start = time.monotonic()
        print('start extract tradition features...')
        tradition_batch_size = 160
        metric_feature = None
        for idx in range(0, len(color_video), tradition_batch_size):
            _metric_feature = cal_quality_metric_3_low_memory(color_video=color_video[idx:min(idx+tradition_batch_size, len(color_video))])
            if not isinstance(_metric_feature, torch.Tensor):
                metric_feature = _metric_feature
            if metric_feature is None:
                metric_feature = _metric_feature
            else:
                metric_feature = torch.vstack((metric_feature, _metric_feature))
        metric_feature = check_blockiness_values_2(metric_feature).to('cuda')  # N x 5
        rt_all['trad_rt'] = str(round(time.monotonic() - _trad_rt_start, 2))
        print('time of tradition feature generation: {}'.format(rt_all['trad_rt']))

        try:
            """
            语义特征提取
            """
            _semantic_start = time.monotonic()
            print('start extract semantic features...')
            semantic_feature = get_video_semantic_feature(extractor=self.model[0], video_data=color_video, frame_batch_size=2)
            rt_all['semantic_rt'] = str(round((time.monotonic() - _semantic_start), 2))
            print("time of semantic feature generation: {}".format(rt_all['semantic_rt']))
            torch.cuda.empty_cache()
            
            """
            运动特征提取
            """
            _motion_start = time.monotonic()
            print('start extract motion features...')
            motion_feature = None
            
            motion_feature = generate_motion_features(extractor=self.model[1], video_data=color_video, transform=True)
            torch.cuda.empty_cache()
            rt_all['motion_rt'] = str(round(time.monotonic() - _motion_start, 2))
            print('time of motion feature generation: {}'.format(rt_all['motion_rt']))

            """
            特征回归
            """
            # MDVQA inference
            _fusion_start = time.monotonic()
            with torch.no_grad():
                metric_feature = metric_feature.unsqueeze(0)
                semantic_feature = semantic_feature.unsqueeze(0)
                motion_feature = motion_feature.unsqueeze(0)

                outputs = self.model[2](semantic_feature, metric_feature, motion_feature)
                score_ori = outputs.item()
            torch.cuda.empty_cache()
            rt_all['fusion_rt'] = str(round(time.monotonic() - _fusion_start, 2))
                
        except Exception as e:
            raise RuntimeError(f'MDVQA_INFERENCE_FAILED {e}')

        score = score_ori * 0.8313 + 21.0112
        score = round(max(0, min(99.99, score)), 2)
        
        metric_feature = metric_feature.cpu().numpy()
        shakiness, over_index, under_index = np.std(metric_feature[0, :, 0]), np.mean(metric_feature[0, :, 2]), \
                np.mean(metric_feature[0, :, 3])
        sharpness, noise, blockiness, colorfullness = np.mean(metric_feature[0, :, 0]), 0, \
                np.mean(metric_feature[0, :, 1]), np.mean(metric_feature[0, :, 4])
        
        rt_all['task_rt'] = str(round(time.monotonic() - _task_rt_start, 2))
        outputs = {
            'score': str(score), 'video': str(video_path),
            'shakiness': str(shakiness), 'over_index': str(over_index), 'under_index': str(under_index), 'sharpness': str(sharpness), 'noise': str(noise), 
            'blockiness': str(blockiness), 'colorfullness': str(colorfullness), 'version': str('mdvqa_cvpr'),
            }
        outputs.update(rt_all)
        print('outputs: {}'.format(json.dumps(outputs)))

        return outputs
