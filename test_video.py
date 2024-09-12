import os
from video_quality.video_quality import MDVQA as MDVQA
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='subjective result analysis')
    parser.add_argument('-i',"--mp4_path", type=str,default='LSVQ_test_video.mp4', help="input video path")
    args = parser.parse_args()


    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_infer = MDVQA(use_cudnn=True,
                    semantic_path=os.path.join(current_dir,
                                                'data/efficientnet_v2_s-dd5fe13b.pth'),
                    motion_path=os.path.join(current_dir, 'data/r2plus1d_18-91a641e6.pth'),
                    mdvqa_path=os.path.join(current_dir, 'data/LSVQ_rp0.pth'), )
    res = model_infer(video_path=args.mp4_path)
    print(res)#94.45
