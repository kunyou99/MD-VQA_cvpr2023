# MD-VQA: Multi-Dimensional Quality Assessment for UGC Live Videos

## Instruction

To addresses the need for effective video quality assessment tools for user-generated content (UGC) live videos, which are often distorted during capture and transcoding, we constructed a UGC live VQA database named Taolive, and developed a Multi-Dimensional VQA (MD-VQA) evaluator to measure visual quality from semantic, distortion, and motion aspects.

## Dataset

This database aims at the quality assessment of UGC live videos. Please refer to our paper (MD-VQA: Multi-Dimensional Quality Assessment for UGC Live Videos, CVPR2023) for more details. The official repo can be accessed [here](https://tianchi.aliyun.com/dataset/148818?t=1679581936815).

## Quick Start

### Prepare environment

```
pip install -r requirements.txt
```

### Get video quality score

We offer two pre-trained model weights: one trained solely on the LSVQ dataset (`LSVQ_rp0.pth`) and another trained on a combined LSVQ and TaoLive dataset (`MDVQA_LSVQ_TaoLive.pth`).  Running the following code, you can evaluate the quality of the local video.

Please be aware that the current model has been trained and tested primarily on videos with resolutions up to the Blu-ray level (1080p). For best results, we recommend using videos at or below this resolution.

```
python3 test_video.py -i "your video path"
```

## Cite

If our work is useful for your research, please use the following BibTeX entry for citation.

```
@inproceedings{zhang2023md,
  title={MD-VQA: Multi-dimensional quality assessment for UGC live videos},
  author={Zhang, Zicheng and Wu, Wei and Sun, Wei and Tu, Danyang and Lu, Wei and Min, Xiongkuo and Chen, Ying and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1746--1755},
  year={2023}
}
```
