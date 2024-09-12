import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms
from video_quality.ResNet3D import r2plus1d_18


class EfficientNetV2Multilevel(nn.Module):
    def __init__(self, weights=None):
        super(EfficientNetV2Multilevel, self).__init__()
        model = models.efficientnet_v2_s(weights=None)
        model.load_state_dict(torch.load(weights, map_location='cpu'))
        self.features_extractor = nn.Sequential(*list(model.children())[0][:-1])
        self.features_extractor.eval()
        for p in self.features_extractor.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        # x --> batch size, channel, height, width
        # feature --> batch size, feature channel, mean/std
        features_mean_list = []
        features_std_list = []
        for ii, model in enumerate(self.features_extractor):
            x = model(x)
            if ii >= 2:
                features_mean = torch.mean(x, dim=[2, 3])
                features_std = torch.std(x, dim=[2, 3])
                # print(features_mean.shape)
                features_mean_list.append(features_mean)
                features_std_list.append(features_std)
        features_mean_list = torch.cat(features_mean_list, dim=1)
        return features_mean_list
        # features_std_list = torch.cat(features_std_list, dim=1)
        # features = torch.stack((features_mean_list, features_std_list), dim=2)
        # return features


def get_video_semantic_feature(extractor, video_data, frame_batch_size, device='cuda:0'):
    video_len = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output_feature_list = []
    with torch.no_grad():
        while frame_end < video_len:
            video_batch = video_transform(video_data[frame_start: frame_end]).to(device)
            output_feature = extractor(video_batch.clone())
            output_feature_list.append(output_feature)
            frame_start += frame_batch_size
            frame_end += frame_batch_size
        last_batch = video_transform(video_data[frame_start:video_len]).to(device)
        output_feature = extractor(last_batch)
        output_feature_list.append(output_feature)
    output_feature_list = torch.cat(output_feature_list, dim=0)
    return output_feature_list


class R2plus1d_18Feature(torch.nn.Module):
    """Modified ResNet50 for different level feature extraction"""
    def __init__(self, res3d_path):
        super(R2plus1d_18Feature, self).__init__()
        res3d_model = r2plus1d_18(pretrained=False)
        res3d_model.load_state_dict(torch.load(res3d_path, map_location='cpu'))
        self.features_extractor = res3d_model.eval()

    def forward(self, x):
        # x --> seq_len=16, channel, height, width
        # feature --> 1, feature dim
        x = x.unsqueeze(0)  # 1, seq_len=16, channel, height, width
        x = x.permute([0, 2, 1, 3, 4])
        feature = self.features_extractor(x)
        feature = feature.squeeze(2).squeeze(2).squeeze(2)
        return feature


def generate_motion_features(extractor, video_data, clip_size=16, device='cuda', is_trt:bool=False, transform:bool=False):
    # video_data-->seq_len, channel, Height, Width
    batch_N = video_data.shape[0] // clip_size
    output_feature_list = []
    with torch.no_grad():
        for k in range(batch_N):
            if transform:
                video_batch = torchvision.transforms.functional.resize(
                    img=video_transform(video_data[k*clip_size: (k+1)*clip_size]), 
                    size=224).to(device)
            else:
                video_batch = video_data[k*clip_size: (k+1)*clip_size].to(device)
            if is_trt:
                output_feature = extractor(video_batch.clone().half())
                output_feature_list.append(output_feature.float())
            else:
                output_feature = extractor(video_batch.clone())
                output_feature_list.append(output_feature)
    output_feature_list = torch.cat(output_feature_list, dim=0)
    return output_feature_list


def video_transform(video_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    video_length, video_height, video_width, video_channel = video_data.shape
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    for frame_idx in range(video_length):
        frame = video_data[frame_idx]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    return transformed_video


class MDVQA_cvpr(nn.Module):
    def __init__(self, inplace=True):
        super(MDVQA_cvpr, self).__init__()
        input_size = sum([256, 160, 128, 64])
        self.semantic_temporal = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(inplace=inplace),
        )

        self.semantic_spatial = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=128, out_features=6),
            nn.LeakyReLU(inplace=inplace),
        )

        self.distortion_spatial = nn.Sequential(
            nn.Linear(in_features=6, out_features=5),
            nn.LeakyReLU(inplace=inplace),
        )

        self.distortion_temporal = nn.Sequential(
            nn.Linear(in_features=6, out_features=5),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=5, out_features=4),
            nn.LeakyReLU(inplace=inplace),
        )

        self.motion_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=512, out_features=128),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(inplace=inplace),
        )

        self.spatial_temporal_fc = nn.Sequential(
            nn.Linear(in_features=64*2 + 6 + 5 + 4, out_features=64),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=64, out_features=16),
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=16, out_features=1),
            nn.LeakyReLU(inplace=inplace),
        )

        self.weight_init()

        self.clip_pooling_t = nn.Sequential(
            nn.Conv1d(in_channels=64 + 6 + 5 + 4, out_channels=64 + 6 + 5 + 4, kernel_size=8, stride=8, padding=0),
            nn.LeakyReLU()
        )

    def weight_init(self):
        initializer = kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, semantic_feature, metric_feature, motion_feature):
        semantic_feature = semantic_feature[:, :, 48:]
        semantic_feature_t = torch.abs(semantic_feature[:, 0::2] - semantic_feature[:, 1::2])
        distortion_feature_t = torch.abs(metric_feature[:, 0::2] - metric_feature[:, 1::2])

        semantic_feature_s = semantic_feature[:, 1::2, :]
        distortion_feature_s = metric_feature[:, 1::2, :]

        semantic_feature_t = self.semantic_temporal(semantic_feature_t)
        distortion_feature_t = self.distortion_temporal(distortion_feature_t)

        semantic_feature_s = self.semantic_spatial(semantic_feature_s)
        distortion_feature_s = self.distortion_spatial(distortion_feature_s)

        feature_fuse = torch.cat((semantic_feature_t, distortion_feature_t, semantic_feature_s, distortion_feature_s),
                                 dim=2)
        feature_fuse = feature_fuse.permute([0, 2, 1])
        feature_fuse = self.clip_pooling_t(feature_fuse)
        feature_fuse = feature_fuse.permute([0, 2, 1])

        feature_m = self.motion_fc(motion_feature)

        feature_fuse = torch.cat((feature_fuse, feature_m), dim=2)

        output = self.spatial_temporal_fc(feature_fuse)

        score = torch.mean(output).cpu()
        return score


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            