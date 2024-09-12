import torch
import torch.nn.functional as F


def sharpness(img_gray):

    device = img_gray.device
    F1 = torch.tensor([[1, -1], [0, 0]]).float()

    F1 = F1.unsqueeze(0).unsqueeze(0).to(device)
    F2 = torch.transpose(F1, 2, 3).to(device)
    img_gray = img_gray.unsqueeze(0).unsqueeze(0)

    H1 = F.conv2d(img_gray, F1)
    H2 = F.conv2d(img_gray, F2)

    H1 = H1.squeeze(0).squeeze(0)
    H2 = H2.squeeze(0).squeeze(0)
    g = torch.sqrt(H1 ** 2 + H2 ** 2)

    row, col = g.shape
    B = round(min(row, col) / 16)

    g_center = g[B: -B, B: -B]

    MaxG = torch.max(g_center)
    MinG = torch.min(g_center)
    MeanG = torch.mean(g_center)

    if MeanG == 0:
        re = torch.tensor(0)
    else:
        CVG = (MaxG - MinG) / (MeanG)
        re = MaxG ** 0.61 * CVG ** 0.39
    return re


def video_sharpness(video_data):
    nof = video_data.shape[0]
    feature = torch.zeros(nof, 1)
    for i in range(nof):
        frame_data = video_data[i, :, :]
        feature[i, 0] = sharpness(frame_data)

    return feature

