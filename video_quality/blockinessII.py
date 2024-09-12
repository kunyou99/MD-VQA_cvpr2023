import torch


def blockinessII(img_gray):
    h, w = img_gray.shape
    img = img_gray.float()
    # feature extraction

    # 1. horizontal feature
    d_h = img_gray[:, 1:w] - img_gray[:, 0:w-1]
    B_h = torch.mean(torch.abs(d_h[:, 7:8*(int(w/8) - 1):8]))
    A_h = (8 * torch.mean(torch.abs(d_h)) - B_h) / 7

    sig_h = torch.sign(d_h)
    left_sig = sig_h[:, 0:w-2]
    right_sig = sig_h[:, 1:w-1]
    Z_h = torch.sum(torch.mul(left_sig, right_sig) < 0) / (left_sig.shape[0] * left_sig.shape[1])

    # 2. vertical feature
    d_v = img_gray[1:h, :] - img_gray[0:h-1, :]
    B_v = torch.mean(torch.abs(d_v[7:8*(int(h/8) - 1):8, :]))
    A_v = (8 * torch.mean(torch.abs(d_v)) - B_v) / 7

    sig_v = torch.sign(d_v)
    up_sig = sig_v[0:h-2, :]
    down_sig = sig_v[1:h-1, :]
    Z_v = torch.sum(torch.mul(up_sig, down_sig) < 0) / (up_sig.shape[0] * up_sig.shape[1])

    # 3. combined features

    B = (B_h + B_v) / 2
    A = (A_h + A_v) / 2
    Z = (Z_h + Z_v) / 2

    # Quality Prediction

    alpha = -245.8909
    beta = 261.9373
    gamma1 = -239.8886
    gamma2 = 160.1664
    gamma3 = 64.2859

    score = alpha + beta * (torch.pow(B, (gamma1 / 10000)) * torch.pow(A, (gamma2 / 10000)) * torch.pow(Z, (gamma3 / 10000)))

    return score


def video_blockinessII(video_data):
    nof = video_data.shape[0]
    feature = torch.zeros(nof, 1)
    for i in range(nof):
        frame_data = video_data[i, :, :]
        feature[i, 0] = blockinessII(frame_data)

    return feature.numpy()





