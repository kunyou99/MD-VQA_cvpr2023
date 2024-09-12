import torch


def img2patch(img, pch_size, stride=1):

    pch_H = pch_W = pch_size
    stride_H = stride_W = stride

    C, H, W = img.shape
    num_H = len(range(0, H - pch_H + 1, stride_H))
    num_W = len(range(0, W - pch_W + 1, stride_W))
    num_pch = num_H * num_W

    pch = torch.zeros((C, pch_H * pch_W, num_pch))
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = img[:, ii:H - pch_H + ii + 1:stride_H, jj:W - pch_W + jj + 1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def noise(img, pch_size=8):
    # image to patch
    pch = img2patch(img, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(dim=1, keepdim=True)
    X = pch - mu
    sigma_X = torch.matmul(X, torch.transpose(X, 0, 1)) / num_pch
    try:
    # sig_value, _ = torch.symeig(sigma_X, eigenvectors=False)
        sig_value, _ = torch.linalg.eigh(sigma_X, UPLO='U')
    except RuntimeError:
        return torch.tensor(0.0)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = torch.mean(sig_value[:ii])
        if torch.sum(sig_value[:ii] > tau) == torch.sum(sig_value[:ii] < tau):
            if tau < 0 or torch.isnan(tau):
                return torch.tensor(0)
            else:
                return torch.sqrt(tau)


def video_noise(video_data):
    nof = video_data.shape[0]
    feature = torch.zeros(nof, 1)
    for i in range(nof):
        frame_data = video_data[i, :, :, :]
        feature[i, 0] = noise(frame_data)

    return feature
