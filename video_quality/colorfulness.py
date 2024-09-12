import torch


def colorfulness(img):
    """
    Calculate the colorfulness for an input image.

    Parameters
    ----------
    img : array_like (3*Height*Width)
        input color image(RGB)

    returns
    ----------
    re : Numerical
        The colorfulness of the input image

    """

    R = img[0, ]
    G = img[1, ]
    B = img[2, ]

    apha = R - G
    beta = (R + G) * 0.5 - B

    mean_apha = torch.mean(apha)
    mean_beta = torch.mean(beta)

    sigma_a = torch.std(apha)
    sigma_b = torch.std(beta)

    if mean_apha == 0 or mean_beta == 0 or sigma_a == 0 or sigma_b == 0:
        re = torch.tensor(0.0)
    else:
        re = 0.02*torch.log(sigma_a**2/(torch.abs(mean_apha))**0.2)*torch.log(sigma_b**2/(torch.abs(mean_beta))**0.2)

    return re


def video_colorfulness(video_data):
    nof = video_data.shape[0]
    feature = torch.zeros(nof, 1)
    for i in range(nof):
        frame_data = video_data[i, ]
        feature[i, 0] = colorfulness(frame_data)

    return feature
