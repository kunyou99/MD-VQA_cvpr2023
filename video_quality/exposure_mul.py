import numpy as np
import cv2


def calcHist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    h, w = image.shape
    hist = hist/(h*w)
    return hist.flatten()


def sobel_sharpness(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 1，0参数表示在x方向求一阶导数
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 0,1参数表示在y方向求一阶导数

    gm = cv2.sqrt(sobelx * sobelx + sobely * sobely)

    return np.mean(gm), gm

# V of HSV color space, exp( -.5 * (seq(:,:,n) - .5).^2 / .2^2 );
def exposure_wellness(img, miu, denom=0.001):
    exposure_wellness_map = np.exp(-(img-miu)*(img-miu)/denom)
    exposure_level = np.mean(exposure_wellness_map)
    return exposure_level, exposure_wellness_map


def exposure_bright_dark(img_hue, img_saturation, img_brightness, brightness_max_min_v, bright_thresh, abs_delta, is_bright, denom=0.01):
    exposure_level_brightness, exposure_level_saturation, exposure_hue = 0, 1, 0
    exposure_wellness_map, exposure_sat_map = None, None
    delta = 1-bright_thresh if is_bright else -bright_thresh
    if (is_bright and (brightness_max_min_v > bright_thresh)) or ((not is_bright) and (brightness_max_min_v < bright_thresh)):
        exposure_level_brightness, exposure_wellness_map = exposure_wellness(img_brightness, bright_thresh+delta, denom)

    if is_bright:
        exposure_level_saturation, exposure_sat_map = exposure_wellness(img_saturation, 0, 0.01)
        exposure_hue, exposure_hue_map = exposure_wellness(img_hue, 0.002, 0.00001)
        exposure_hue *= np.exp(-10*np.std(exposure_hue_map))

    return exposure_level_brightness, exposure_hue, exposure_level_saturation, exposure_wellness_map, exposure_sat_map


def find_texture_center(sharpness_map):
    h, w = sharpness_map.shape

    total_sharpness = np.sum(sharpness_map)

    center_sharpness = 0
    stride = 4
    top, bottom, left, right = 0, 0, 0, 0
    for i in range(0, h - stride, stride):
        center = sharpness_map[i:, :]
        center_avg_sharpness = np.sum(center)
        if center_avg_sharpness < 0.995 * total_sharpness:
            top = i - stride
            center_sharpness = np.sum(sharpness_map[top:, :])
            break

    for j in range(0, h - stride, stride):
        center = sharpness_map[top:h - j + 1, :]
        center_avg_sharpness = np.sum(center)
        if center_avg_sharpness < 0.995 * center_sharpness:
            bottom = j - stride
            center_sharpness = np.sum(sharpness_map[top:(h-bottom+1), :])
            break

    for i in range(0, w - stride, stride):
        center = sharpness_map[top:(h-bottom+1), i:]
        center_avg_sharpness = np.sum(center)
        if center_avg_sharpness < 0.995 * center_sharpness:
            left = i - stride
            center_sharpness = np.sum(sharpness_map[top:(h-bottom+1), left:])
            break

    for j in range(0, w - stride, stride):
        center = sharpness_map[top:(h - bottom + 1), left:(w-j+1)]
        center_avg_sharpness = np.sum(center)
        if center_avg_sharpness < 0.995 * center_sharpness:
            right = j - stride
            center_sharpness = np.sum(sharpness_map[top:(h - bottom + 1), left:(w-right+1)])
            break

    return top, bottom, left, right, sharpness_map[top:(h-bottom+1), left:(w-right+1)]

def exposure_wellness_BGR_img(input_img, bright_thresh=0.95, dark_thresh=0.1):
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    h, w, c = img.shape

    img_hue, img_saturation, img_brightness = img[:, :, 0] / 255., img[:, :, 1] / 255., img[:, :, 2] / 255.
    sharpness, sharpness_map = sobel_sharpness(img_brightness)
    brightness_max_v, brightness_min_v = np.max(img_brightness), np.min(img_brightness)

    if np.sum(sharpness_map) < 0.01:
        return 0, 0

    if brightness_max_v < dark_thresh:
        return 0, 1
    if brightness_min_v > bright_thresh:
        return 1, 0

    top, bottom, left, right, sharpness_map = find_texture_center(sharpness_map)
    img = img[top:(h - bottom + 1), left:(w - right + 1), :]

    img_hue, img_saturation, img_brightness = img[:, :, 0] / 255., img[:, :, 1] / 255., img[:, :, 2] / 255.
    bright_mean, brightness_std = np.mean(img_brightness), np.std(img_brightness)

    hist_sat = calcHist(img[:, :, 1])

    over_exposure_brightness, exposure_hue, exposure_level_saturation, over_exposure_map, over_sat_map = exposure_bright_dark(img_hue, img_saturation, img_brightness, brightness_max_v, bright_thresh, 0.035, True, denom=0.01)
    under_exposure_brightness, _, _, under_exposure_map, under_sat_map = exposure_bright_dark(img_hue, img_saturation, img_brightness, brightness_min_v, dark_thresh, 0.025, False, denom=0.01)

    hue_weight, blank_area_weight, blank_v_mean_over, blank_v_mean_under = 1, 1, 1, 1
    if bright_mean < 0.3:
        hue_weight = np.exp(-5 * (0.3 - bright_mean))

    ret_under_exposure = under_exposure_brightness*np.exp(-5*sharpness)

    sat_hist_sim = 0.5 * np.sum(hist_sat[:1]) + 0.35 * np.sum(hist_sat[:5]) + 0.15 * np.sum(hist_sat[:25])
    sat_hist_sim = 1 - np.exp(-5 * sat_hist_sim)

    sharpness_std_mean_ratio = np.std(sharpness_map)/sharpness
    sharpness_std_weight = 1 if sharpness_std_mean_ratio < 2 else (0.25 if sharpness_std_mean_ratio > 4 else (1.75-0.375*sharpness_std_mean_ratio))

    ret_over_exposure = exposure_hue*hue_weight*exposure_level_saturation + over_exposure_brightness*(1-np.exp(-2.5*(exposure_level_saturation+sat_hist_sim)))*np.exp(-0.5*sharpness)*sharpness_std_weight

    return np.clip(ret_over_exposure, 0, 1), np.clip(ret_under_exposure, 0, 1)


def exposure_ds_mul_new_single_thread(color_video, frame_idx_sel, threads=4):
    frames = [color_video[idx] for idx in frame_idx_sel]

    expsoure_0 = []
    expsoure_1 = []
    for frame_k in frames:
        height, width, c = frame_k.shape
        height = height // 2
        width = width // 2
        frame_temp = cv2.resize(frame_k, (width, height))
        expsoure_k_0, expsoure_k_1 = exposure_wellness_BGR_img(frame_temp)
        expsoure_0.append(expsoure_k_0)
        expsoure_1.append(expsoure_k_1)

    return np.array(expsoure_0), np.array(expsoure_1)
