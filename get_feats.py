from cv2 import DualTVL1OpticalFlow_create as DualTVL1
import sys
import cv2
import numpy as np
import imageio

def resize(im):
    r, c, _ = im.shape
    h = r if r <= c else c
    ratio = 256 / float(h)
    return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)

def perform_ofa(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
    previous_frame = cv2.cvtColor(v[0], cv2.COLOR_BGR2GRAY)
    flows = np.zeros((f - 1, r, c, 2))
    for i in range(1, f):
        current_frame = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        xflow = flow[..., 0]
        yflow = flow[..., 1]
        flows[i - 1, :, :, 0] = xflow
        flows[i - 1, :, :, 1] = yflow

        previous_frame = current_frame

    return np.array(flows)

def perform_ofb(v):
    v = v.astype('uint8')
    f, r, c, d = v.shape
    previous_frame = cv2.cvtColor(v[0], cv2.COLOR_BGR2GRAY)
    flows = np.zeros((f - 1, r, c, 2))
    tvl1 = DualTVL1()
    for i in range(1, f):
        current_frame = cv2.cvtColor(v[i], cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(previous_frame, current_frame, None)

        xflow = flow[..., 0]
        yflow = flow[..., 1]
        flows[i - 1, :, :, 0] = xflow
        flows[i - 1, :, :, 1] = yflow

        previous_frame = current_frame

    return np.array(flows)

def vid2npy(f):
    vid = imageio.get_reader(f, 'ffmpeg')
    x = []
    for frame in vid.iter_data():
        x.append(frame)
    return np.array(x)

fp = sys.argv[1]
print('loading video')
video = vid2npy(fp).astype('float64')
print('performing flow')
flow = np.load('fullflow.npy')

#RGB POSTPROCESSING
video = np.array([resize(e) for e in video])
X_std = (video - video.min()) / (video.max() - video.min())
video = X_std * (1.0 - -1.0) + -1.0
_, r, c, _ = video.shape
cy = r / 2.0
cx = c / 2.0
x = int(cx-112)
y = int(cy-112)
centre_crop_rgb = video[:, y:y+224, x:x+224, :]
print(centre_crop_rgb.shape)


flow = np.array([resize(e) for e in flow])
flow = np.clip(flow, -20, 20)
X_std = (flow - flow.min()) / (flow.max() - flow.min())
flow = X_std * (1.0 - -1.0) + -1.0
_, r, c, _ = flow.shape
cy = r / 2.0
cx = c / 2.0
x = int(cx-112)
y = int(cy-112)
centre_crop_flow = flow[:, y:y+224, x:x+224, :]
print(centre_crop_flow.shape)

centre_crop_rgb = np.expand_dims(centre_crop_rgb, axis=0)
centre_crop_flow = np.expand_dims(centre_crop_flow, axis=0)

np.save('rgb.npy', centre_crop_rgb[:, :79, :, :, :])
np.save('flow.npy', centre_crop_flow[:, :79, :, :, :])
