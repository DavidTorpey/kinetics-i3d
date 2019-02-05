from multiprocessing import Pool

from compute_i3d import I3DInferrer as I3D
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
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


def get_cropped_videos(video):
    _, r, c, _ = video.shape

    X = []
    while len(X) < 5:
        x = np.random.randint(0, c)
        y = np.random.randint(0, r)
        if y + 224 <= r and x + 224 <= c:
            snippet = video[:, y:y + 224, x:x + 224, :]
            X.append(snippet)
    return np.array(X)


def rescale(vol):
    X_std = (vol - vol.min()) / (vol.max() - vol.min())
    return X_std * (1.0 - -1.0) + -1.0


def vid2npy(f):
    vid = imageio.get_reader(f, 'ffmpeg')
    x = []
    for frame in vid.iter_data():
        x.append(frame)
    return np.array(x)


def cropped_to_snippets(cropped_vols, p):
    X = []
    for cropped_vol in cropped_vols:
        snippets = sample_snippets(cropped_vol, p)
        for snippet in snippets:
            X.append(snippet)
    return np.array(X)


def sample_snippets(video, p):
    WINDOW_LENGTH = 79
    OVERLAP_PERCENTAGE = 0.1
    step_size = int(np.ceil(WINDOW_LENGTH * (1 - OVERLAP_PERCENTAGE)))
    snippets = []
    for i in range(0, len(video) - WINDOW_LENGTH, step_size):
        snippet = video[i:i + WINDOW_LENGTH]
        snippets.append(snippet)
    snippets = np.array(snippets)
    idx = np.random.permutation(len(snippets))
    num_to_sample = int(np.ceil(p * len(snippets)))
    return snippets[idx[:num_to_sample]]

def compute(f):
    print(f)

    with open('temp.txt', 'a') as fff:
        fff.write(f + '\n')

    video = vid2npy(f).astype('float64')
    flow = perform_ofa(video)
    return [video, flow]

def get_rgb_and_flow(video, flow):
    p = 1.0

    video = np.array([resize(e) for e in video])
    video = rescale(video)
    cropped_rgb_videos = get_cropped_videos(video)
    RGB = cropped_to_snippets(cropped_rgb_videos, p)

    flow = np.array([resize(e) for e in flow])
    flow = np.clip(flow, -20, 20)
    flow = rescale(flow)
    cropped_flow_videos = get_cropped_videos(flow)
    FLOW = cropped_to_snippets(cropped_flow_videos, p)

    return RGB, FLOW


i3d = I3D('/home/wits-user/david/data/ucf101/ucf101_I3D/rgb/train1/model.ckpt-5000', '/home/wits-user/david/data/ucf101/ucf101_I3D/flow/train1/model.ckpt-5000')

classes = open('classes.txt').read().splitlines()
train_file = open('/home/wits-user/david/data/ucf101/ucfTrainTestlist/trainlist01.txt').read().splitlines()
paths = []
y = []
for l in train_file:
    fn = l.split(' ')[0]
    classidx = int(l.split(' ')[1])
    target = classes[classidx - 1]
    path = '/home/wits-user/david/data/ucf101/UCF-101/{}'.format(fn)
    paths.append(path)
    y.append(target)


pool = Pool(4)
BATCH_SIZE = 5
NUM_BATCHES = int(np.ceil(len(paths) / float(BATCH_SIZE)))
for i in range(NUM_BATCHES):
    path_batch = paths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
    l = pool.map(compute, path_batch)

    for j, (e, target) in enumerate(zip(l, y)):
        e1 = e[0]
        e2 = e[1]
        rgb_snippets, flow_snippets = get_rgb_and_flow(e1, e2)

        flow_logits = []
        rgb_logits = []
        for rgb, flow in zip(rgb_snippets, flow_snippets):
            rgb = np.expand_dims(rgb, axis=0)
            flow = np.expand_dims(flow, axis=0)
            flow_logit, _, rgb_logit, _ = i3d.infer(rgb, flow)

            flow_logits.append(flow_logit)
            rgb_logits.append(rgb_logit)
        flow_logits = np.array(flow_logits)
        rgb_logits = np.array(rgb_logits)

        np.save('/home/wits-user/david/data/ucf101/generated/' + path_batch[j].split('/')[-1] + '_flowlogits.npy', flow_logits)
        np.save('/home/wits-user/david/data/ucf101/generated/' + path_batch[j].split('/')[-1] + '_rgblogits.npy', rgb_logits)
