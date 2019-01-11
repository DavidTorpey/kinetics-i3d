import numpy as np
from compute_i3d import I3DInferrer as I3D



i3d = I3D('/home/david/Music/ucf101_I3D/rgb/train1/model.ckpt-5000', '/home/david/Music/ucf101_I3D/flow/train1/model.ckpt-5000')

rgb = np.load('/home/david/Documents/code/bits-and-bobs/rgb.npy')
flow = np.load('/home/david/Documents/code/bits-and-bobs/flow.npy')

i3d.infer(rgb, flow)
