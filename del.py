import numpy as np

def get_cropped_videos(video):
    _, r, c, _ = video.shape
    
    X = []
    while len(X) < 5:
        x = np.random.randint(0, c)
        y = np.random.randint(0, r)
        if y + 224 <= r and x + 224 <= c:
            snippet = video[:, y:y+224, x:x+224, :]
            X.append(snippet)
    return np.array(X)


classes = open('classes.txt').read().splitlines()

train_file = open('/dev/shm/ucfTrainTestlist/trainlist01.txt').read().splitlines()
for l in train_file:
    fn = l.split(' ')[0]
    classidx = int(l.split(' ')[1])
    target = classes[classidx-1]
    path = '/dev/shm/UCF-101/{}/{}'.format(target, fn)



