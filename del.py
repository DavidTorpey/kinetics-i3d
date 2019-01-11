
classes = open('classes.txt').read().splitlines()

train_file = open('/dev/shm/ucfTrainTestlist/trainlist01.txt').read().splitlines()
for l in train_file:
    fn = l.split(' ')[0]
    classidx = int(l.split(' ')[1])
    target = classes[classidx-1]
    path = '/dev/shm/UCF-101/{}/{}'.format(target, fn)



