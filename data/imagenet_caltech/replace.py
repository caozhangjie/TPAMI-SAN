import os

for fname in os.listdir('.'):
    if 'txt' in fname:
        dd = open(fname, 'r').read().replace('/Checkpoint/liangjian/da_dataset/office_home', '/workspace2/caozhangjie/Transfer-Learning-Library/examples/domain_adaptation/classification/data/office-home')
        open(fname, 'w').write(dd)

