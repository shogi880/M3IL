import numpy as np
import tqdm
import glob

files = glob.glob('/root/share/TecNets/datasets/768_push_the_N_to_the_red_area/*.npy')

for file in tqdm(files):
    content = np.load(file)
    print("before...", content.shape)
    content = content.reshape(768)
    print("after...", content.shape)
    # np.save(file, content)
print('done: ', len(files))