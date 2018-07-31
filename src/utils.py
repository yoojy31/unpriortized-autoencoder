import os
import time
import scipy

def timer(f, *args):
    time1 = time.time()
    result = f(*args)
    time2 = time.time()
    return time2 - time1, result

def save_img_batch(save_dir, img_batch, processing, tag):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, img in enumerate(img_batch):
        img = processing(img)
        save_data_path = os.path.join(save_dir, '%03d-%s.png' % (i, tag))
        scipy.misc.imsave(save_data_path, img)
