import random

def cvt_rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return r * 0.299 + g * 0.587 + b * 0.114

def sample_patches(img, ppi, size):
    # ppi: patch per image
    # size: patch size
    (h, w) = img.shape
    patches = list()
    for _ in range(ppi):
        y_offset = int(random.uniform(0, h-size+1))
        x_offset = int(random.uniform(0, w-size+1))
        patch = img[y_offset:y_offset+size, x_offset:x_offset+size]
        patches.append(patch)
    return patches
