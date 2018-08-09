import os
import time
import scipy
import torch

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

def forward_masked_z(decoder, z):
    code_size = z.shape[1]

    _x_elmt = list()
    _x_accum = list()
    for i in range(code_size):
        m_elmt = torch.zeros((1, code_size, 1, 1)).cuda()
        m_accum = torch.zeros((1, code_size, 1, 1)).cuda()
        m_elmt[0, i] = 1
        m_accum[0, :i+1] = 1

        z_elmt = m_elmt * z
        z_accum = m_accum * z

        _x_elmt_i = decoder(z_elmt)
        _x_accum_i = decoder(z_accum)
        _x_elmt.append(_x_elmt_i)
        _x_accum.append(_x_accum_i)

    _x_elmt = torch.cat(_x_elmt, dim=0)
    _x_accum = torch.cat(_x_accum, dim=0)
    return _x_elmt, _x_accum
