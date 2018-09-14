import os
import sys
from collections import OrderedDict
import scipy
import torch

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def create_result_dir(result_dir):
    log_dir = os.path.join(result_dir, 'log')
    result_dir_dict = {
        'train': os.path.join(log_dir, 'train'),
        'valid': os.path.join(log_dir, 'valid'),
        'eval': os.path.join(log_dir, 'eval'),
        'img': os.path.join(result_dir, 'img'),
        'snapshot': os.path.join(result_dir, 'snapshot')}

    make_dir(result_dir)
    make_dir(log_dir)
    make_dir(result_dir_dict['train'])
    make_dir(result_dir_dict['valid'])
    make_dir(result_dir_dict['eval'])
    make_dir(result_dir_dict['img'])
    make_dir(result_dir_dict['snapshot'])
    return result_dir_dict

def print_ordered_dict(ordered_dict, indent='', tag='tag'):
    dict_dict = OrderedDict()
    sys.stdout.write(indent + '(%s)' % tag)
    for key, item in ordered_dict.items():
        if isinstance(item, dict):
            dict_dict[key] = item
        else:
            sys.stdout.write(', %s: %s' % (key, str(item)))
    sys.stdout.write('\n')

    for key, item_dict in dict_dict.items():
        indent_ = indent + '\t'
        print_ordered_dict(item_dict, indent=indent_, tag=key)

def save_img_batch(save_dir, imgs, processing, tag):
    make_dir(save_dir)

    for i, img in enumerate(imgs):
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
