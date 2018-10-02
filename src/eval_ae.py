from collections import OrderedDict
from tqdm import tqdm

def evalate(ae, data_loader):
    accum_loss_dict = OrderedDict()
    data_loader_pbar = tqdm(data_loader)

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)
        loss_dict = ae.forward(x, forward_type='all')

        for key, value in loss_dict.items():
            try:
                accum_loss_dict[key] += value.item()
            except KeyError:
                accum_loss_dict[key] = 0
                accum_loss_dict[key] += value.item()

        loss_str = ''
        for key, value in accum_loss_dict.items():
            loss_str += '%s:%.5f ' % (key, value/(b+1))
        data_loader_pbar.set_description('[evalation] %s|' % loss_str)

    for key, value in accum_loss_dict.items():
        accum_loss_dict[key] = value / data_loader.__len__()
    mean_loss_dict = accum_loss_dict
    return mean_loss_dict
