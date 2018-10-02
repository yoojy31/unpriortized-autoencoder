from tqdm import tqdm
import torch

def evalate(args, ae, data_loader, perc_loss_fn):
    accum_mse_loss = 0
    accum_perc_loss = 0
    data_loader_pbar = tqdm(data_loader)

    ae.train(mode=False)
    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)
        _x = ae.forward(x, 'all')

        x.detach()
        _x.detach()
        mse_loss = args.mse_w * torch.nn.functional.mse_loss(_x, x)
        perc_loss = args.perc_w * perc_loss_fn.forward(_x, x)

        accum_mse_loss += mse_loss.item()
        accum_perc_loss += perc_loss.item()

        data_loader_pbar.set_description(
            '[evalation] mse:%.5f perc:%.5f |' \
            % ((accum_mse_loss / (b+1)), (accum_perc_loss / (b+1))))
    mse_loss = accum_mse_loss / data_loader.__len__()
    perc_loss = accum_perc_loss / data_loader.__len__()
    return mse_loss, perc_loss
