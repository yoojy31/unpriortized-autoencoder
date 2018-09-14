from tqdm import tqdm
import torch.nn.functional as F

def evalate(autoencoder, data_loader):
    accum_mse_loss = 0
    data_loader_pbar = tqdm(data_loader)

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        autoencoder.train(mode=False)
        x_ = autoencoder.forward(x, 'all')
        mse_loss = F.mse_loss(x_, x)

        accum_mse_loss += mse_loss.item()
        data_loader_pbar.set_description(
            '[evalation] mse:%.5f |' \
            % (accum_mse_loss / (b+1)))
    mse_loss = accum_mse_loss / data_loader.__len__()
    return mse_loss
