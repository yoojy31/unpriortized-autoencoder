from tqdm import tqdm
import loss

def evaluate(armdn, ae, data_loader):
    accum_mdn_loss = 0
    data_loader_pbar = tqdm(data_loader)

    ae.train(mode=False)
    armdn.train(mode=False)
    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        z = ae.forward(x, 'encoder')
        mu, sig, pi = armdn.forward(z)
        mdn_loss = loss.mdn_loss_fn(z, mu, sig, pi)

        accum_mdn_loss += mdn_loss.item()
        data_loader_pbar.set_description(
            '[evalation] mdn:%.5f |' \
            % (accum_mdn_loss / (b+1)))
    mdn_loss = accum_mdn_loss / data_loader.__len__()
    return mdn_loss
