from tqdm import tqdm

def evaluate(armdn, ae, data_loader):
    accum_mdn_loss = 0
    data_loader_pbar = tqdm(data_loader)

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        z = ae.forward(x, forward_type='encoder')
        mu, sig, pi = armdn.forward(z)
        mdn_loss = armdn.calc_loss(z, mu, sig, pi)

        accum_mdn_loss += mdn_loss.item()
        data_loader_pbar.set_description(
            '[evalation] mdn:%.5f |' % (accum_mdn_loss / (b+1)))
    mdn_loss = accum_mdn_loss / data_loader.__len__()
    return mdn_loss
