from tqdm import tqdm
import option

def main():
    pass

def evaluate(armdn, ae, data_loader, gmmn=False):
    accum_mdn_loss = 0
    data_loader_pbar = tqdm(data_loader)

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        if ae is not None:
            z = ae.forward(x, forward_type='encoder', dout=0.0)
            x = z

        if not gmmn:
            mu, sig, pi = armdn.forward(x)
            mdn_loss = armdn.calc_loss(x, mu, sig, pi)
        else:
            _z = armdn.forward(x.shape[0])
            mdn_loss = armdn.calc_loss(_z, x)

        accum_mdn_loss += mdn_loss.item()
        data_loader_pbar.set_description(
            '[evalation] mdn:%.5f |' % (accum_mdn_loss / (b+1)))
    mdn_loss = accum_mdn_loss / data_loader.__len__()
    return mdn_loss

if __name__ == "__main__":
    main()
