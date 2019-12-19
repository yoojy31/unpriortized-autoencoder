import numpy as np
from tqdm import tqdm
import torch
# import option
# from kde import kde_eval_mnist, kde_eval_tfd

def main():
    pass

def evaluate(armdn, ae, data_loader, gmmn=False):
    data_loader_pbar = tqdm(data_loader)

    # mdn loss-----------------------------------------------------
    mdn_loss_list = list()
    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        if ae is not None:
            z = ae.forward(x, forward_type='encoder', dout=0.0)
            x = z

        if not gmmn:
            mu, sig, pi = armdn.forward(x)
            mdn_loss = armdn.calc_loss(x, mu, sig, pi, average=False)
        else:
            _z = armdn.forward(x.shape[0])
            mdn_loss = armdn.calc_loss(_z, x, average=False)

        mdn_loss_list.append(mdn_loss.detach().cpu().numpy())
        data_loader_pbar.set_description('[evalation] mdn: |')

    mdn_loss = np.concatenate(mdn_loss_list, axis=0)
    mdn_mean, mdn_std = np.mean(mdn_loss), np.std(mdn_loss)
    print('\tmdn_mean: %f, mdn_std: %f' % (mdn_mean, mdn_std))
    # mdn_loss = accum_mdn_loss / data_loader.__len__()

    # kde log likelihood-------------------------------------------
    # kde_eval_mnist(ae, armdn, )

    return mdn_mean, mdn_std

def evaluate_two(armdn, ae, data_loader):
    data_loader_pbar = tqdm(data_loader)

    ll_1_list = list()
    ll_2_list = list()
    ll_1_i_list = list()
    ll_2_i_list = list()
    ll_i_list = list()

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        x_1 = x[range(0, x.shape[0], 2)].clone()
        x_2 = x[range(1, x.shape[0], 2)].clone()

        if ae is not None:
            z_1 = ae.forward(x_1, forward_type='encoder', dout=0.0)
            z_2 = ae.forward(x_2, forward_type='encoder', dout=0.0)
            x_1 = z_1
            x_2 = z_2

        rand_idx1 = torch.randint(0, x_1.shape[0], (x_1.shape[0],)).long()
        rand_idx2 = torch.randint(0, x_2.shape[0], (x_2.shape[0],)).long()

        lls_1 = -1 * torch.sum(calc_mdn_loss(armdn, x_1), dim=1).detach().cpu().numpy()
        lls_2 = -1 * torch.sum(calc_mdn_loss(armdn, x_2), dim=1).detach().cpu().numpy()
        lls_1_i = lowest_interp_ll(armdn, x_1[rand_idx1], x_1[rand_idx2], weights=[0.9])
        lls_2_i = lowest_interp_ll(armdn, x_2[rand_idx1], x_2[rand_idx2], weights=[0.9])
        lls_i = lowest_interp_ll(armdn, x_1, x_2, weights=[0.9])

        ll_1_list.append(lls_1)
        ll_2_list.append(lls_2)
        ll_1_i_list.append(lls_1_i)
        ll_2_i_list.append(lls_2_i)
        ll_i_list.append(lls_i)
        data_loader_pbar.set_description('[evalation] |')

    lls_1 = np.concatenate(ll_1_list, axis=0)
    lls_2 = np.concatenate(ll_2_list, axis=0)
    lls_1_i = np.concatenate(ll_1_i_list, axis=0)
    lls_2_i = np.concatenate(ll_2_i_list, axis=0)
    lls_i = np.concatenate(ll_i_list, axis=0)

    ll_mean_1, ll_std_1 = np.mean(lls_1), np.std(lls_1)
    ll_mean_2, ll_std_2 = np.mean(lls_2), np.std(lls_2)
    ll_mean_1_i, ll_std_1_i = np.mean(lls_1_i), np.std(lls_1_i)
    ll_mean_2_i, ll_std_2_i = np.mean(lls_2_i), np.std(lls_2_i)
    ll_mean_i, ll_std_i = np.mean(lls_i), np.std(lls_i)

    # print('[ALL MEAN, STD]')
    print('\tll_mean_1: %f, ll_std_1: %f' % (ll_mean_1, ll_std_1))
    print('\tll_mean_2: %f, ll_std_2: %f' % (ll_mean_2, ll_std_2))
    print('\tll_mean_1_i: %f, ll_std_1_i: %f' % (ll_mean_1_i, ll_std_1_i))
    print('\tll_mean_2_i: %f, ll_std_2_i: %f' % (ll_mean_2_i, ll_std_2_i))
    print('\tll_mean_i: %f, ll_std_i: %f' % (ll_mean_i, ll_std_i))

    # args_low_1 = lls_1.argsort()
    # args_low_2 = lls_2.argsort()
    # args_low_i = lls_i.argsort()
    # args_low_1_i = lls_1_i.argsort()
    # args_low_2_i = lls_2_i.argsort()

    # ll_mean_1, ll_std_1 = np.mean(lls_1[args_low_1][:500]), np.std(lls_1[args_low_1][:500])
    # ll_mean_2, ll_std_2 = np.mean(lls_2[args_low_2][:500]), np.std(lls_2[args_low_2][:500])
    # ll_mean_i, ll_std_i = np.mean(lls_i[args_low_i][:500]), np.std(lls_i[args_low_i][:500])
    # ll_mean_1_i, ll_std_1_i = np.mean(lls_1_i[args_low_1_i][:500]), np.std(lls_1_i[args_low_1_i][:500])
    # ll_mean_2_i, ll_std_2_i = np.mean(lls_2_i[args_low_2_i][:500]), np.std(lls_2_i[args_low_2_i][:500])

    # print('[LOWEST-500 MEAN, STD]')
    # print('\tll_mean_1: %f, ll_std_1: %f' % (ll_mean_1, ll_std_1))
    # print('\tll_mean_2: %f, ll_std_2: %f' % (ll_mean_2, ll_std_2))
    # print('\tll_mean_i: %f, ll_std_i: %f' % (ll_mean_i, ll_std_i))
    # print('\tll_mean_1_i: %f, ll_std_1_i: %f' % (ll_mean_1_i, ll_std_1_i))
    # print('\tll_mean_2_i: %f, ll_std_2_i: %f' % (ll_mean_2_i, ll_std_2_i))

def lowest_interp_ll(armdn, x1, x2, weights=[0.3, 0.5, 0.7]):
    ll_list = list()
    for w in weights:
        x_i = x1 * w + x2 * (1 - w)
        ll = -1 * torch.sum(calc_mdn_loss(armdn, x_i, False), dim=1, keepdim=True)
        ll_list.append(ll.detach().cpu().numpy())
    lowest_ll = np.min(np.concatenate(ll_list, axis=1), axis=1)
    return lowest_ll

def calc_mdn_loss(armdn, x, average=False):
    mu, sig, pi = armdn.forward(x)
    mdn_loss = armdn.calc_loss(x, mu, sig, pi, average=average)
    return mdn_loss


if __name__ == "__main__":
    main()
