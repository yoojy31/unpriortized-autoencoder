from collections import OrderedDict
from tqdm import tqdm
import torch
import option
import distribution


def main():
    # Create network and optimizer---------------------------------------------------------------
    ae_class = option.network_dict[args.ae]
    ae = ae_class(args)
    ae.build()

    armdn_class = option.network_dict[args.armdn]
    if armdn_class is not None:
        armdn = armdn_class(args)
        armdn.build()
    else:
        armdn = None

    # Load models and optimizer and use cuda-----------------------------------------------------
    if args.load_snapshot_dir is not None:
        assert ae.load(args.load_snapshot_dir)
        ae.cuda()
        if armdn is not None:
            assert armdn.load(args.load_snapshot_dir)
            armdn.cuda()

    # Load dataset-------------------------------------------------------------------------------
    eval_dataset_cls = option.dataset_dict[args.valid_dataset]
    eval_dataset = eval_dataset_cls(args, args.train_set_path, True)

    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8)

    # Evaluation---------------------------------------------------------------------------------
    evaluate(ae, eval_data_loader, armdn, likelihood=True)


def evaluate(ae, data_loader, armdn=None, likelihood=False):
    accum_loss_dict = OrderedDict()
    data_loader_pbar = tqdm(data_loader)

    for b, batch_dict in enumerate(data_loader_pbar):
        x = batch_dict['image'].cuda()
        x.requires_grad_(False)
        loss_dict = ae.forward(x, forward_type='all')

        if likelihood:
            likelihood_x = cal_likelihood_x(ae, armdn, x, 1.0)
            log_likelihood_x = torch.log(likelihood_x)
            loss_dict['log_likelihood_x'] = log_likelihood_x

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


def cal_likelihood_x(ae, armdn, x, z_sampling=1):
    mu, sig = ae.forward(x, forward_type='encoder')

    likelihood_x = 0
    for _ in range(z_sampling):
        z = mu + sig * torch.randn(sig.size()).cuda()
        _x = ae.forward(z, forward_type='decoder')

        posterior = ae.posterior_pdf(z)
        prior = torch.exp(-1 * armdn.forward(z, forward_type='loss'))
        likelihood = torch.exp(distribution.log_bernoulli_pdf(x, _x))
        likelihood_x += (likelihood * prior / posterior)
    likelihood_x /= len(z_sampling)
    return likelihood_x


if __name__ == "__main__":
    main()
    args = option.eval_parser.parse_args()
