import os
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter

import utils
import option
import eval_armdn

def train():
    # Create result directories------------------------------------------------------------------
    result_dir_dict = utils.create_result_dir(args.result_dir)

    # Create network and optimizer---------------------------------------------------------------
    armdn = option.network_dict[args.armdn](args)
    armdn.build()
    train_params = armdn.parameters()
    optim = torch.optim.Adam(train_params, lr=args.lr, betas=(args.beta1, 0.999))

    # Load models and optimizer and use cuda-----------------------------------------------------
    if args.load_snapshot_dir is not None:
        if armdn.load(args.load_snapshot_dir):
            assert utils.load_optim(optim, args.load_snapshot_dir)
    if len(args.devices) > 1:
        armdn = torch.nn.DataParallel(armdn)
    armdn.cuda()

    # Autoencoder--------------------------------------------------------------------------------
    ae_class = option.network_dict[args.ae]
    if ae_class is not None:
        ae = ae_class(args)
        ae.build()
        if len(args.devices) > 1:
            ae = torch.nn.DataParallel(ae)
        assert ae.load(args.load_snapshot_dir)
        ae.cuda()
    else:
        ae = None

    # Load dataset-------------------------------------------------------------------------------
    train_dataset_cls = option.dataset_dict[args.train_dataset]
    valid_dataset_cls = option.dataset_dict[args.valid_dataset]
    train_dataset = train_dataset_cls(args, args.train_set_path, True)
    valid_dataset = valid_dataset_cls(args, args.valid_set_path, True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4)

    # Create logger------------------------------------------------------------------------------
    train_logger = SummaryWriter(result_dir_dict['train'])
    valid_logger = SummaryWriter(result_dir_dict['valid'])
    eval_logger = SummaryWriter(result_dir_dict['eval'])

    # -------------------------------------------------------------------------------------------
    # Training ----------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------
    if ae_class is not None:
        ae.encoder.train(mode=True)
        ae.decoder.train(mode=False)
        for param in ae.parameters():
            param.requires_grad = False

    # Latent variable ordering-------------------------------------------------------------------
    if args.ordering and ae_class is not None:
        var = calc_z_variation(ae, train_data_loader)
        order = (var * -1).argsort().argsort()
        armdn.set_order(order)
        print('[z_order]', armdn.order)

    gmmn = True if args.armdn == 'gmmn' else False
    num_batch = train_data_loader.__len__()
    for e in range(args.init_epoch, args.max_epoch + 1):

        # Save image result----------------------------------------------------------------------
        if ae_class is not None:
            armdn.train(mode=False)
            z_ = armdn.sample(args.batch_size, args.tau)
            if 'ae1' in args.ae:
                z_ = ae.dropout(z_, args.z_dout_rate)
            x__ = ae.forward(z_, forward_type='decoder')
            save_img_dir = os.path.join(result_dir_dict['img'], 'epoch-%d' % e)
            utils.save_img_batch(save_img_dir, x__, valid_dataset.post_process, 'gen_img')

        # Evalutaion-----------------------------------------------------------------------------
        if e % args.eval_epoch_intv == 0:
            armdn.train(mode=False)
            for param in train_params:
                param.requires_grad = False

            eval_mse_loss = eval_armdn.evaluate(armdn, ae, valid_data_loader, gmmn)
            global_step = e * num_batch
            eval_logger.add_scalar('mdn_loss', eval_mse_loss, global_step)

        # Learning rate decay--------------------------------------------------------------------
        if e in args.lr_decay_epochs:
            utils.decay_lr(optim, args.lr_decay_rate)

        # Save snapshot--------------------------------------------------------------------------
        if e in args.save_snapshot_epochs:
            snapshot_dir = os.path.join(result_dir_dict['snapshot'], 'epoch-%d' % e)
            utils.make_dir(snapshot_dir)
            assert armdn.save(snapshot_dir)
            assert utils.save_optim(optim, snapshot_dir)
            if ae_class is not None:
                make_ae_link(args.load_snapshot_dir, snapshot_dir, ae)

        # Break loop-----------------------------------------------------------------------------
        if e == args.max_epoch:
            break

        # Train one epoch------------------------------------------------------------------------
        for param in train_params:
            param.requires_grad = True
        valid_set_cycle = itertools.cycle(valid_data_loader)
        train_loader_pbar = tqdm(train_data_loader)
        accum_batching_time = 0
        accum_training_time = 0

        # Get batch dict-------------------------------------------------------------------------
        t1_1 = time.time()
        for b, train_batch_dict in enumerate(train_loader_pbar):
            t1_2 = time.time()

            # Train one step---------------------------------------------------------------------
            t2_1 = time.time()

            x = train_batch_dict['image'].cuda()
            x.requires_grad_(False)
            if ae_class is not None:
                z = ae.forward(x, forward_type='encoder', dout=0.0)
                x = z

            armdn.train(mode=True)
            if not gmmn:
                mu, sig, pi = armdn.forward(x)
                train_mdn_loss = armdn.calc_loss(x, mu, sig, pi)
            else:
                _z = armdn.forward(x.shape[0])
                train_mdn_loss = armdn.calc_loss(_z, x)
            # mu, sig, pi = armdn.forward(x)
            # train_mdn_loss = armdn.calc_loss(x, mu, sig, pi)

            optim.zero_grad()
            train_mdn_loss.backward()
            optim.step()
            t2_2 = time.time()

            global_step = e * num_batch + b
            if b % 10 == 0:
                train_logger.add_scalar('mdn_loss', train_mdn_loss.item(), global_step)

            # Validation-------------------------------------------------------------------------
            if b % args.valid_iter_intv == 0:
                valid_batch_dict = next(valid_set_cycle)
                x = valid_batch_dict['image'].cuda()
                x.requires_grad_(False)
                if ae_class is not None:
                    z = ae.forward(x, forward_type='encoder', dout=0.0)
                    x = z

                armdn.train(mode=False)
                if not gmmn:
                    mu, sig, pi = armdn.forward(x)
                    valid_mdn_loss = armdn.calc_loss(x, mu, sig, pi)
                else:
                    _z = armdn.forward(x.shape[0])
                    valid_mdn_loss = armdn.calc_loss(_z, x)
                valid_logger.add_scalar('mdn_loss', valid_mdn_loss.item(), global_step)

            # Set description and write log------------------------------------------------------
            accum_batching_time += (t1_2 - t1_1)
            accum_training_time += (t2_2 - t2_1)
            if b % 10 == 0:
                train_loader_pbar.set_description(
                    '[training] epoch:%d/%d, ' % (e, args.max_epoch) +
                    'batching:%.3fs/b, training:%.3fs/b |' % \
                    (accum_batching_time / (b + 1), accum_training_time / (b + 1)))
            t1_1 = time.time()

    # Close logger-------------------------------------------------------------------------------
    train_logger.close()
    valid_logger.close()
    eval_logger.close()

def calc_z_variation(ae, train_data_loader):
    accum_var = np.squeeze(np.zeros(args.z_size))

    train_loader_pbar = tqdm(train_data_loader)
    for train_batch_dict in train_loader_pbar:
        x = train_batch_dict['image'].cuda()
        x.requires_grad_(False)

        z = ae.forward(x, forward_type='encoder')
        var = torch.var(torch.squeeze(z), dim=0)
        var = var.detach().cpu().numpy()
        accum_var += var

        train_loader_pbar.set_description('[z-varation] |')
    var = accum_var / train_data_loader.__len__()
    return var

def make_ae_link(load_snapshot_dir, save_snapshot_dir, ae):
    name = ae.__class__.__name__ + 'encoder.pth'
    from_path = os.path.join(load_snapshot_dir, name)
    to_path = os.path.join(save_snapshot_dir, name)
    os.symlink(from_path, to_path)

    name = ae.__class__.__name__ + 'decoder.pth'
    from_path = os.path.join(load_snapshot_dir, name)
    to_path = os.path.join(save_snapshot_dir, name)
    os.symlink(from_path, to_path)

if __name__ == '__main__':
    args = option.train_parser.parse_args()
    args = utils.parse_train_args(args)
    torch.cuda.set_device(args.devices[0])
    train()
