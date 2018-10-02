import os
import time
import itertools
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

import utils
import option
import eval_armdn

def train():
    # Create result directories------------------------------------------------------------------
    result_dir_dict = utils.create_result_dir(args.result_dir)

    # Create network and optimizer---------------------------------------------------------------
    ae = option.network_dict[args.ae](args)
    ae.build()
    armdn = option.network_dict[args.armdn](args)
    armdn.build()
    train_params = armdn.parameters()
    optim = torch.optim.Adam(train_params, lr=args.lr, betas=(args.beta1, 0.999))

    # Load models and optimizer and use cuda-----------------------------------------------------
    # assert ae.load(args.load_snapshot_dir)
    if armdn.load(args.load_snapshot_dir):
        assert utils.load_optim(optim, args.load_snapshot_dir)
    if len(args.devices) > 1:
        ae = torch.nn.DataParallel(ae)
        armdn = torch.nn.DataParallel(armdn)
    ae.cuda()
    armdn.cuda()

    # Load dataset-------------------------------------------------------------------------------
    dataset_cls = option.dataset_dict[args.dataset]
    train_dataset = dataset_cls(args, args.train_set_path)
    valid_dataset = dataset_cls(args, args.valid_set_path)

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

    # Training-----------------------------------------------------------------------------------
    ae.train(mode=False)
    for param in ae.parameters():
        param.requires_grad = False
    num_batch = train_data_loader.__len__()
    for e in range(args.init_epoch, args.max_epoch + 1):

        # Save image result----------------------------------------------------------------------
        armdn.train(mode=False)
        z_ = armdn.sample(args.batch_size, args.tau)
        x__ = ae.forward(z_, forward_type='decoder')
        save_img_dir = os.path.join(result_dir_dict['img'], 'epoch-%d' % e)
        utils.save_img_batch(save_img_dir, x__, valid_dataset.post_processing, 'gen_img')

        # Evalutaion-----------------------------------------------------------------------------
        if e % args.eval_epoch_intv == 0:
            armdn.train(mode=False)
            for param in train_params:
                param.requires_grad = False
            eval_mse_loss = eval_armdn.evaluate(armdn, ae, valid_data_loader)
            global_step = e * num_batch
            eval_logger.add_scalar('mdn_loss', eval_mse_loss, global_step)

        # Learning rate decay--------------------------------------------------------------------
        if e in args.lr_decay_epochs:
            utils.decay_lr(optim, args.lr_decay_rate)

        # Save snapshot--------------------------------------------------------------------------
        if e in args.save_snapshot_epochs:
            snapshot_dir = os.path.join(result_dir_dict['snapshot'], 'epoch-%d' % e)
            utils.make_dir(snapshot_dir)
            make_ae_link(args.load_snapshot_dir, snapshot_dir, ae)
            assert armdn.save(snapshot_dir)
            assert utils.save_optim(optim, snapshot_dir)

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

            armdn.train(mode=True)
            z = ae.forward(x, forward_type='encoder')
            mu, sig, pi = armdn.forward(z)
            train_mdn_loss = armdn.calc_loss(z, mu, sig, pi)

            optim.zero_grad()
            train_mdn_loss.backward()
            optim.step()
            t2_2 = time.time()

            # Validation-------------------------------------------------------------------------
            if b % args.valid_iter_intv == 0:
                valid_batch_dict = next(valid_set_cycle)
                x = valid_batch_dict['image'].cuda()
                x.requires_grad_(False)

                armdn.train(mode=False)
                z = ae.forward(x, forward_type='encoder')
                mu, sig, pi = armdn.forward(z)
                valid_mdn_loss = armdn.calc_loss(z, mu, sig, pi)

            # Set description and write log------------------------------------------------------
            accum_batching_time += (t1_2 - t1_1)
            accum_training_time += (t2_2 - t2_1)
            train_loader_pbar.set_description(
                '[training] epoch:%d/%d, ' % (e, args.max_epoch) +
                'batching:%.3fs/b, training:%.3fs/b |' % \
                (accum_batching_time / (b + 1), accum_training_time / (b + 1)))

            global_step = e * num_batch + b
            train_logger.add_scalar('mdn_loss', train_mdn_loss.item(), global_step)
            valid_logger.add_scalar('mdn_loss', valid_mdn_loss.item(), global_step)
            t1_1 = time.time()

    # Close logger-------------------------------------------------------------------------------
    train_logger.close()
    valid_logger.close()
    eval_logger.close()

def make_ae_link(load_snapshot_dir, save_snapshot_dir, ae):
    name = ae.__class__.__name__ + '.pth'
    from_path = os.path.join(load_snapshot_dir, name)
    to_path = os.path.join(save_snapshot_dir, name)
    os.symlink(from_path, to_path)

if __name__ == '__main__':
    args = option.train_parser.parse_args()
    args = utils.parse_train_args(args)
    torch.cuda.set_device(args.devices[0])
    train()
