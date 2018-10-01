import os
import time
import itertools
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

import loss
import utils
import option
import eval_ae

def train():
    # Create result directories--------------------------------------------------------
    result_dir_dict = utils.create_result_dir(args.result_dir)

    # Create network and optimizer-----------------------------------------------------
    ae = option.network_dict[args.ae](args)
    ae.build()
    params = ae.parameters()
    optim = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    perc_loss_fn = loss.PerceptualLoss()

    # Load models and optimizer and use cuda-------------------------------------------
    if args.load_snapshot_dir is not None:
        assert ae.load(args.load_snapshot_dir)
        assert utils.load_optim(optim, args.load_snapshot_dir)
    if len(args.devices) > 1:
        ae = torch.nn.DataParallel(ae)
    perc_loss_fn.cuda()
    ae.cuda()

    # Load dataset---------------------------------------------------------------------
    dataset_cls = option.dataset_dict[args.dataset]
    train_dataset = dataset_cls(args, args.train_set_path)
    valid_dataset = dataset_cls(args, args.valid_set_path)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4)

    # Create logger--------------------------------------------------------------------
    train_logger = SummaryWriter(result_dir_dict['train'])
    valid_logger = SummaryWriter(result_dir_dict['valid'])
    eval_logger = SummaryWriter(result_dir_dict['eval'])

    # Training-------------------------------------------------------------------------
    num_batch = train_data_loader.__len__()
    x_save = next(iter(valid_data_loader))['image'].cuda()
    for e in range(args.init_epoch, args.max_epoch + 1):

        # Save image result------------------------------------------------------------
        ae.train(mode=False)
        _x_save = ae.forward(x_save, 'all')
        save_img_dir = os.path.join(result_dir_dict['img'], 'epoch-%d' % e)
        utils.save_img_batch(
            save_img_dir, x_save, valid_dataset.inv_preprocessing, 'input_img')
        utils.save_img_batch(
            save_img_dir, _x_save, valid_dataset.inv_preprocessing, 'recon_img')

        # Evalutaion-------------------------------------------------------------------
        if e % args.eval_epoch_intv == 0:
            eval_mse_loss, eval_perc_loss = eval_ae.evalate(
                args, ae, valid_data_loader, perc_loss_fn)
            global_step = e * num_batch
            eval_logger.add_scalar('mse_loss', eval_mse_loss, global_step)
            eval_logger.add_scalar('perc_loss', eval_perc_loss, global_step)

        # Learning rate decay----------------------------------------------------------
        if e in args.lr_decay_epochs:
            utils.decay_lr(optim, args.lr_decay_rate)

        # Save snapshot----------------------------------------------------------------
        if e in args.save_snapshot_epochs:
            snapshot_dir = os.path.join(
                result_dir_dict['snapshot'], 'epoch-%d' % e)
            utils.make_dir(snapshot_dir)
            if len(args.devices) > 1:
                assert ae.module.save(snapshot_dir)
            else:
                assert ae.save(snapshot_dir)
            assert utils.save_optim(optim, snapshot_dir)

        # Break loop-------------------------------------------------------------------
        if e == args.max_epoch:
            break

        # Train one epoch--------------------------------------------------------------
        valid_set_cycle = itertools.cycle(valid_data_loader)
        train_loader_pbar = tqdm(train_data_loader)
        accum_batching_time = 0
        accum_training_time = 0

        # Get batch dict---------------------------------------------------------------
        t1_1 = time.time()
        for b, train_batch_dict in enumerate(train_loader_pbar):
            t1_2 = time.time()

            # Train one step-----------------------------------------------------------
            t2_1 = time.time()
            x = train_batch_dict['image'].cuda()
            x.requires_grad_(True)

            ae.train(mode=True)
            _x = ae.forward(x, 'all')
            train_mse_loss = args.mse_w * torch.nn.functional.mse_loss(_x, x)
            train_perc_loss = args.perc_w * perc_loss_fn.forward(_x, x)

            optim.zero_grad()
            (train_mse_loss + train_perc_loss).backward()
            optim.step()
            t2_2 = time.time()

            # Validation---------------------------------------------------------------
            if b % args.valid_iter_intv == 0:
                valid_batch_dict = next(valid_set_cycle)
                x = valid_batch_dict['image'].cuda()
                x.requires_grad_(False)

                ae.train(mode=False)
                _x = ae.forward(x, 'all')
                valid_mse_loss = args.mse_w * torch.nn.functional.mse_loss(_x, x)
                valid_perc_loss = args.perc_w * perc_loss_fn.forward(_x, x)

            # Set description and write log--------------------------------------------
            accum_batching_time += (t1_2 - t1_1)
            accum_training_time += (t2_2 - t2_1)
            train_loader_pbar.set_description(
                '[training] epoch:%d/%d, ' % (e, args.max_epoch) +
                'batching:%.3fs/b, training:%.3fs/b |' % \
                (accum_batching_time / (b + 1), accum_training_time / (b + 1)))

            global_step = e * num_batch + b
            train_logger.add_scalar('mse_loss', train_mse_loss.item(), global_step)
            valid_logger.add_scalar('mse_loss', valid_mse_loss.item(), global_step)
            train_logger.add_scalar('perc_loss', train_perc_loss.item(), global_step)
            valid_logger.add_scalar('perc_loss', valid_perc_loss.item(), global_step)
            t1_1 = time.time()

    # Close logger---------------------------------------------------------------------
    train_logger.close()
    valid_logger.close()
    eval_logger.close()

if __name__ == '__main__':
    args = option.train_parser.parse_args()
    args = utils.parse_train_args(args)
    torch.cuda.set_device(args.devices[0])
    train()
