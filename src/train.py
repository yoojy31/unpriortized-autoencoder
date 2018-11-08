import os
import time
import itertools
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

import utils
import option
import eval_ae
import eval_armdn
from interpolation import interpolate_z

def train():
    # Create result directories------------------------------------------------------------------
    result_dir_dict = utils.create_result_dir(args.result_dir)

    # Create network-----------------------------------------------------------------------------
    ae = option.network_dict[args.ae](args)
    armdn = option.network_dict[args.armdn](args)
    ae.build()
    armdn.build()

    # Create optimizer---------------------------------------------------------------------------
    ae_params = list(ae.encoder.parameters()) + list(ae.decoder.parameters())
    armdn_params = list(armdn.parameters())
    train_params = ae_params + armdn_params
    ae_optim = torch.optim.Adam(ae_params, lr=args.lr, betas=(args.beta1, 0.999))
    armdn_optim = torch.optim.Adam(armdn_params, lr=args.lr, betas=(args.beta1, 0.999))

    # Load models and optimizer and use cuda-----------------------------------------------------
    if args.load_snapshot_dir is not None:
        if ae.load(args.load_snapshot_dir) and armdn.load(args.load_snapshot_dir):
            assert utils.load_optim(ae_optim, args.load_snapshot_dir)
            assert utils.load_optim(armdn_optim, args.load_snapshot_dir)
    if len(args.devices) > 1:
        ae = torch.nn.DataParallel(ae)
        armdn = torch.nn.DataParallel(armdn)
    ae.cuda()
    armdn.cuda()

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
    num_batch = train_data_loader.__len__()
    x_save = next(iter(valid_data_loader))['image'].cuda()
    for e in range(args.init_epoch, args.max_epoch + 1):

        # Save image result----------------------------------------------------------------------
        ae.train(mode=False)
        armdn.train(mode=False)
        _x_save = ae.forward(x_save, forward_type='autoencoder')
        _z = armdn.sample(args.batch_size, args.tau)
        __x_save = ae.forward(_z, forward_type='decoder')

        save_img_dir = os.path.join(result_dir_dict['img'], 'epoch-%d' % e)
        utils.save_img_batch(save_img_dir, x_save, valid_dataset.post_process, 'input_img')
        utils.save_img_batch(save_img_dir, _x_save, valid_dataset.post_process, 'recon_img')
        interpolate_z(x_save, ae, save_img_dir, valid_dataset)
        save_img_dir = os.path.join(save_img_dir, 'generation')
        utils.save_img_batch(save_img_dir, __x_save, valid_dataset.post_process, 'gen_img')
        if 'ae1' in args.ae:
            _x_save = ae.forward(x_save, forward_type='autoencoder', dout=0.0)
            utils.save_img_batch(save_img_dir, _x_save, valid_dataset.post_process, 'recon_img(wo_dout)')

        # Evalutaion-----------------------------------------------------------------------------
        if e % args.eval_epoch_intv == 0:
            ae.train(mode=False)
            armdn.train(mode=False)
            for param in train_params:
                param.requires_grad = False

            ae_eval_loss_dict = eval_ae.evaluate(ae, valid_data_loader)
            armdn_eval_loss = eval_armdn.evaluate(armdn, ae, valid_data_loader)
            global_step = e * num_batch
            for key, value in ae_eval_loss_dict.items():
                eval_logger.add_scalar(key, value, global_step)
            eval_logger.add_scalar('mdn', armdn_eval_loss, global_step)

        # Learning rate decay--------------------------------------------------------------------
        if e in args.lr_decay_epochs:
            utils.decay_lr(ae_optim, args.lr_decay_rate)
            utils.decay_lr(armdn_optim, args.lr_decay_rate)

        # Save snapshot--------------------------------------------------------------------------
        if e in args.save_snapshot_epochs:
            snapshot_dir = os.path.join(result_dir_dict['snapshot'], 'epoch-%d' % e)
            utils.make_dir(snapshot_dir)
            assert ae.save(snapshot_dir)
            assert armdn.save(snapshot_dir)
            assert utils.save_optim(ae_optim, snapshot_dir, tag='ae')
            assert utils.save_optim(armdn_optim, snapshot_dir, tag='armdn')

        # Break loop-----------------------------------------------------------------------------
        if e == args.max_epoch:
            break

        # Train one epoch------------------------------------------------------------------------
        for param in train_params:
            param.requires_grad = True
        valid_set_cycle = itertools.cycle(valid_data_loader)
        train_loader_pbar = tqdm(train_data_loader)
        accum_batching_time =   0
        accum_training_time = 0

        # Get batch dict-------------------------------------------------------------------------
        t1_1 = time.time()
        for b, train_batch_dict in enumerate(train_loader_pbar):
            t1_2 = time.time()

            # Train one step---------------------------------------------------------------------
            t2_1 = time.time()

            x = train_batch_dict['image'].cuda()
            x.requires_grad_(False)

            ae.train(mode=True)
            armdn.train(mode=True)

            if args.input_drop > 0:
                if args.patch_drop:
                    x_crpt = utils.drop_patch(x, args.input_drop)
                else:
                    x_crpt = utils.drop_pixel(x, args.input_drop)
                z = ae.forward(x_crpt, forward_type='encoder')
            else:
                z = ae.forward(x, forward_type='encoder')

            if args.z_mask_warm_up > 1:
                mask_rate = min((e + 1) / args.z_mask_warm_up, 1)
                mask_idx = int(args.z_size * mask_rate)
                mask = torch.zeros((1, args.z_size, 1, 1)).cuda()
                mask[0, :mask_idx] = 1
                z = z * mask
                z_armdn = z[:, :mask_idx].detach()
            else:
                z_armdn = z.detach()

            x_ = ae.forward(z, forward_type='decoder')
            mu, sig, pi = armdn.forward(z)

            train_ae_loss_dict = ae.calc_loss(x_, x)
            train_ae_recon_loss = train_ae_loss_dict['pixel'] + train_ae_loss_dict['perc']
            train_ae_loss = train_ae_recon_loss

            mu, sig, pi = armdn.forward(z_armdn)
            train_armdn_loss = armdn.calc_loss(z_armdn, mu, sig, pi)

            ae_optim.zero_grad()
            train_ae_loss.backward()
            ae_optim.step()
            t2_2 = time.time()

            armdn_optim.zero_grad()
            train_armdn_loss.backward()
            armdn_optim.step()
            t2_2 = time.time()

            global_step = e * num_batch + b
            if b % 10 == 0:
                train_logger.add_scalar('pixel', train_ae_loss_dict['pixel'].item(), global_step)
                train_logger.add_scalar('perc', train_ae_loss_dict['perc'].item(), global_step)
                train_logger.add_scalar('mdn', train_armdn_loss.item(), global_step)

            # Validation-------------------------------------------------------------------------
            if b % args.valid_iter_intv == 0:
                valid_batch_dict = next(valid_set_cycle)
                x = valid_batch_dict['image'].cuda()
                x.requires_grad_(False)

                ae.train(mode=False)
                armdn.train(mode=False)
                z = ae.forward(x, forward_type='encoder')
                x_ = ae.forward(z, forward_type='decoder')
                mu, sig, pi = armdn.forward(z)

                valid_ae_loss_dict = ae.calc_loss(x_, x)
                valid_armdn_loss = armdn.calc_loss(z, mu, sig, pi)

                global_step = e * num_batch + b
                if b % 10 == 0:
                    valid_logger.add_scalar('pixel', valid_ae_loss_dict['pixel'].item(), global_step)
                    valid_logger.add_scalar('perc', valid_ae_loss_dict['perc'].item(), global_step)
                    valid_logger.add_scalar('mdn', valid_armdn_loss.item(), global_step)

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

if __name__ == '__main__':
    args = option.train_parser.parse_args()
    args = utils.parse_train_args(args)
    torch.cuda.set_device(args.devices[0])
    train()
