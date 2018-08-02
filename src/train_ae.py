import os
import torch
import options
import utils

def main():

    # Create result directories
    log_dir = os.path.join(args.result_dir, 'log')
    img_dir = os.path.join(args.result_dir, 'img')
    snapshot_dir = os.path.join(args.result_dir, 'snapshot')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    # Create network
    encoder_class = options.encoder_dict[args.encoder]
    decoder_class = options.decoder_dict[args.decoder]
    discrim_class = options.discrim_dict[args.discrim]

    encoder = None if encoder_class is None else encoder_class(args)
    decoder = None if decoder_class is None else decoder_class(args)
    discrim = None if discrim_class is None else discrim_class(args)

    # Create optimizer
    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())

    optim = torch.optim.Adam(
        encoder_params + decoder_params,
        lr=args.lr, betas=(args.beta1, 0.999))

    discrim_optim = None if discrim is None else \
        torch.optim.Adam(discrim.parameters(), lr=args.lr,
                         betas=(args.beta1, 0.999))

    # Load dataset
    dataset_class = options.dataset_dict[args.dataset]
    train_dataset = dataset_class(args, args.train_set_path)
    valid_dataset = dataset_class(args, args.valid_set_path)

    train_set_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4)
    valid_set_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=1)

    # Create trainer
    trainer_arg_dict = dict()
    trainer_arg_dict['encoder'] = encoder
    trainer_arg_dict['decoder'] = decoder
    trainer_arg_dict['optim'] = optim

    if discrim is not None:
        trainer_arg_dict['discrim'] = discrim
    if discrim_optim is not None:
        trainer_arg_dict['discrim_optim'] = discrim_optim
    trainer_arg_dict['log_dir'] = log_dir

    trainer_class = options.trainer_dict[args.trainer]
    trainer = trainer_class(**trainer_arg_dict)

    if args.snapshot_dir is not None:
        trainer.load_snapshot(args.snapshot_dir)
    trainer.use_cuda(args.cur_device)

    # Training
    num_batch = train_set_loader.__len__()
    infer_batch_dict = next(iter(valid_set_loader))

    for value_dict in trainer.train(train_set_loader, valid_set_loader,
                                    args.start_epoch, args.finish_epoch,
                                    args.valid_intv):

        batch_num = trainer.global_step % num_batch
        epoch = int(trainer.global_step / num_batch)

        if trainer.global_step % args.print_intv == 0:
            print('< Epoch %d/%d, iteration %d/%d >' % \
                (epoch, args.finish_epoch, batch_num, num_batch))
            for key, value in value_dict.items():
                print('\t%s: %f' % (key, value))
            print('')

        if trainer.global_step % num_batch == 0:
            # save images
            x, _, _x = trainer.forward(infer_batch_dict, False)
            save_img_dir = os.path.join(img_dir, 'epoch-%03d' % epoch)
            utils.save_img_batch(save_img_dir, x, valid_dataset.inv_preprocessing, 'x')
            utils.save_img_batch(save_img_dir, _x, valid_dataset.inv_preprocessing, '_x')
            print('< Save image batch: %s >\n' % save_img_dir)

            # decay learning rate
            if epoch != args.start_epoch and epoch % args.lr_decay_intv == 0:
                new_lr = trainer.decay_lr(args.lr_decay_rate)
                print('< Decay learning rate: %f --> %f >\n' %
                      (new_lr/args.lr_decay_rate, new_lr))

            # save snapshot
            if epoch != args.start_epoch and epoch % args.snapshot_intv == 0:
                save_snaphot_dir = os.path.join(snapshot_dir, 'epoch-%03d' % epoch)
                trainer.save_snapshot(save_snaphot_dir)
                print('< Save snapshot: %s >\n' % save_snaphot_dir)

if __name__ == '__main__':
    args = options.train_parser.parse_args()
    main()
