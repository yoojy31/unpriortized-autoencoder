import os
import scipy
import torch
import utils
import option
# input : x_save -> (batch, ch, w, h)?
# ouput: None
# interpolate save

def main():
    utils.make_dir(args.result_dir)

    ae = option.network_dict[args.ae](args)
    ae.build()
    if args.load_snapshot_dir is not None:
        assert ae.load(args.load_snapshot_dir)
    ae.cuda()
    ae.train(mode=False)

    dataset_cls = option.dataset_dict[args.dataset]
    dataset = dataset_cls(args, args.dataset_path, True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    x = next(iter(data_loader))['image'].cuda()
    interpolate_z(x, ae, args.pairs, args.result_dir, dataset)

    sbatch_img_dir = os.path.join(args.result_dir, 'image')
    utils.make_dir(sbatch_img_dir)
    utils.save_img_batch(sbatch_img_dir, x, dataset.post_process, 'image')

def interpolate_z(x, ae, pairs, save_img_dir, valid_dataset):
    # pairs example: ((1, 2), (0, 10))
    # entirely random select? entirely fixed select? siquential?
    # nums = [x for x in range(len(x))]
    # random.shuffle(nums)

    input_x = x[:len(pairs) * 2].clone()
    for i, pair in enumerate(pairs):
        input_x[i*2] = x[pair[0]]
        input_x[i*2+1] =x[pair[1]]

    z = ae.forward(input_x, forward_type='encoder') # maybe, (2,code_size,1,1)
    save_img_dir = os.path.join(save_img_dir, 'interpolate')
    utils.make_dir(save_img_dir)
    utils.save_img_batch(save_img_dir, input_x, valid_dataset.post_process, 'input_img')

    for i in range(0, z.size(0), 2):
        for _r in range(0, 11, 2):
            r = _r / 10
            z1, z2 = z[i:i+1], z[i+1:i+2]
            z_intp = z1 * (1 - r) + z2 * r
            _x_intp = ae.forward(z_intp, forward_type='decoder')
            _x_intp = valid_dataset.post_process(torch.squeeze(_x_intp))
            save_data_path = os.path.join(save_img_dir, '%03d-intp_img_%.2f.png' % (i, r))
            scipy.misc.imsave(save_data_path, _x_intp)

if __name__=="__main__":
    args = option.interp_parser.parse_args()
    args.pairs = args.pairs.split('|')
    for i, pair in enumerate(args.pairs):
        pair = pair.split(',')
        args.pairs[i] = (int(pair[0]), int(pair[1]))
    main()
