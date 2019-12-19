import os
from tqdm import tqdm
import torch
import utils
import option

def split_dataset():
    train_dir = os.path.join(args.result_dir, 'train')
    valid_dir = os.path.join(args.result_dir, 'valid')
    test_dir = os.path.join(args.result_dir, 'test')
    utils.make_dir(train_dir)
    utils.make_dir(valid_dir)
    utils.make_dir(test_dir)

    dataset_cls = option.dataset_dict[args.dataset]
    train_dataset = dataset_cls(args, args.train_set_path, True)
    test_dataset = dataset_cls(args, args.test_set_path, True)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2)

    print('The number of samples in training dataset: %d' % train_dataset.__len__())

    train_loader_pbar = tqdm(train_data_loader)
    for i, train_batch in enumerate(train_loader_pbar):
        img = train_batch['image']
        if i < args.split_train_idx:
            utils.save_img_batch(train_dir, img, train_dataset.post_process, '%07d' % i)
        else:
            utils.save_img_batch(valid_dir, img, train_dataset.post_process, '%07d' % i)

    test_loader_pbar = tqdm(test_data_loader)
    for j, test_batch in enumerate(test_loader_pbar):
        img = test_batch['image']
        if j < args.split_train_idx:
            utils.save_img_batch(test_dir, img, test_dataset.post_process, '%07d' % (i + j))

if __name__ == '__main__':
    args = option.dataset_split_parser.parse_args()
    split_dataset()
