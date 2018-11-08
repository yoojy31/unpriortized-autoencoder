import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter
# import matplotlib.patches as mpatches
import torch
import utils
import option

MNIST_NAME = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
MNIST_COLOR = ('gray', 'purple', 'blue', 'pink',
    'brown', 'orange', 'green', 'magenta', 'cyan', 'black')
TWO_NAME = ('0', '1')
TWO_COLOR = ('blue', 'orange')

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

    z_list = list()
    y_list = list()
    for i, batch_dict in enumerate(data_loader):
        if (args.max_iters > 0) and (i >= args.max_iters):
            break

        x = batch_dict['image'].cuda()
        x.requires_grad_(False)

        z = ae.forward(x, forward_type='encoder')
        z = z.squeeze()
        z = z.cpu().detach().numpy()
        z_list.append(z)

        if 'label' in batch_dict.keys():
            y = batch_dict['label'].detach().numpy()
            y_list.append(y)

    z = np.concatenate(z_list, axis=0)
    if len(y_list) > 0:
        y = np.concatenate(y_list, axis=0)#.astype(np.uint8)
    else:
        y = None

    if args.dataset == 'mnist':
        colors = MNIST_COLOR
    elif args.dataset == 'two':
        colors = TWO_COLOR
    else:
        colors = {'gray'}

    print('start pca')
    draw_scatter_plot(z, y, save_dir=args.result_dir, projection='PCA', colors=colors)
    print('save pca result\n')

    # half = int(z.shape[0] / 2)
    # print('start tsne')
    # draw_scatter_plot(z[:half], y[:half], save_dir=args.result_dir, projection='TSNE', colors=colors)
    # print('save tsne result')

def draw_scatter_plot(data, label, save_dir, projection='PCA', colors={'gray'}):
    if data.shape[1] == 2:
        projection = 'none'
    elif projection == 'PCA':
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
    elif projection == 'TSNE':
        tsne = TSNE(learning_rate=100)
        data = tsne.fit_transform(data)
    else:
        assert projection in ('PCA', 'TSNE')
    label = np.expand_dims(label, axis=1)
    data = np.concatenate((label, data), axis=1)

    # category_scatter(x=1, y=2, label_col=0, data=data, markers='o', markersize=3, colors=colors, legend_loc='upper right')
    category_scatter(x=1, y=2, label_col=0, data=data, markers='o', markersize=3, colors=colors, legend_loc='upper right')
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    plt.savefig(os.path.join(save_dir, 'scatter-plot_%s.png') % projection)
    # plt.show()

if __name__ == "__main__":
    args = option.plot_parser.parse_args()
    main()
