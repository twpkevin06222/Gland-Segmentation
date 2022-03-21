import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import segmentation


def plot_comparison(input_img, caption=None, plot=True, save_path=None, save_name=None, save_as='png',
                    save_dpi=300, captions_font = 20, n_row=1, n_col=2,
                    figsize=(5, 5), cmap='gray'):
    '''
    Plot comparison of multiple image but only in column wise!
    :param input_img: Input image list
    :param caption: Input caption list
    :param save_path: Path to save plot
    :param save_name: Name to be save for plot
    :param: save_as: plot save extension, 'png' by DEFAULT
    :param n_row: Number of row is 1 by DEFAULT
    :param n_col: Number of columns
    :param figsize: Figure size during plotting (5,5) by DEFAULT
    :return: Plot of (n_row, n_col)
    '''
    print()
    if caption is not None:
        assert len(caption) == len(input_img), "Caption length and input image length does not match"
    assert len(input_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        axes[i].imshow(np.squeeze(input_img[i]), cmap=cmap)
        if caption is not None:
            axes[i].set_xlabel(caption[i], fontsize=captions_font)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'{}.{}'.format(save_name, save_as), save_dpi=save_dpi)
    if plot:
        plt.show()
    else:
        return fig


def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def overlay_mask(image, mask, colors=[(0, 1.0, 0)],
                 alpha=0.12):
    """
    Helper function to plot overlay image segmentation of the original image
    @param image: input image
    @param mask: segmentation mask
    @param colors: color spectrum of type (R,G,B)
    @param alpha: masking opacity
    @return: overlay image
    """
    # normalize image
    if np.max(image)>1.0:
        image = image/255.0
    # gray scale image
    if mask.ndim is 3:
        mask = mask[:,:,0]
    mask_image = color.label2rgb(mask, image,
                             colors=colors, alpha=alpha,
                             bg_label = 0)
    return mask_image


def overlay_boundary(image, mask, color=(0, 1.0, 0),
                     mode='thick'):
    """
    Helper function to plot overlay image segmentation boundary of the
    original image
    @param image: input image
    @param mask: segmentation mask
    @param color: color spectrum of type (R,G,B)
    @param mode: mode of the boundary line
    @return: overlay image with segmented contour
    """
    # normalize image
    if np.max(image)>1.0:
        image = image/255.0
    if mask.ndim is 3:
        mask = mask[:,:,0]
    boundary_image = segmentation.mark_boundaries(image, mask,
                                      color = color, mode=mode)
    return boundary_image


def plot_labels_color(label_im, cmap='tab20c'):
    """
    Helper function to visualize the gland level masking
    by looping through the color map (cmap) defined in matplotlib
    @param label_im: annotated image
    @param cmap: cmap defined in the documentation of matplotlib
    @return: gland level segmentation with color based on cmap
    """
    # Construct a colour image to superimpose
    color_mask = np.zeros(label_im.shape)
    get_cmap = plt.cm.get_cmap(cmap)
    # Loop through the cmap for each color with it's associated labels
    for i in range(np.max(label_im)):
        color_mask[label_im[:, :, 0] == i + 1] = list(get_cmap(i))[:-1]

    return color_mask


def min_max_norm(img, axis=(1, 2)):
    """
    Channel-wise Min max normalization for
    images with input [batch size, slices, width, channel]
    @param img: Input image of 4D array
    @return: Min max norm of the image per channel
    """
    inp_shape = img.shape
    img_min = np.broadcast_to(img.min(axis=axis, keepdims=True), inp_shape)
    img_max = np.broadcast_to(img.max(axis=axis, keepdims=True), inp_shape)
    x = (img-img_min)/(img_max-img_min+float(1e-18))
    return x

