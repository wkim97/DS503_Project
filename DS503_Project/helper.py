from model import Model
import torch
import os
import matplotlib.pyplot as plt

def show_images(images):
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap = 'gray')
                idx += 1
    plt.show()

def recursive_interpolation(complete_model : Model, uncomplete_model : Model, num_iter : int = 10, n_samples : int = 9, image_shape = (1, 28, 28)) :
    """
    complete_model --> This model must generate a perfect image.
    uncomplete_molde --> This model must generate an unperfect image. That means the model do not convert the input to pure noise.
    """
    imgs = complete_model.p_sample(n_samples = n_samples)
    show_images(imgs)
    num = 0
    print('\nImage generation in progressing...\n')
    while(num < 1 or num > n_samples) :
        num = int(input('Pick target image. (1 ~ n) \nInput : ')) # {1, 2 , ... , n_samples}
        if num < 1 or num > n_samples :
            print('Wrong Number!')
    num -= 1
    res = imgs[num].reshape(-1, *image_shape)
        
    for i in range(num_iter) :
        print('\nImage interpolation in progressing...\n')
        imgs = complete_model.p_sample(n_samples = n_samples)
        interpolated_imgs = uncomplete_model.multi_interpolate_sample(imgs, res, image_size = image_shape)
        show_images( interpolated_imgs )
        num = 0
        while(num < 1 or num > n_samples) :
            num = int(input('Pick target image. (1 ~ n) \nInput : ')) # {1, 2 , ... , n_samples}
            if num < 1 or num > n_samples :
                print('Wrong Number!')
        num -= 1
        res = torch.cat( (res, imgs[num].reshape(-1, *image_shape ) ) )
    return res