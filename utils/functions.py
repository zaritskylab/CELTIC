import matplotlib.pyplot as plt

def get_cell_stages():
    return ['M0','M1M2','M3','M4M5','M6M7_complete','M6M7_single']

def show_images_subplots(shape, images, titles=None, figsize=(20,20), axis_off=False, cmap='viridis', origin='upper', vmin=None, vmax=None, save=None, tight_layout=False):
    
    rows, columns = shape
    
    if type(cmap)==str or cmap is None:
        cmap_list = [cmap]*len(images)
    elif type(cmap)==list and len(cmap)==shape[1]:
        cmap_list = cmap * shape[0]
    elif cmap:
        assert 0, 'wrong cmap param'


    fig = plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, columns, i+1)
        if img is not None:
          plt.imshow(img, cmap=cmap_list[i], origin=origin, vmin=vmin, vmax=vmax)
        if titles:
          plt.title(titles[i])
        if axis_off:
            plt.axis('off')
        plt.grid(0)
    
    if save:
        file_name, dpi = save
        plt.savefig(file_name, dpi=dpi)
        
    if tight_layout:
        plt.tight_layout()
