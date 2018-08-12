import numpy as np
import cv2
import matplotlib.pyplot as plt


def showImages(images, imagesName, figTitle= None, cols = 4, rows = 5, figsize=(15,10), cmap = None, figName = None):
    """
        Display `images` on a [`cols`, `rows`] subplot grid.
        """
    if len(images) != len(imagesName):
        raise ValueError('Lenth of images and imagesName are not same')
    
    imgLength = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    plt.suptitle(figTitle)
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < imgLength:
            imagePathName = imagesName[index],
            image = images[index]
            ax.imshow(image,cmap=cmap)
            ax.set_title(imagePathName)
            ax.axis('off')

    if figName != None:
        print("save fig name: ", figName)
        saveName = "./output_images/" + figName + ".png"
        fig.savefig(saveName)

    plt.show()


def showSidebySide(original, new, firstTitle = "original", newTitle = "new"):
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
    axes[0].imshow(original)
    axes[0].set_title(firstTitle)
    axes[1].imshow(new)
    axes[1].set_title(newTitle)
    ## save figure
    saveName = "./output_images/" + newTitle + ".png"
    fig.savefig(saveName)
    plt.show()
