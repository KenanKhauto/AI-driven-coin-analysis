import matplotlib.pyplot as plt
import numpy as np

def show_paired_images(pairs, rows, figsize=(10, 5)):
    """
    Display paired images (obverse and reverse) in a single figure with 2 columns.
    :param pairs: List of dictionaries containing paired images as PIL objects.
    :param rows: Number of rows to display.
    """
    cols = 2  # Fixed columns: one for obverse, one for reverse
    total_images = min(rows, len(pairs))  # Ensure we don't exceed available pairs
    
    # Create a grid for the figure
    fig, axes = plt.subplots(total_images, cols, figsize=figsize)
    fig.suptitle("Obverse and Reverse Images", fontsize=16)

    for i, pair in enumerate(pairs[:total_images]):
        # Display Obverse
        obverse_image = pair['obverse']['image']
        axes[i, 0].imshow(obverse_image)
        axes[i, 0].set_title(f"Obverse: {pair['obverse']['filename']}")
        axes[i, 0].axis('off')

        # Display Reverse
        reverse_image = pair['reverse']['image']
        axes[i, 1].imshow(reverse_image)
        axes[i, 1].set_title(f"Reverse: {pair['reverse']['filename']}")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
    
    plt.show()


def preprocess_image(image, size=(224, 224)):
    image = image.resize(size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array