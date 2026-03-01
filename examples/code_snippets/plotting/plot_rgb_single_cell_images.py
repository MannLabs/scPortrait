"""
Generate RGB composite single-cell images
==========================================
"""

import matplotlib.pyplot as plt
from scportrait.pl import generate_composite

from scportrait.data._single_cell_images import dataset2_h5sc

# select images you want to plot and colorize
h5sc = dataset2_h5sc()
images = h5sc.obsm["single_cell_images"][:, 2:5][[0, 2, 5]]

for _i, img in enumerate(images):
    plt.figure()
    plt.imshow(generate_composite(img))
    plt.axis("off")
