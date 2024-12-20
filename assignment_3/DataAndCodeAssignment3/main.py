# -*- coding: utf-8 -*-
"""
Run Beethoven reconstruction code 

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 14:11:54 2016
"""

import os
import argparse
import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt
from tqdm import tqdm

image_folder = 'figures'
def run(dataset, mode='woodham', smooth=None, threshold=None, image=False):
    image_append = ""
    
    # read Beethoven data
    I, mask, S = ps_utils.read_data_file(dataset)

    # get indices of non zero pixels in mask
    nz = np.where(mask > 0)
    m, n = mask.shape
    n_images = I.shape[2]

    # for each mask pixel, collect image data
    J = np.zeros((n_images, len(nz[0])))
    for i in range(n_images):
        Ii = I[:, :, i]
        J[i, :] = Ii[nz]

    if threshold is not None:
        image_append += "_t"
        Mi = []
        # Add tqdm progress bar here
        for i in tqdm(range(J.shape[-1]), desc="Running RANSAC", unit="pixel"):
            array, _, _ = ps_utils.ransac_3dvector(data=(J[:, i], S), threshold=threshold)
            Mi.append(array)
        M = np.stack(Mi, axis=0).T
    else:
        # solve for M = rho*N
        if n_images == 3:
            Si = la.inv(S)
        else:
            Si = np.conjugate(S).T
        M = np.dot(Si, J)

    # get albedo as norm of M and normalize M
    indicies = np.where(M == 0)
    epsilon = 1e-5
    M[indicies] = epsilon
    Rho = la.norm(M, axis=0)
    N = M / np.tile(Rho, (3, 1))

    n1 = np.zeros((m, n))
    n2 = np.zeros((m, n))
    n3 = np.ones((m, n))
    n1[nz] = N[0, :]
    n2[nz] = N[1, :]
    n3[nz] = N[2, :]

    if smooth is not None:
        image_append += "_s"
        n1, n2, n3 = ps_utils.smooth_normal_field(n1=n1, n2=n2, n3=n3, mask=mask, iters=smooth)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(n1)
    ax2.imshow(n2)
    ax3.imshow(n3)
    if image:
        image_name = "{}_normals".format(dataset)
        plt.savefig(os.path.join(image_folder, image_name + image_append + ".png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    z = np.nan_to_num(z)

    ps_utils.display_surface(z, display_angles=(60,0))
    if image:
        image_name = "{}_surface1".format(dataset)
        plt.savefig(os.path.join(image_folder, image_name + image_append + ".png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    ps_utils.display_surface(z, display_angles=(0,90))
    if image:
        image_name = "{}_surface2".format(dataset)
        plt.savefig(os.path.join(image_folder, image_name + image_append + ".png"), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Run assignment 3 for a given dataset"
    )

    parser.add_argument(
        "Dataset",
        type=str,
        help="Mandatory argument."
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        help="Threshold value for ransac, runs woodham if not specified"
    )

    parser.add_argument(
        "-s", "--smooth",
        type=int,
        help="Iterations to run smooth normal field for, does not smooth if not specified"
    )

    parser.add_argument(
        "-i", "--image",
        action="store_true",
        help="Runs script in non-interactive mode, saves plots to image files instead."
    )

    args = parser.parse_args()

    run(args.Dataset, threshold=args.threshold, smooth=args.smooth, image=args.image)

if __name__ == "__main__":
    main()
