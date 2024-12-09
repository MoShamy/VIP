# -*- coding: utf-8 -*-
"""
Run Beethoven reconstruction code 

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 14:11:54 2016
"""

import argparse
import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt


def run(dataset, mode='woodham', smooth=None, threshold=None):
    # read Beethoven data
    I, mask, S = ps_utils.read_data_file(dataset)

    # get indices of non zero pixels in mask
    nz = np.where(mask > 0)
    m,n = mask.shape
    print(S.shape)
    n_images = I.shape[2]

    # for each mask pixel, collect image data
    J = np.zeros((n_images, len(nz[0])))
    for i in range(n_images):
        Ii = I[:,:,i]
        J[i,:] = Ii[nz]
    print(J.shape)
    if threshold is not None:
        M = ps_utils.ransac_3dvector(data=(J,S), threshold=threshold)
    else:
        # solve for M = rho*N
        S_dagger = np.conjugate(S).T
        M = np.dot(S_dagger, J)
    # get albedo as norm of M and normalize M
    Rho = la.norm(M, axis=0)
    N = M/np.tile(Rho, (3,1))

    n1 = np.zeros((m,n))
    n2 = np.zeros((m,n))
    n3 = np.ones((m,n))
    n1[nz] = N[0,:]
    n2[nz] = N[1,:]
    n3[nz] = N[2,:]

    if smooth is not None:
        n1, n2, n3 = ps_utils.smooth_normal_field(n1=n1, n2=n2, n3=n3, mask=mask, iters=smooth)
    _,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(n1)
    ax2.imshow(n2)
    ax3.imshow(n3)
    plt.show()

    z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
    z = np.nan_to_num(z)

    ps_utils.display_surface(z)

def main():
    parser = argparse.ArgumentParser(
        description = "Run assignment 3 for a given dataset"
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

    args = parser.parse_args()

    run(args.Dataset, threshold=args.threshold, smooth=args.smooth)

if __name__ == "__main__":
    main()

