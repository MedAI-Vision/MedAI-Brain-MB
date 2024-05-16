import ants


def ReRegis(fixed, moving, outpath):
    """
    Perform image registration between a fixed and a moving image using ANTsPy.
    This function aligns the moving image to the fixed image using the SyN algorithm.

    Parameters:
    - fixed (str): File path to the fixed image.
    - moving (str): File path to the moving image.
    - outpath (str): File path where the transformed moving image will be saved.

    Returns:
    - list: The forward transformations from the fixed image to the aligned moving image.
    """
    fix_img = ants.image_read(fixed)
    move_img = ants.image_read(moving)
    mytx = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='SyN')
    ants.image_write(mytx['warpedmovout'], outpath)
    return mytx['fwdtransforms']


def ApplyRegis(fixed, moving, outpath, transformlist):
    """
    Apply a series of transforms to a moving image to align it with a fixed image and save the result.
    This function uses a list of transforms (from function ReRegis) to apply to the moving image so that it aligns with
    the fixed image, often used after an image registration process. The result is saved to the given path.

    Parameters:
    - fixed (str): File path to the fixed image.
    - moving (str): File path to the moving image.
    - outpath (str): File path where the transformed and aligned moving image will be saved.
    - transformlist (list): A list of transforms that will be applied to the moving image.

    The function uses 'nearestNeighbor' interpolation, which is often suitable for labeled or segmented images.
    """
    fix_img = ants.image_read(fixed)
    move_img = ants.image_read(moving)
    reg_label_img = ants.apply_transforms(fix_img, move_img, transformlist=transformlist,
                                          interpolator='nearestNeighbor')
    ants.image_write(reg_label_img, outpath)


if __name__ == '__main__':
    fixed = './T2_Ax.nii.gz'
    moving = './T1_E_Ax.nii.gz'
    outpath = './T1_E_Ax_reslice.nii.gz'
    tflist = ReRegis(fixed, moving, outpath)

    moving_another = './T1_Ax.nii.gz'
    outpath_another = './T1_Ax_reslice.nii.gz'
    ApplyRegis(fixed, moving_another, outpath_another, tflist)