import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from skimage import exposure
from radiomics import featureextractor
from tqdm import tqdm
import ants
from intensity_normalization.normalize.zscore import ZScoreNormalize

# --------------------------------------------Parameters----------------------------------------------#
TASK_TYPE = "Medo_Subtypes"
MASK_NAME = 'intra'  # mask_type: two choices: intra / peri (Intra-tumor or Peri-tumor)
COHORT = 'domestic_subtypes'
CSV_PATH = '/code/feature/domestic_subtypes.txt'
OUTPUT_PATH = "/data/***/features_all/{}N4B_cliped/{}/".format(COHORT, MASK_NAME)
USE_MODALITY = ['T2', 'T1E']  # modalities: T1 / T1E / T2 / DWI
IF_CORRECT_MASK = True
"""
    DATA_TRANSFORMATION_TYPE:
    1: zero out based on brain mask
    2: zero out based on brain mask, only histogram equalization with mask=brain_mask
    3: zero out based on brain mask, do match_histogram with mask=brain_mask
"""
DATA_TRANSFORMATION_TYPE = "2-3"
IF_CROP = False


def init_extractor():
    """
    initialize the feature extractor
    """
    setting = {}
    setting['sigma'] = [1, 3, 5]
    setting['correctMask'] = IF_CORRECT_MASK
    extractor = featureextractor.RadiomicsFeatureExtractor(**setting)
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()
    extractor.enableImageTypeByName('LoG')
    return extractor


def save_np_to_nifti(np_array, affine, filepath):
    nib.Nifti1Image(np_array, affine).to_filename(filepath)


def data_transformation(modality, mask, dir_path, modl):
    """
    DATA_TRANSFORMATION_PROCESS:
    1. N4Bias_field_correction
    2. Zero out based on Brain Mask
    3. Denoise: Non-Local mean
    3.5: Histogram Equalization or Match Histogram or PASS
    4. z-score normalization
    """

    modality_data, modality_affine = modality.get_fdata(), modality.affine
    mask_data, mask_affine = mask.get_fdata(), mask.affine

    modality_data = modality_data.astype(np.float32)
    modality_data = ants.from_numpy(modality_data)

    # 1. N4Bias_field_correction
    modality_data = ants.n4_bias_field_correction(modality_data)
    modality_data = modality_data.numpy().copy()

    v_min, v_max = np.percentile(modality_data, q=(0.5, 99.5))
    modality_data = exposure.rescale_intensity(
        modality_data,
        in_range=(v_min, v_max),
        out_range=np.float32
    )

    modality_data = (modality_data - modality_data.min()) / (modality_data.max() - modality_data.min() + 1e-15)

    # 2. Brain Mask
    brain_path = dir_path + 'brain_mask.nii.gz'
    brain = nib.load(brain_path).get_fdata()
    modality_data = modality_data * brain.copy()

    # 3. Denoising: Non-Local mean
    modality_data = ants.from_numpy(modality_data)
    modality_data = ants.denoise_image(modality_data, ants.get_mask(modality_data))
    modality_data = modality_data.numpy().copy()

    v_min, v_max = np.percentile(modality_data, q=(0.5, 99.5))
    modality_data = exposure.rescale_intensity(
        modality_data,
        in_range=(v_min, v_max),
        out_range=np.float32
    )
    modality_data = (modality_data - modality_data.min()) / (modality_data.max() - modality_data.min() + 1e-15)

    # Crop images to reduce the size of the data
    if IF_CROP:
        crop_data = brain.get_fdata()
        x_max = max(np.where(crop_data)[0])
        x_min = min(np.where(crop_data)[0])
        y_max = max(np.where(crop_data)[1])
        y_min = min(np.where(crop_data)[1])

        modality_data = modality_data[x_min:x_max, y_min:y_max, :]
        mask_data = mask_data[x_min:x_max, y_min:y_max, :]

    # 3.5 Histogram Equalization or Match Histogram or PASS
    if DATA_TRANSFORMATION_TYPE == "1":
        pass
    elif DATA_TRANSFORMATION_TYPE == "2":
        modality_data = exposure.equalize_hist(modality_data, mask=brain.get_fdata())
    elif DATA_TRANSFORMATION_TYPE == "3":
        reference_path = '/data/***/All_Native_Space/refer_subject/' + modl
        refer_brain = '/data/***/All_Native_Space/refer_subject/brain_mask.nii.gz'
        reference = nib.load(reference_path).get_fdata() * nib.load(refer_brain).get_fdata()
        modality_data = exposure.match_histograms(modality_data, reference)

    # 4. z-score
    z_norm = ZScoreNormalize()
    modality_data = z_norm(modality_data, brain)
    modality_data = (modality_data - modality_data.min()) / (modality_data.max() - modality_data.min() + 1e-15)

    # Save the processed data
    nib.Nifti1Image(modality_data, modality_affine).to_filename("/output/modality.nii")
    nib.Nifti1Image(mask_data, mask_affine).to_filename("/output/tumor_mask.nii")


os.makedirs(OUTPUT_PATH, exist_ok=True)
extractor = init_extractor()
df = pd.read_csv(CSV_PATH, sep='\t')
need_id = [str(a) for a in df['Number'].tolist()]
print("Total number of subjects: {}".format(len(need_id)))

for i in tqdm(range(df.shape[0])[:]):
    try:
        id = str(df.loc[i, 'Number'])  # just for recognizing the subject, not real
        all_feature = {}
        # the path of each modality
        mod_path = {mod_name: df.loc[i, mod_name] for mod_name in USE_MODALITY}
        # the path of mask
        mask_path = df.loc[i, MASK_NAME]
        dir_path = df.loc[i, 'Path']

        # loop for each modality
        for mod in USE_MODALITY:
            modl = mod_path[mod].split('/')[-1]
            modality = nib.load(mod_path[mod])
            mask = nib.load(mask_path)
            # do data transformation
            data_transformation(modality, mask, dir_path, modl)
            # extract feature from nifti file to dictionary
            mod_feature = extractor.execute("/output/modality.nii", "/output/tumor_mask.nii")
            # add modality name to each feature's name
            mod_feature = {mod + "_" + key: mod_feature[key] for key in mod_feature}
            all_feature.update(mod_feature)
        # add id information to dictionary
        all_feature['id'] = id
        # save feature dictionary to pickle file
        with open(os.path.join(OUTPUT_PATH, "{}.txt".format(id.split("_")[0])), "wb") as f:
            pickle.dump(all_feature, f)
        print("{} is successfully saved".format(id))
    except Exception as error:
        print(error)
        continue
