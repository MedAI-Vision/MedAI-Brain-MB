# Artificial Intelligence Analysis on Medulloblastoma

- Data Preprocessing
- Tumor Segmentation
- Feature Extraction, Selection and Classification

## Data Preprocessing for Segmentation and Feature Extraction

#### Data Preprocessing for nnUNet Segmentation

- Convert DICOM to NIFTI
  ```python
  !pip install dicom2nifti
  import dicom2nifti
  dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)
  ```
- Registration (Unify origins, directions, and spacing among different modalities, if needed)
  ```python
  !pip install antspyx
  import ants
  _ = ants.registration(fixed=fix_img, moving=move_img, type_of_transform='SyN')
  ```
- Other Preprocessing Methods (e.g. skull-stripping, intensity normalization, etc.)

#### Data Preprocessing for feature extraction (based on preprocessing for nnUNet)

- Resampling (Resample to the same resolution)
  ```python
  !pip install SimpleITK
  import SimpleITK as sitk
  resample = sitk.ResampleImageFilter()
  resample.SetInterpolator(sitk.sitkLinear)
  resample.SetOutputDirection(image.GetDirection())
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputSpacing([0.429, 0.429, 6.5]) # the most resolution of the dataset
  resample.SetSize([origin_x/0.429*origin_spacing[0], origin_y/0.429*origin_spacing[1], origin_z/6.5*origin_spacing[2]])
  image = resample.Execute(image)
  ```
- N4Bias field correction
  ```python
   import ants
   modality_data = ants.n4_bias_field_correction(modality_data)
  ```
- Skull-stripping (Please refer to [ROBEX](https://www.nitrc.org/projects/robex).)
- Denoise images using non-local means filter
  ```python
  import ants
  modality_data = ants.from_numpy(modality_data)
  modality_data = ants.denoise_image(modality_data, ants.get_mask(modality_data))
  modality_data = modality_data.numpy()
  ```
- Intensity normalization
  ```python
  !pip install intensity-normalization
  from intensity_normalization.normalize.zscore import ZScoreNormalize
  z_norm = ZScoreNormalize()
  modality_data = z_norm(modality_data, brain)
  ```
- Other Preprocessing Methods (e.g. histogram, etc.)

## nnUNet for Medulloblastoma Segmentation

### Quick Start

For inference based on trained model

- Install PyTorch and nnUNet

  ```shell
  pip install torch
  git clone https://github.com/MIC-DKFZ/nnUNet.git
  cd nnUNet && git checkout nnunetv1 && pip install -e .
  ```

  Please notice that we use nnUNet-version-1 instead of current version-2.

- Create and set workdirs for nnUNet

  Create `nnUNet_workdir` and three subfolders, please read the documentation
  of [nnUNetv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) for details. If you only want to test the model, you
  needn't create any files or folds under `nnUNet_raw_data` and `nnUNet_preprocessed`. We provide `RESULTS_FOLDER` which
  contains trained model's checkpoints. In our [colab notebook](https://colab.research.google.com/drive/1FoBDfPAeU_PH22VQyn-vrUED_bHM5Vpc?usp=sharing![image](https://github.com/MedAI-Brain/MB_AI/assets/148333553/2a2fa14f-ab6b-4e82-9c77-8059da06ee77)
), we provide the download link and method in the notebook, you can download `RESULTS_FOLDER` either to your desktop or just to google drive.

  ```shell
  nnUNetData
  	|__  nnUNet_raw_data
  	|  |__  Task505_MedoTumor
  	|    |__  imagesTr
  	|    |__  imagesTs(optional)
  	|    |__  labelsTr
  	|    |__  dataset.json
  	|__  nnUNet_preprocessed
  	|__  RESULTS_FOLDER
  ```

  Set the environment variables for nnUNet

  ```shell
  export nnUNet_raw_data_base="nnUNetData"
  export nnUNet_preprocessed="nnUNetData/nnUNet_preprocessed"
  export RESULTS_FOLDER="nnUNetData/RESULTS_FOLDER"
  ```

- Download the checkpoints

  Please download trained checkpoints via this link. It is a zip file of `RESULTS_FOLDER` , providing 5-fold trained
  checkpoints and plans of model architecture and preprocessing.

- Dataset Conversion

  The modalities of the data should be `T1-enhanced` and `T2`. And the two modalities of one subjects should be aligned,
  with same resolutions, origins and directions. Please refer nnUNetv1 to rename your nifti files. This is an example.

  ```shell
  imagesTsYOURS
  	|__  Native_001_0000.nii.gz (T1-enhanced)
  	|__  Native_001_0001.nii.gz (T2)
  	|__  Native_002_0000.nii.gz
  	|__  ...
  ```

- Predict segmentation masks

  We use the `3d_fullres` architecture of nnUNet. It is a 3d convolution network.

  ```shell
  nnUNet_predict -i [PATH OF TEST DATASET] -o [OUTPUT PATH] -t 505 -m 3d_fullres
  ```

​ The parameter `505` is th ID of our experiment.

### Training and Testing Pipeline

#### Dataset and Experiment preparation

Please refer [nnUNetv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) for details.

#### Training

- Installation and experiment variables are same with **Quick Start**.

- Please
  refer [generate_dataset_json](https://github.com/MIC-DKFZ/nnUNet/blob/nnunetv1/nnunet/dataset_conversion/utils.py) to
  create `dataset.json` for training dataset.

- Start five-fold training (3D model: 3d_fullres; 2D model: 2d)

  ```shell
  CUDA_VISIBLE_DEVICES=0 nnUNet_train [3d_fullres/2d] nnUNetTrainerV2 TaskXXX_name 0 --npz
  CUDA_VISIBLE_DEVICES=1 nnUNet_train [3d_fullres/2d] nnUNetTrainerV2 TaskXXX_name 1 --npz
  CUDA_VISIBLE_DEVICES=2 nnUNet_train [3d_fullres/2d] nnUNetTrainerV2 TaskXXX_name 2 --npz
  CUDA_VISIBLE_DEVICES=3 nnUNet_train [3d_fullres/2d] nnUNetTrainerV2 TaskXXX_name 3 --npz
  CUDA_VISIBLE_DEVICES=4 nnUNet_train [3d_fullres/2d] nnUNetTrainerV2 TaskXXX_name 4 --npz
  ```

- Find best configuration of five-fold cross training (Optional)

  ```shell
  nnUNet_find_best_configuration -m 2d -t XXX –strict
  ```

#### Testing

```shell
nnUNet_predict -i [PATH OF TEST DATASET] -o [OUTPUT PATH] -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m [3d_fullres/2d] -p nnUNetPlansv2.1 -t XXX
```

Please notice that you don't have to make `dataset.json` for testing dataset.

## Feature Extraction, Selection and Classification

#### Feature Extraction
```python
!pip install pyradiomics
import radiomics
feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
feature_vector_intra = feature_extractor.execute(image, intra_mask)
feature_vector_peri = feature_extractor.execute(image, peri_mask)
```

#### Feature Selection
- Random forest
  ```python
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier
  rfc = RandomForestClassifier(n_estimators=1500)
  feature_importance = rfc.fit(data, label).feature_importances_
  sort_ind = np.argsort(feature_importance,)[::-1]
  reduce_col = [data.columns[sort_ind[i]] for i in range(feature_num)]
  data = data.loc[:,reduce_col]
  ```
- AUC score between feature and label
  ```python
  import numpy as np
  from sklearn.metrics import roc_auc_score
  auc_score = []
  for i in range(feature_num):
      auc_score.append(roc_auc_score(label, data.iloc[:,i]))
  # in most cases, the larger the abs(auc_score-0.5), the more important the feature
  sort_ind = ***
  reduce_col = [data.columns[sort_ind[i]] for i in range(feature_num)]
  data = data.loc[:,reduce_col]
  ```
- Other methods like LASSO, etc.
