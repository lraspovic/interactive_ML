export const BAND_OPTIONS = [
  { id: 'blue',     label: 'Blue' },
  { id: 'green',    label: 'Green' },
  { id: 'red',      label: 'Red' },
  { id: 'nir',      label: 'NIR' },
  { id: 'red_edge', label: 'Red Edge' },
  { id: 'swir1',    label: 'SWIR-1' },
  { id: 'swir2',    label: 'SWIR-2' },
  { id: 'sar_vv',   label: 'SAR VV' },
  { id: 'sar_vh',   label: 'SAR VH' },
]

export const SPECTRAL_INDICES = [
  { id: 'ndvi', label: 'NDVI',  description: 'Vegetation',    required: ['nir', 'red'] },
  { id: 'ndwi', label: 'NDWI',  description: 'Water',         required: ['green', 'nir'] },
  { id: 'ndbi', label: 'NDBI',  description: 'Built-up',      required: ['swir1', 'nir'] },
  { id: 'bsi',  label: 'BSI',   description: 'Bare Soil',     required: ['swir1', 'red', 'nir', 'blue'] },
  { id: 'evi',  label: 'EVI',   description: 'Enhanced Veg.', required: ['nir', 'red', 'blue'] },
]

export const MODEL_FAMILIES = {
  classical: [
    { id: 'random_forest',      label: 'Random Forest' },
    { id: 'xgboost',            label: 'XGBoost' },
    { id: 'svm',                label: 'SVM' },
    { id: 'gradient_boosting',  label: 'Gradient Boosting' },
  ],
  deep_learning: [
    { id: 'unet',         label: 'U-Net' },
    { id: 'unet_resnet',  label: 'U-Net + ResNet' },
  ],
}

export const HYPERPARAMS = {
  random_forest: [
    { id: 'n_estimators', label: 'Trees',     type: 'number', default: 100 },
    { id: 'max_depth',    label: 'Max Depth', type: 'number', default: '', placeholder: 'unlimited' },
  ],
  xgboost: [
    { id: 'n_estimators',  label: 'Estimators',    type: 'number', default: 100 },
    { id: 'learning_rate', label: 'Learning Rate',  type: 'number', default: 0.1 },
    { id: 'max_depth',     label: 'Max Depth',      type: 'number', default: 6 },
  ],
  svm: [
    { id: 'kernel', label: 'Kernel', type: 'select', options: ['rbf', 'linear', 'poly'], default: 'rbf' },
    { id: 'C',      label: 'C',      type: 'number', default: 1.0 },
  ],
  gradient_boosting: [
    { id: 'n_estimators',  label: 'Estimators',    type: 'number', default: 100 },
    { id: 'learning_rate', label: 'Learning Rate',  type: 'number', default: 0.1 },
    { id: 'max_depth',     label: 'Max Depth',      type: 'number', default: 3 },
  ],
  unet: [
    { id: 'epochs',      label: 'Epochs',      type: 'number', default: 50 },
    { id: 'batch_size',  label: 'Batch Size',  type: 'number', default: 8 },
    { id: 'encoder',     label: 'Encoder',     type: 'select', options: ['resnet34', 'resnet50', 'efficientnet-b0'], default: 'resnet34' },
  ],
  unet_resnet: [
    { id: 'epochs',      label: 'Epochs',      type: 'number', default: 50 },
    { id: 'batch_size',  label: 'Batch Size',  type: 'number', default: 8 },
    { id: 'encoder',     label: 'Encoder',     type: 'select', options: ['resnet50', 'resnet101'], default: 'resnet50' },
  ],
}

export const DEFAULT_CLASSES = [
  { name: 'Forest',     color: '#2d6a4f' },
  { name: 'Grassland',  color: '#95d5b2' },
  { name: 'Cropland',   color: '#e9c46a' },
  { name: 'Urban',      color: '#adb5bd' },
  { name: 'Water',      color: '#4895ef' },
  { name: 'Bare Soil',  color: '#bc6c25' },
]
