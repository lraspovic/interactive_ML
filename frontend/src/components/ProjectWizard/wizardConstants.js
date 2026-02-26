// ─── Sensor options ──────────────────────────────────────────────────────────
export const SENSOR_OPTIONS = [
  { id: 'sentinel-2-l2a', label: 'Sentinel-2 L2A' },
  { id: 'sentinel-1-rtc', label: 'Sentinel-1 RTC' },
]

// Logical band names available per collection.
// These match COLLECTION_BAND_TO_ASSET keys in the backend spectral_catalogue.
export const SENSOR_BANDS = {
  'sentinel-2-l2a': [
    { id: 'blue',       label: 'Blue (B02)' },
    { id: 'green',      label: 'Green (B03)' },
    { id: 'red',        label: 'Red (B04)' },
    { id: 'red_edge_1', label: 'Red Edge 1 (B05)' },
    { id: 'red_edge_2', label: 'Red Edge 2 (B06)' },
    { id: 'red_edge_3', label: 'Red Edge 3 (B07)' },
    { id: 'nir',        label: 'NIR (B08)' },
    { id: 'nir_2',      label: 'NIR narrow (B8A)' },
    { id: 'swir1',      label: 'SWIR-1 (B11)' },
    { id: 'swir2',      label: 'SWIR-2 (B12)' },
  ],
  'sentinel-1-rtc': [
    { id: 'sar_vv', label: 'SAR VV' },
    { id: 'sar_vh', label: 'SAR VH' },
  ],
}

// Default RGB-only sensor config (matches wizard default)
export const DEFAULT_SENSORS = [
  { sensor_type: 'sentinel-2-l2a', bands: ['blue', 'green', 'red'] },
]

// GLCM texture statistics available for selection
export const GLCM_STATISTICS = [
  'contrast',
  'dissimilarity',
  'homogeneity',
  'energy',
  'correlation',
  'ASM',
]

export const GLCM_WINDOW_SIZES = [3, 5, 7, 9]

// ─── Model config ────────────────────────────────────────────────────────────
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
