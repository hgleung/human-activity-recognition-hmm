import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Directories
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')
TRAIN_DIR    = os.path.join(DATA_DIR, 'train')
TEST_DIR     = os.path.join(DATA_DIR, 'test')
PROCESSED_DIR= os.path.join(DATA_DIR, 'processed')

# Ensure processed directory is clean
if os.path.exists(PROCESSED_DIR):
    import shutil
    shutil.rmtree(PROCESSED_DIR)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Raw file paths
X_TRAIN_FILE   = os.path.join(TRAIN_DIR, 'X_train.txt')
y_TRAIN_FILE   = os.path.join(TRAIN_DIR, 'y_train.txt')
SUB_TRAIN_FILE = os.path.join(TRAIN_DIR, 'subject_train.txt')

X_TEST_FILE    = os.path.join(TEST_DIR, 'X_test.txt')
y_TEST_FILE    = os.path.join(TEST_DIR, 'y_test.txt')
SUB_TEST_FILE  = os.path.join(TEST_DIR, 'subject_test.txt')

# Output paths
X_TRAIN_PROC   = os.path.join(PROCESSED_DIR, 'X_train.npy')
Y_TRAIN_PROC   = os.path.join(PROCESSED_DIR, 'y_train.npy')
SUB_TRAIN_PROC = os.path.join(PROCESSED_DIR, 'subject_train.npy')
X_TEST_PROC    = os.path.join(PROCESSED_DIR, 'X_test.npy')
Y_TEST_PROC    = os.path.join(PROCESSED_DIR, 'y_test.npy')
SUB_TEST_PROC  = os.path.join(PROCESSED_DIR, 'subject_test.npy')

# Load raw summary features and labels
print("Loading raw features and labels...")
X_train = np.loadtxt(X_TRAIN_FILE)
y_train = np.loadtxt(y_TRAIN_FILE, dtype=int)
subjects_train = np.loadtxt(SUB_TRAIN_FILE, dtype=int)
X_test  = np.loadtxt(X_TEST_FILE)
y_test  = np.loadtxt(y_TEST_FILE, dtype=int)
subjects_test  = np.loadtxt(SUB_TEST_FILE, dtype=int)

# Scale features
print("Scaling features with StandardScaler...")
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Apply PCA to reduce dimensionality
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95, svd_solver='full')  # Retain 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"Reduced dimensions: {X_train_pca.shape[1]}")

# Save processed arrays
print(f"Saving processed data to {PROCESSED_DIR}...")
np.save(X_TRAIN_PROC, X_train_pca)
np.save(Y_TRAIN_PROC, y_train)
np.save(SUB_TRAIN_PROC, subjects_train)
np.save(X_TEST_PROC, X_test_pca)
np.save(Y_TEST_PROC, y_test)
np.save(SUB_TEST_PROC, subjects_test)

print("Preprocessing complete.")
