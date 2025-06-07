import os
import numpy as np
from hmmlearn import hmm

# Directory containing processed .npy files
DATA_DIR = "data/processed"

def load_data():
    # Load the scaled features, labels, and subject IDs saved by preprocessing.py
    Xtr  = np.load(os.path.join(DATA_DIR, "X_train.npy"))     # shape = (n_windows, n_features)
    ytr  = np.load(os.path.join(DATA_DIR, "y_train.npy"))     # true activity labels
    strn = np.load(os.path.join(DATA_DIR, "subject_train.npy"))# subject IDs
    Xte  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    yte  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    ste  = np.load(os.path.join(DATA_DIR, "subject_test.npy"))
    return Xtr, ytr, strn, Xte, yte, ste

# Group data by subject
def group_by_subject(X, y, subjects):
    subject_sequences = {}
    for subj_id in np.unique(subjects):
        idx = np.where(subjects == subj_id)[0]
        subject_sequences[subj_id] = (X[idx], y[idx])
    return subject_sequences

# Main HMM training and demo
def main():
    print("Loading data...")
    Xtr, ytr, strn, Xte, yte, ste = load_data()
    sequences = group_by_subject(Xtr, ytr, strn)
    print(f"Number of subjects: {len(sequences)}")

    # For demonstration, use data from one subject
    example_subj = list(sequences.keys())[0]
    X_seq, y_seq = sequences[example_subj]
    print(f"Subject {example_subj} sequence length: {len(y_seq)}")

    # Fit a Gaussian HMM (n_components = number of distinct activities)
    n_components = len(np.unique(ytr))
    model = hmm.GaussianHMM(
        n_components    = n_components,
        covariance_type = 'diag',
        n_iter          = 100,
        tol             = 1e-4,
        min_covar       = 1e-2,
        verbose         = True
    )

    # Concatenate all subjects' data and pass sequence lengths
    X_all   = np.vstack([seq for seq, _ in sequences.values()])
    lengths = [seq.shape[0] for seq, _ in sequences.values()]
    model.fit(X_all, lengths=lengths)

    # Predict hidden states for the example subject
    hidden_states = model.predict(X_seq)
    print(f"Predicted hidden states (first 20): {hidden_states[:20]}")
    print(f"True activity labels    (first 20): {y_seq[:20]}")

if __name__ == "__main__":
    main()
