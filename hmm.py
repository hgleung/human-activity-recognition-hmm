import numpy as np
from hmmlearn import hmm

# Paths to data files
data_dir      = "data"
X_path        = f"{data_dir}/train/X_train.txt"
y_path        = f"{data_dir}/train/y_train.txt"
subject_path  = f"{data_dir}/train/subject_train.txt"

# Load data
def load_data():
    # shape = (n_windows, n_features)
    X        = np.loadtxt(X_path)
    # true activity labels
    y        = np.loadtxt(y_path, dtype=int)
    # subject IDs
    subjects = np.loadtxt(subject_path, dtype=int)
    return X, y, subjects

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
    X, y, subjects = load_data()
    sequences      = group_by_subject(X, y, subjects)
    print(f"Number of subjects: {len(sequences)}")

    # For demonstration, use data from one subject
    example_subj = list(sequences.keys())[0]
    X_seq, y_seq = sequences[example_subj]
    print(f"Subject {example_subj} sequence length: {len(y_seq)}")

    # Fit a Gaussian HMM (n_components = number of distinct activities)
    n_components = len(np.unique(y))
    model = hmm.GaussianHMM(
        n_components     = n_components,
        covariance_type  = 'diag',
        n_iter           = 50,
        verbose          = True
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
