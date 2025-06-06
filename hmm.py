import numpy as np
from hmmlearn import hmm

# Paths to data files
data_dir = "data/train"
x_path = f"{data_dir}/X_train.txt"
y_path = f"{data_dir}/y_train.txt"
subject_path = f"{data_dir}/subject_train.txt"

# Load data
def load_data():
    X = np.loadtxt(x_path)
    y = np.loadtxt(y_path, dtype=int)
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
    sequences = group_by_subject(X, y, subjects)
    print(f"Number of subjects: {len(sequences)}")

    # For demonstration, use data from one subject
    example_subj = list(sequences.keys())[0]
    X_seq, y_seq = sequences[example_subj]
    print(f"Subject {example_subj} sequence length: {len(y_seq)}")

    # Fit a Gaussian HMM (number of states = number of activities)
    n_components = 6
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=10, verbose=True)
    model.fit(X_seq)

    # Predict hidden states for the sequence
    hidden_states = model.predict(X_seq)
    print(f"Predicted hidden states (first 20): {hidden_states[:20]}")
    print(f"True activity labels (first 20): {y_seq[:20]}")

if __name__ == "__main__":
    main()
