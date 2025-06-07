import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from hmmlearn import hmm
from hmm import load_data, group_by_subject

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def map_states_to_labels(hidden_states, true_labels, n_states):
    """Assign each HMM state to the most frequent true label."""
    state2label = {}
    for s in range(n_states):
        mask = (hidden_states == s)
        if mask.sum() == 0:
            state2label[s] = -1
        else:
            vals, counts = np.unique(true_labels[mask], return_counts=True)
            state2label[s] = vals[np.argmax(counts)]
    return state2label

 def evaluate_model(n_components, sequences_train, sequences_test):
    # Fit model and measure training time
    X_all   = np.vstack([X for X, _ in sequences_train.values()])
    lengths = [X.shape[0] for X, _ in sequences_train.values()]
    model   = hmm.GaussianHMM(n_components=n_components,
                              covariance_type='diag', n_iter=50)
    t0 = time.time()
    model.fit(X_all, lengths=lengths)
    train_time = time.time() - t0

    # Map states->labels using training data
    all_hidden = []
    all_true   = []
    for X, y in sequences_train.values():
        hidden = model.predict(X)
        all_hidden.append(hidden)
        all_true.append(y)
    hidden_train = np.concatenate(all_hidden)
    true_train   = np.concatenate(all_true)
    state2lab    = map_states_to_labels(hidden_train, true_train, n_components)

    # Evaluate on test data
    preds, trues = [], []
    for X, y in sequences_test.values():
        h = model.predict(X)
        p = np.array([state2lab[s] for s in h])
        preds.append(p)
        trues.append(y)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc   = accuracy_score(trues, preds)
    cm    = confusion_matrix(trues, preds, labels=np.unique(trues))
    return acc, cm, train_time

if __name__ == "__main__":
    # Load & split data
    X, y, subjects = load_data()
    seqs = group_by_subject(X, y, subjects)
    # Split train/test by subject ID (odd vs even)
    train_ids = [sid for sid in seqs if sid % 2 == 1]
    test_ids  = [sid for sid in seqs if sid % 2 == 0]
    train_seqs = {sid: seqs[sid] for sid in train_ids}
    test_seqs  = {sid: seqs[sid] for sid in test_ids}

    # Accuracy vs. # Hidden States
    state_list = [4, 6, 8, 10]
    acc_results = {}
    time_results = {}
    cm_best = None
    best_acc = -np.inf
    best_states = None

    for n in state_list:
        acc, cm, ttime = evaluate_model(n, train_seqs, test_seqs)
        acc_results[n] = acc
        time_results[n] = ttime
        print(f"{n} states: accuracy={acc:.3f}, train_time={ttime:.2f}s")
        if acc > best_acc:
            best_acc = acc
            cm_best = cm
            best_states = n

    # Plot Accuracy vs. #States
    plt.figure()
    plt.plot(list(acc_results.keys()), list(acc_results.values()), marker='o')
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Hidden States')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'accuracy_vs_states.png'))
    plt.close()

    # Plot Confusion Matrix for Best Model
    plt.figure(figsize=(6, 6))
    labels_unique = np.unique(y)
    plt.imshow(cm_best, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(range(len(labels_unique)), labels_unique, rotation=45)
    plt.yticks(range(len(labels_unique)), labels_unique)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({best_states} States)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plt.close()

    # Posterior Marginals Over Time (first test subject)
    example_sid = test_ids[0]
    X_seq, y_seq = test_seqs[example_sid]
    model = hmm.GaussianHMM(n_components=best_states, covariance_type='diag', n_iter=50)
    model.fit(np.vstack([X for X, _ in train_seqs.values()]),
              lengths=[X.shape[0] for X, _ in train_seqs.values()])
    post = model.predict_proba(X_seq)
    plt.figure(figsize=(10, 4))
    for s in range(best_states):
        plt.plot(post[:, s], label=f'State {s}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Time Step')
    plt.ylabel('Posterior Probability')
    plt.title(f'Posterior Marginals for Subject {example_sid}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'posterior_marginals.png'))
    plt.close()

    # Accuracy vs. Training-Set Size
    frac_list = [0.5, 0.75, 1.0]
    acc_size = {}
    for frac in frac_list:
        n_train = int(len(train_ids) * frac)
        sample_ids = train_ids[:n_train]
        sub_train = {sid: train_seqs[sid] for sid in sample_ids}
        acc, _, _ = evaluate_model(best_states, sub_train, test_seqs)
        acc_size[frac] = acc
        print(f"Train frac {frac:.2f}: acc={acc:.3f}")

    plt.figure()
    plt.plot(frac_list, list(acc_size.values()), marker='o')
    plt.xlabel('Training Set Fraction')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Training-Set Size')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'accuracy_vs_train_size.png'))
    plt.close()

    # Runtime vs. # Hidden States
    plt.figure()
    plt.bar(list(time_results.keys()), list(time_results.values()))
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs. Hidden States')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'runtime_vs_states.png'))
    plt.close()
