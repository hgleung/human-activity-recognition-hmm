import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from hmmlearn import hmm
from hmm import load_data, group_by_subject

# Ensure figure directory exists
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
    """Fit an HMM and evaluate accuracy and training time."""
    # Prepare training data
    X_all   = np.vstack([X for X, _ in sequences_train.values()])
    lengths = [X.shape[0] for X, _ in sequences_train.values()]
    model   = hmm.GaussianHMM(
        n_components    = n_components,
        covariance_type = 'diag',
        n_iter          = 50,
        tol             = 1e-4,
        min_covar       = 1e-2,
        verbose         = False
    )

    # Train and time it
    t0 = time.time()
    model.fit(X_all, lengths=lengths)
    train_time = time.time() - t0

    # Map states → labels on training set
    all_hidden = []
    all_true   = []
    for X, y in sequences_train.values():
        hidden = model.predict(X)
        all_hidden.append(hidden)
        all_true.append(y)
    hidden_train = np.concatenate(all_hidden)
    true_train   = np.concatenate(all_true)
    state2lab    = map_states_to_labels(hidden_train, true_train, n_components)

    # Evaluate on train set
    preds_train = []
    for X, y in sequences_train.values():
        h = model.predict(X)
        p = np.array([state2lab[s] for s in h])
        preds_train.append(p)
    preds_train = np.concatenate(preds_train)
    train_acc = accuracy_score(true_train, preds_train)

    # Evaluate on test set
    preds, trues = [], []
    for X, y in sequences_test.values():
        h = model.predict(X)
        p = np.array([state2lab[s] for s in h])
        preds.append(p)
        trues.append(y)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = accuracy_score(trues, preds)
    cm  = confusion_matrix(trues, preds, labels=np.unique(trues))
    return train_acc, acc, cm, train_time

def plot_accuracy_vs_states(results):
    """Line plot of accuracy vs. number of hidden states."""
    states = sorted(results.keys())
    accs   = [results[s] for s in states]
    plt.figure()
    plt.plot(states, accs, marker='o')
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Hidden States')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'accuracy_vs_states.png'))
    plt.close()

def plot_confusion(cm, labels):
    """Heatmap of confusion matrix."""
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    # Load preprocessed data and group by subject
    X_tr, y_tr, s_tr, X_te, y_te, s_te = load_data()
    train_seqs = group_by_subject(X_tr, y_tr, s_tr)
    test_seqs  = group_by_subject(X_te, y_te, s_te)

    # Sweep over different numbers of hidden states
    state_list   = [n*2 for n in range(2, 11)]
    train_acc_results = {}
    test_acc_results  = {}
    time_results = {}
    best_acc     = -np.inf
    best_cm      = None
    best_states  = None

    for n in state_list:
        train_acc, test_acc, cm, ttime = evaluate_model(n, train_seqs, test_seqs)
        train_acc_results[n] = train_acc
        test_acc_results[n]  = test_acc
        time_results[n] = ttime
        print(f"{n} states: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, time={ttime:.2f}s")
        if test_acc > best_acc:
            best_acc    = test_acc
            best_cm     = cm
            best_states = n

    # Plot Train vs. Test Accuracy vs. Hidden States
    plt.figure()
    plt.plot(list(train_acc_results.keys()), list(train_acc_results.values()), marker='o', label='Train Accuracy')
    plt.plot(list(test_acc_results.keys()), list(test_acc_results.values()), marker='o', label='Test Accuracy')
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Test Accuracy vs. Hidden States')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'train_vs_test_accuracy_vs_states.png'))
    plt.close()

    # Plot Accuracy vs. Hidden States (test only for backward compatibility)
    plot_accuracy_vs_states(test_acc_results)

    # Plot Confusion Matrix for best model
    plot_confusion(best_cm, labels=np.unique(y_tr))

    # Posterior marginals over time for one test subject
    example_sid = list(test_seqs.keys())[0]
    X_seq, _   = test_seqs[example_sid]
    model      = hmm.GaussianHMM(
        n_components    = best_states,
        covariance_type = 'diag',
        n_iter          = 50,
        tol             = 1e-4,
        min_covar       = 1e-2
    )
    # Re-train on full training set
    X_all    = np.vstack([X for X, _ in train_seqs.values()])
    lengths  = [X.shape[0] for X, _ in train_seqs.values()]
    model.fit(X_all, lengths=lengths)

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

    # Accuracy vs. training-set size
    frac_list = [0.5, 0.75, 1.0]
    acc_size  = {}
    train_ids = list(train_seqs.keys())
    for frac in frac_list:
        n_train    = int(len(train_ids) * frac)
        subset_ids = train_ids[:n_train]
        sub_train  = {sid: train_seqs[sid] for sid in subset_ids}
        acc_sub, _, _ = evaluate_model(best_states, sub_train, test_seqs)
        acc_size[frac] = acc_sub
        print(f"Train frac {frac:.2f}: acc={acc_sub:.3f}")

    plt.figure()
    plt.plot(list(acc_size.keys()), list(acc_size.values()), marker='o')
    plt.xlabel('Training Set Fraction')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Training-Set Size')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'accuracy_vs_train_size.png'))
    plt.close()

    # Runtime vs. hidden states
    plt.figure()
    plt.bar(list(time_results.keys()), list(time_results.values()))
    plt.xlabel('Number of Hidden States')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs. Hidden States')
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, 'runtime_vs_states.png'))
    plt.close()
