
import pickle
import numpy as np
import pandas as pd

train_file = "C:\\Users\\denis\\Desktop\\Retele_Neuronale\\tema2\\extended_mnist_train.pkl"
test_file  = "C:\\Users\\denis\\Desktop\\Retele_Neuronale\\tema2\\extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)
with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)
train_X = np.vstack(train_data).astype(np.float32)    # shape (m, 784)
train_y = np.array(train_labels, dtype=np.int64)      # shape (m,)

test_data = []
test_labels = []
for image, label in test:
    test_data.append(image.flatten())
    test_labels.append(label)
test_X = np.vstack(test_data).astype(np.float32)
test_y = np.array(test_labels, dtype=np.int64)

# normalize to [0,1]
train_X /= 255.0
test_X  /= 255.0

# one-hot helper
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# softmax (stable)
def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# compute loss (cross-entropy)
def cross_entropy(probs, y_onehot):
    m = probs.shape[0]
    # add small eps to avoid log(0)
    logp = np.log(probs + 1e-12)
    return -np.sum(y_onehot * logp) / m

# accuracy
def accuracy(preds, y):
    return np.mean(preds == y)

# model parameters initialization
rng = np.random.RandomState(1)
D = train_X.shape[1]  # 784
C = 10
W = rng.normal(scale=0.01, size=(D, C)).astype(np.float32)  # shape (784,10)
b = np.zeros(C, dtype=np.float32)

# training hyperparams
epochs = 80
batch_size = 128
lr = 0.18
reg = 1e-4  #L2 regularization

m = train_X.shape[0]

# training loop (minibatch SGD)
for epoch in range(1, epochs + 1):
    # shuffle
    perm = rng.permutation(m)
    X_shuf = train_X[perm]
    y_shuf = train_y[perm]
    epoch_loss = 0.0
    for i in range(0, m, batch_size):
        Xb = X_shuf[i:i+batch_size]
        yb = y_shuf[i:i+batch_size]
        mb = Xb.shape[0]

        # forward
        logits = Xb.dot(W) + b  # (mb, C)
        probs = softmax(logits)  # (mb, C)

        # loss
        yb_onehot = one_hot(yb, C)
        loss = cross_entropy(probs, yb_onehot) + 0.5 * reg * np.sum(W * W)
        epoch_loss += loss * mb

        # backward (gradients)
        # gradient of cross-entropy with softmax: probs - y_onehot
        dlogits = (probs - yb_onehot) / mb  # (mb, C)
        dW = Xb.T.dot(dlogits) + reg * W   # (D, C)
        db = np.sum(dlogits, axis=0)        # (C,)

        # parameter update
        W -= lr * dW
        b -= lr * db

    epoch_loss /= m

    # evaluate on training set for diagnostics (can be done on a subset to save time)
    train_logits = train_X.dot(W) + b
    train_preds = np.argmax(softmax(train_logits), axis=1)
    train_acc = accuracy(train_preds, train_y)

    print(f"Epoch {epoch:02d} - loss: {epoch_loss:.4f} - train acc: {train_acc:.4f}")

# final evaluation on test set
test_logits = test_X.dot(W) + b
test_preds = np.argmax(softmax(test_logits), axis=1)
test_acc = accuracy(test_preds, test_y)
print(f"Test accuracy: {test_acc:.4f}")

# write predictions to CSV
predictions = test_preds  # uses test_preds computed earlier in this file
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(int(label))

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(predictions), "rows")