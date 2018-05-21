# USAGE
# python gradient_descent.py

# import the necessary packages
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

def next_training_batch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
    help="size of SGD mini-batches")
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
    cluster_std=1.05, random_state=20)
X = np.c_[np.ones((X.shape[0])), X]

print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))

lossHistory = []


for epoch in np.arange(0, args["epochs"]):
    epochLoss = []
 
    for (batchX, batchY) in next_training_batch(X, y, args["batch_size"]):

        preds = sigmoid_activation(batchX.dot(W))
 

        error = preds - batchY
 

        loss = np.sum(error ** 2)
        epochLoss.append(loss)
 

        gradient = batchX.T.dot(error) / batchX.shape[0]
 

        W += -args["alpha"] * gradient
 

    lossHistory.append(np.average(epochLoss))
    print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

for i in np.random.choice(250, 10):
	activation = sigmoid_activation(X[i].dot(W))

	label = 0 if activation < 0.5 else 1

	# show our output classification
	print("activation={:.4f}; predicted_label={}, true_label={}".format(
		activation, label, y[i]))

Y = (-W[0] - (W[1] * X)) / W[2]
plt.figure()
plt.scatter(X[:, 1], X[:, 2], marker="o", c=y)
plt.plot(X, Y, "r-")

fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
