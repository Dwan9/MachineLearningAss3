# import the necessary packages
import matplotlib.pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import argparse

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-0.5*x))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.0001,
	help="learning rate")
args = vars(ap.parse_args())

# generate a 2-class classification problem with 250 data points,
# where each data point is a 2D feature vector
#(X, y) = make_blobs(n_samples=250, n_features=2, centers=2,
#	cluster_std=1.05, random_state=20)
#cancer data set
cancer = load_breast_cancer()
X, X_test, y, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# insert a column of 1's as the first entry in the feature
# vector -- this is a little trick that allows us to treat
# the bias as a trainable parameter *within* the weight matrix
# rather than an entirely separate variable
X = np.c_[np.ones((X.shape[0])), X]
X_test = np.c_[np.ones((X_test.shape[0])), X_test]

# initialize our weight matrix such it has the same number of
# columns as our input features
print("[INFO] starting training...")
W = np.random.uniform(size=(X.shape[1],))
W[0] = -0.5

# initialize a list to store the loss value for each epoch
lossHistory = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	# take the dot product between our features `X` and the
	# weight matrix `W`, then pass this value through the
	# sigmoid activation function, thereby giving us our
	# predictions on the dataset
	preds = sigmoid_activation(X.dot(W))

	# now that we have our predictions, we need to determine
	# our `error`, which is the difference between our predictions
	# and the true values
	error = preds - y

	# given our `error`, we can compute the total loss value as
	# the sum of squared loss -- ideally, our loss should
	# decrease as we continue training
	loss = np.sum(error ** 2)
	lossHistory.append(loss)
	print("[INFO] epoch #{}, loss={:.7f}".format(epoch + 1, loss))

	# the gradient update is therefore the dot product between
	# the transpose of `X` and our error, scaled by the total
	# number of data points in `X`
	gradient = X.T.dot(error) / X.shape[0]

	# in the update stage, all we need to do is nudge our weight
	# matrix in the opposite direction of the gradient (hence the
	# term "gradient descent" by taking a small step towards a
	# set of "more optimal" parameters
	#print (W)
	W += -args["alpha"] * gradient
	#print (W)

# to demonstrate how to use our weight matrix as a classifier,
# let's look over our a sample of training examples
#for i in np.random.choice(140, 10):
	# compute the prediction by taking the dot product of the
	# current feature vector with the weight matrix W, then
	# passing it through the sigmoid activation function
#	activation = sigmoid_activation(X_test[i].dot(W))

	# the sigmoid function is defined over the range y=[0, 1],
	# so we can use 0.5 as our threshold -- if `activation` is
	# below 0.5, it's class `0`; otherwise it's class `1`
#	label = 0 if activation < 0.5 else 1

	# show our output classification
#	print("activation={:.4f}; predicted_label={}, true_label={}".format(
#		activation, label, y_test[i]))

#confusion matrix
TP=0
FN=0
FP=0
TN=0
#for idx in np.arange(0, args["epochs"]):
for idx in np.arange(0, 140):
        activation = sigmoid_activation(X_test[idx].dot(W))
        label = 0 if activation < 0.5 else 1
        if label == 0 & y_test[idx] == 0:
                TN = TN+1
        elif label == 0 & y_test[idx] == 1:
                FN = FN+1
        elif label == 1 & y_test[idx] == 1:
                TP = TP+1
        else:
                FP = FP+1
con_matrix = np.matrix([[TP, FN], [FP, TN]])
print("confusion matrix")
print(con_matrix)
print("Accuracy: ", (TP+TN)/(TP+TN+FP+FN))
print("Recall: ", (TP/(TP+FN)))
print("Precision: ", (TP/(TP+FP)))

# compute the line of best fit by setting the sigmoid function
# to 0 and solving for X2 in terms of X1
print(W)
Y = (-W[0] - (W[27] * X_test)-(W[24] * X_test)-(W[28] * X_test))/ W[11]

# plot the original data along with our line of best fit
plt.figure()
plt.scatter(X_test[:, 27], X_test[:, 11], marker="o", c=y_test)
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.plot(X_test, Y, "r-")

# construct a figure that plots the loss over time
fig = plt.figure()
plt.plot(np.arange(0, args["epochs"]), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
