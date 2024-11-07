# import numpy as np
#
# m1 = np.load('./model2_accuracy.npy')
# print(m1.max())

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

#
# data = np.loadtxt(r"data/mitbih_train.csv", dtype=np.float32,
#                   delimiter=',')
# first_class = data[data[:, -1] == 0]
# second_class = data[data[:, -1] == 1]
# third_class = data[data[:, -1] == 2]
# fourth_class = data[data[:, -1] == 3]
# fifth_class = data[data[:, -1] == 4]
#
# first_class = first_class[:, :-1]
# second_class = second_class[:, :-1]
# third_class = third_class[:, :-1]
# fourth_class = fourth_class[:, :-1]
# fifth_class = fifth_class[:, :-1]
#
# first_sample = first_class[np.random.choice(first_class.shape[0], 1)][0]
# second_sample = second_class[np.random.choice(second_class.shape[0], 1)][0]
# third_sample = third_class[np.random.choice(third_class.shape[0], 1)][0]
# fourth_sample = fourth_class[np.random.choice(fourth_class.shape[0], 1)][0]
# fifth_sample = fifth_class[np.random.choice(fifth_class.shape[0], 1)][0]
#
# plt.figure(1, figsize=(8, 12))
# plt.subplot(5, 1, 1)
# plt.plot(first_sample)
# plt.title('First Class: Normal')
#
# plt.subplot(5, 1, 2)
# plt.plot(second_sample)
# plt.title('Second Class: Supraventricular')
#
# plt.subplot(5, 1, 3)
# plt.plot(third_sample)
# plt.title('Third Class: Ventricular')
#
# plt.subplot(5, 1, 4)
# plt.plot(fourth_sample)
# plt.title('Fourth Class: Fusion')
#
# plt.subplot(5, 1, 5)
# plt.plot(fifth_sample)
# plt.title('Fifth Class: Quasi Periodic')
#
# plt.tight_layout()
# plt.show()

model1_accuracy_list = np.load('./model1_accuracy.npy')
model1_loss_list = np.load('./model1_loss.npy')
gt_list_model1 = np.load('./model1_gt.npy')
predicted_list_model1 = np.load('./model1_predicted.npy')

# Model One: Self-Attention Classifier
# Curve Visualization
plt.figure(figsize=(10, 5))
# Loss Curve
plt.subplot(2, 1, 1)
plt.plot(model1_loss_list)
plt.title("Model One: Self-Attention Classifier Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
# Accuracy Curve
plt.subplot(2, 1, 2)
plt.plot(model1_accuracy_list)
plt.title("Model One: Self-Attention Classifier Accuracy Curve")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
# Show
plt.tight_layout()
plt.show()
# Draw Confusion Matrix
conf_matrix = confusion_matrix(gt_list_model1, predicted_list_model1)
plt.figure(figsize=(10, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Model One: Self-Attention Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
