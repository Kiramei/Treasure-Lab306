import numpy as np

# X_train, y_train = np.load(r'E:\WorkSpace\py\Experiment\py\Last\code\preprocess\train_lbp.npy',
#                            allow_pickle=True).transpose()
# X_test, y_test = np.load(r'E:\WorkSpace\py\Experiment\py\Last\code\preprocess\test_lbp.npy',
#                          allow_pickle=True).transpose()

X = np.load(r'E:\WorkSpace\py\Experiment\py\Last\code\preprocess\train_lbp_1.npy',
                           allow_pickle=True)
Y = np.load(r'E:\WorkSpace\py\Experiment\py\Last\code\preprocess\test_lbp_1.npy',
                         allow_pickle=True)
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)
