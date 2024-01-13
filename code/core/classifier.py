import joblib
import logging
import numpy as np
from sklearn import svm
from prepare import Config
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# You are asked to use SVM to classify the data of images

# 1. Load the data
# 2. Use PCA to reduce the dimension of the data
# 3. Use SVM to classify the data
# 4. Show the result
class Classifier(Config):
    def __init__(self,
                 train_data=np.load('../preprocess/train.npy', allow_pickle=True),
                 test_data=np.load('../preprocess/test.npy', allow_pickle=True),
                 need_pca=True,
                 pca_n_components=10):
        super().__init__()
        # 1. Load the Parameters
        self.model = None
        self.need_pca = need_pca
        self.test_data = test_data
        self.test_label = None
        self.train_data = train_data
        self.train_label = None
        self.pca_n_components = pca_n_components

    def __prepare__(self):
        # 2. Use PCA to reduce the dimension of the data
        # 2.1 Get the label of the data
        self.train_label = self.train_data[:, 1]
        self.test_label = self.test_data[:, 1]
        # 2.2 Get the data
        self.train_data = self.train_data[:, 0]
        self.test_data = self.test_data[:, 0]
        if len(self.train_data.shape) == 1:
            self.train_data = np.array([np.array(r).astype(np.float32) for r in self.train_data])
            self.test_data = np.array([np.array(r).astype(np.float32) for r in self.test_data])
        # 2.3 logging the shape of the data
        logging.log(logging.INFO, f'Shape of train data: {self.train_data.shape}')
        logging.log(logging.INFO, f'Shape of test data: {self.test_data.shape}')
        self.train_data = np.array([x.flatten() for x in self.train_data])
        self.test_data = np.array([x.flatten() for x in self.test_data])
        # 2.4 Use PCA to reduce the dimension of the data
        if self.need_pca:
            logging.log(logging.INFO, 'Start PCA Decomposition...')
            pca = PCA(n_components=self.pca_n_components)
            pca.fit(self.train_data)
            self.train_data = pca.transform(self.train_data)
            self.test_data = pca.transform(self.test_data)
            # 2.5 Log the shape of the data after PCA
            logging.log(logging.INFO, f'Shape of train data after PCA: {self.train_data.shape}')
            logging.log(logging.INFO, f'Shape of test data after PCA: {self.test_data.shape}')

    def classify(self, kernel='linear'):
        # 3. Use SVM to classify the data
        # 3.1 Create the model
        self.model = svm.SVC(kernel=kernel)
        # 3.2 Train the model
        logging.log(logging.INFO, 'Start training SVM...')
        self.model.fit(self.train_data, self.train_label)
        # Save the model
        joblib.dump(self.model, 'model.pkl')

    def predict(self):
        # 3.3 Predict the result
        result = self.model.predict(self.test_data)
        # 3.4 Show the result
        logging.log(logging.INFO, f'Result: {result}')
        # 3.5 Show the accuracy
        logging.log(logging.INFO, f'Accuracy: {np.sum(result == self.test_label) / len(self.test_label)}')

        # 4. Show the result
        # 4.1 Show the data
        plt.scatter(self.train_data[:, 0], self.train_data[:, 1])
        plt.show()
        # 4.2 Show the result
        plt.scatter(self.test_data[:, 0], self.test_data[:, 1])
        plt.show()

        # 5. Show the result of the model by Sklearn
        # 5.1 Show the support vectors
        logging.log(logging.INFO, f'Support vectors: {self.model.support_vectors_}')
        # 5.2 Show the number of support vectors
        logging.log(logging.INFO, f'Number of support vectors: {self.model.n_support_}')
        # 5.3 Show the weight of the model
        logging.log(logging.INFO, f'Weight of the model: {self.model.coef_}')
        # 5.4 Show the bias of the model
        logging.log(logging.INFO, f'Bias of the model: {self.model.intercept_}')
        # 5.5 Show the score of the model
        logging.log(logging.INFO, f'Score of the model: {self.model.score(self.test_data, self.test_label)}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    classifier = Classifier(
        train_data=np.load('../preprocess/train_pca_64.npy', allow_pickle=True),
        test_data=np.load('../preprocess/test_pca_64.npy', allow_pickle=True),
        need_pca=False
    )
    classifier.__prepare__()
    classifier.classify()
    classifier.predict()
