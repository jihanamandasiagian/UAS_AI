import numpy as np
import pandas as pd

# KNN
class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = distances.argsort()[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_labels).argmax())
        return np.array(predictions)

# Decision Tree
class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.tree = self._build_tree(np.arange(len(y)), depth=0)

    def _build_tree(self, indices, depth):
        if len(set(self.y_train[indices])) == 1 or depth == self.max_depth:
            return np.bincount(self.y_train[indices]).argmax()

        best_feature, best_threshold = self._find_best_split(indices)
        left_indices = indices[self.X_train[indices, best_feature] < best_threshold]
        right_indices = indices[self.X_train[indices, best_feature] >= best_threshold]

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(left_indices, depth + 1),
            'right': self._build_tree(right_indices, depth + 1)
        }

    def _find_best_split(self, indices):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(self.X_train.shape[1]):
            thresholds = np.unique(self.X_train[indices, feature])
            for threshold in thresholds:
                left = self.y_train[indices[self.X_train[indices, feature] < threshold]]
                right = self.y_train[indices[self.X_train[indices, feature] >= threshold]]
                gini = self._gini_impurity(left, right)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, left, right):
        def gini(labels):
            proportions = np.bincount(labels, minlength=2) / len(labels)
            return 1 - np.sum(proportions ** 2)

        n = len(left) + len(right)
        return len(left) / n * gini(left) + len(right) / n * gini(right)

    def _predict(self, node, x):
        if isinstance(node, dict):
            if x[node['feature']] < node['threshold']:
                return self._predict(node['left'], x)
            else:
                return self._predict(node['right'], x)
        else:
            return node

    def predict(self, X):
        return np.array([self._predict(self.tree, x) for x in X])

# Voting Classifier
class VotingClassifier:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models]) 
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def load_data():
    data = pd.read_csv('heart_disease_data.csv')
    X = data.drop(columns=['num']).values  
    y = data['num'].astype(int).values 
    return X, y

def main(Input_data):
    X, y = load_data()
    split_ratio = 0.9
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    knn = KNearestNeighbors(k=11)
    dt = DecisionTreeClassifier(max_depth=10)
    voting = VotingClassifier(models=[knn, dt])

    voting.fit(X_train, y_train)

    def predict_heart_disease(input_data):
        expected_features = ['age', 'trestbps', 'chol', 'oldpeak', 'ca','thalch',
            'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal',
            'cp_typical angina', 'restecg_lv hypertrophy', 'restecg_normal',
            'restecg_st-t abnormality', 'sex_Female', 'sex_Male', 'exang_False',
            'exang_True', 'fbs_False', 'fbs_True']

        if len(input_data) < len(expected_features):
            input_data += [0] * (len(expected_features) - len(input_data))
        
        input_array = np.array(input_data).reshape(1, -1)
        return voting.predict(input_array)[0]

    print(len(Input_data))
    prediction = predict_heart_disease(Input_data)
    print("Prediction:", f"Heart Disease {prediction}" if prediction >= 1 else "No Heart Disease")
    return prediction