import numpy as np
import os


class DecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth, self.min_samples_split, self.root = max_depth, min_samples_split, None

    def fit(self, X, y):
        self.root = self.split(X, y, 1)

    def split(self, X, y, depth):
        node, error, splitting_variable, splitting_threshold = {}, float("inf"), 0, 0
        for i in range(X.shape[1]):
            for threshold in X[:, i]:
                error_new = self.get_error(X, y, i, threshold)
                if error_new < error:
                    splitting_variable, splitting_threshold, error = i, threshold, error_new

        node["splitting_variable"], node["splitting_threshold"] = splitting_variable, splitting_threshold

        left_X, left_y = X[X[:, splitting_variable] <= splitting_threshold], \
                         y[X[:, splitting_variable] <= splitting_threshold]
        right_X, right_y = X[X[:, splitting_variable] > splitting_threshold], \
                           y[X[:, splitting_variable] > splitting_threshold]

        node["left"] = self.split(left_X, left_y, depth + 1) \
            if len(left_X) >= self.min_samples_split and depth < self.max_depth \
            else np.mean(left_y)
        node["right"] = self.split(right_X, right_y, depth + 1) \
            if len(right_X) >= self.min_samples_split and depth < self.max_depth \
            else np.mean(right_y)

        return node

    def get_error(self, X, y, feature_index, threshold):
        features = X[:, feature_index]
        y_left, y_right = y[features <= threshold], y[features > threshold]
        return np.var(y_left) * len(y_left) + np.var(y_right) * len(y_right)

    def predict(self, X):
        return np.apply_along_axis(self.predict_record, axis=1, arr=X)

    def predict_record(self, x):
        node = self.root
        while not isinstance(node, float):
            node = node["left"] if x[node["splitting_variable"]] <= node["splitting_threshold"] else node["right"]
        return node


class GradientBoostingRegressor():
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=5, min_samples_split=2):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.estimators = np.empty((self.n_estimators,), dtype=np.object)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.f0 = 0

    def fit(self, X, y):
        self.f0 = np.mean(y)
        yim = np.full(len(y), self.f0)
        for i in range(self.n_estimators):
            dt = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            dt.fit(X, y - yim)
            self.estimators[i] = dt
            yim += self.learning_rate * dt.predict(X)

    def predict(self, X):
        pred = np.full(len(X), self.f0)
        for estimator in self.estimators:
            pred += estimator.predict(X) * self.learning_rate
        return pred


if __name__ == '__main__':
    # test for DT
    x_train = np.genfromtxt("Test_data" + os.sep + "x_0.csv", delimiter=",")
    y_train = np.genfromtxt("Test_data" + os.sep + "y_0.csv", delimiter=",")
    tree = DecisionTreeRegressor(min_samples_split=2)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_train)
    y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_dt.csv", delimiter=",")
    print(y_pred - y_test_pred < 1e-5)

    # test for GBDT
    x_train = np.genfromtxt("Test_data" + os.sep + "x_0.csv", delimiter=",")
    y_train = np.genfromtxt("Test_data" + os.sep + "y_0.csv", delimiter=",")
    gbr = GradientBoostingRegressor(n_estimators=10)
    gbr.fit(x_train, y_train)
    y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_gb.csv", delimiter=",")
    y_pred = gbr.predict(x_train)
    print(y_pred - y_test_pred < 1e-5)
