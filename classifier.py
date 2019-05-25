from sklearn import neural_network, metrics
import pickle
import contextlib

DUMP_LOCATION = 'artifacts/faces-celeba/classifier.pkl'

def create_classifier():
    clsfr = neural_network.MLPClassifier(hidden_layer_sizes=(100,), max_iter=1, warm_start=True)
    return clsfr

def dump_classifier(classifier):
    with contextlib.closing(open(DUMP_LOCATION, 'wb')) as pfile:
        pickle.dump(classifier, pfile)

def load_classifier():
    with contextlib.closing(open(DUMP_LOCATION, 'rb')) as pfile:
        clsfr = pickle.load(pfile)
    return clsfr

def train(clsfr, x, y, iterations=1):
    for _ in range(iterations):
        clsfr.fit(x, y)

def rate(clsfr, x, y):
    y_predicted = (clsfr.predict(x) > 0.5).astype('int')
    return metrics.accuracy_score(y, y_predicted)
