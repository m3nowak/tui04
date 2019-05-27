from sklearn import neural_network, metrics
import pickle
import contextlib

DUMP_LOCATION = 'artifacts/faces-celeba/classifier{}.pkl'

def create_classifier(verbose=False, layer_sizes=(100,)):
    clsfr = neural_network.MLPClassifier(hidden_layer_sizes=layer_sizes, verbose=verbose)
    return clsfr

def dump_classifier(clsfr, name=''):
    with contextlib.closing(open(DUMP_LOCATION.format(name), 'wb')) as pfile:
        pickle.dump(clsfr, pfile)

def load_classifier(name=''):
    with contextlib.closing(open(DUMP_LOCATION.format(name), 'rb')) as pfile:
        clsfr = pickle.load(pfile)
    return clsfr

def train(clsfr, x, y):
    clsfr.fit(x, y)

def rate(clsfr, x, y):
    y_predicted = (clsfr.predict(x) > 0.5).astype('int')
    return metrics.accuracy_score(y, y_predicted)
