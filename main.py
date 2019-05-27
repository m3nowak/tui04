import dataset
import classifier

def main():
    check()

def check():
    s = classifier.load_classifier('100x25')
    print(s)

def learn():
    ds_x, ds_y = dataset.load_dataset()
    clsfr = classifier.create_classifier(verbose=True, layer_sizes=(100,25))
    classifier.train(clsfr, ds_x[:9000], ds_y[:9000])
    print(classifier.rate(clsfr, ds_x[9000:], ds_y[9000:]))
    classifier.dump_classifier(clsfr, '100x25')

if __name__ == "__main__":
    main()