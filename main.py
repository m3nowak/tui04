import dataset
import classifier

def main():
    ds_x, ds_y = dataset.load_dataset()
    clsfr = classifier.create_classifier(verbose=True)
    classifier.train(clsfr, ds_x[:9000], ds_y[:9000])
    print(classifier.rate(clsfr, ds_x[9000:], ds_y[9000:]))
    classifier.dump_classifier(clsfr)

if __name__ == "__main__":
    main()