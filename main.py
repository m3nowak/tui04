import dataset
import classifier

def main():
    ds_x, ds_y = dataset.load_dataset()
    clsfr = classifier.create_classifier(verbose=True)
    classifier.train(clsfr, ds_x[:300], ds_y[:300])
    print(classifier.rate(clsfr, ds_x[300:400], ds_y[300:400]))
    classifier.dump_classifier(clsfr, '2')

if __name__ == "__main__":
    main()