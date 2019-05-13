import os
import numpy as np
import sys


def main(argv):
    train_files = []
    train_labels = []

    eval_files = []
    eval_labels = []

    styles = os.listdir(argv[1])
    styles.sort()

    for label, style in enumerate(styles):
        if style != ".DS_Store":
            datasets = os.listdir(argv[1] + "/" + style)
            for dataset in datasets:
                if dataset != ".DS_Store":
                    dataset_dir = argv[1] + "/" + style + "/" + dataset
                    if dataset == "train_set":
                        imgs = os.listdir(dataset_dir)
                        for img in imgs:
                            if img != ".DS_Store":
                                train_files.append(dataset_dir + "/" + img)
                                train_labels.append(label)
                    elif dataset == "eval_set":
                        imgs = os.listdir(dataset_dir)
                        for img in imgs:
                            if img != ".DS_Store":
                                eval_files.append(dataset_dir + "/" + img)
                                eval_labels.append(label)

    np.save('train_imgs', train_files)
    np.save('train_lbs', train_labels)

    np.save('eval_imgs', eval_files)
    np.save('eval_lbs', eval_labels)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Program accepts the directory parameter!")
        sys.exit(1)

    print('starting making dataset...')
    main(sys.argv)
    print('finished making dataset')
