import os
import numpy as np
import sys


special_styles = []


def main(Input):
    train_files = []
    train_labels = []

    eval_files = []
    eval_labels = []

    styles = os.listdir(Input)
    styles.sort()

    for label, style in enumerate(styles):
        if style != ".DS_Store" and style not in special_styles:
            datasets = os.listdir(Input + "/" + style)
            for dataset in datasets:
                if dataset != ".DS_Store":
                    dataset_dir = Input + "/" + style + "/" + dataset
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
    print('starting making dataset...')
    main('old_Seperation')
    print('finished making dataset')
