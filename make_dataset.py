import os
import numpy as np


def main():
    base = 'output/color_sepe/'
    folders = os.listdir(base)
    label = 0

    train_files = []
    train_labels = []

    eval_files = []
    eval_labels = []

    vali_files = []
    vali_labels = []

    for folder in folders:
        if folder != 'aa':
            files = os.listdir(base + folder + '/')
            files = files[:1760]
            train_set_num = int(1408-176)
            eval_set_num = int(1408)
            train_set = files[:train_set_num]
            eval_set = files[train_set_num:eval_set_num]
            vali_set = files[eval_set_num:]
            for file in train_set:
                train_files.append(base + folder + '/' + file)
                train_labels.append(label)
            for file in eval_set:
                eval_files.append(base + folder + '/' + file)
                eval_labels.append(label)
            for file in vali_set:
                vali_files.append(base + folder + '/' + file)
                vali_labels.append(label)
            label += 1
    np.save('rbtrain_imgs', train_files)
    np.save('rbtrain_lbs', train_labels)
    np.save('rbeval_imgs', eval_files)
    np.save('rbeval_lbs', eval_labels)
    np.save('rbvali_imgs', vali_files)
    np.save('rbvali_lbs', vali_labels)


if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')
