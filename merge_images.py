import os


def merge(_base_):
    num_sub_imgs = 16
    file_names = os.listdir(_base_)
    file_names.sort()
    if 'adriaen-brouwer_smallholders-playing-dice0.png' in file_names:
        print('True')
    size = len(file_names)
    index = 0
    while index < size:
        one_img_names = file_names[index:index + num_sub_imgs]
        print(one_img_names)
        index += num_sub_imgs


if __name__ == "__main__":
    base = 'output/step4/Baroque'
    merge(base)
