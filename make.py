import numpy as np
import sys
import time
import os
import random

def main(argv):
  styles = os.listdir(argv[1])
  styles.sort()
 
  train_files = []
  train_labels = []
  eval_files = []
  eval_labels = []
  for label, style in enumerate(styles):
    imgs = os.listdir(argv[1] + "/" + style)
    
    imgs.sort()
    split_index = int(len(imgs) * 0.9)
    if split_index % 4 > 0:
      split_index = (int(split_index / 4) + 1) * 4
    train_imgs = imgs[:split_index]
    eval_imgs = imgs[split_index:]
    train_labels = np.append(train_labels, [label] * len(train_imgs))
    eval_labels = np.append(eval_labels, [label] * len (eval_imgs))
    
    for train_img in train_imgs:
      train_files.append(argv[1] + "/" + style + "/" + train_img)
    for eval_img in eval_imgs:
      eval_files.append(argv[1] + "/" + style + "/" + eval_img)
    
  np.save('train_imgs', train_files)
  np.save('train_lbs', train_labels)

  np.save('eval_imgs', eval_files)
  np.save('eval_lbs', eval_labels)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Program accepts the directory parameter.")
  print("start making dataset...")
  start_time = time.time()
  main(sys.argv)
  end_time = time.time()
  print('finished:' + str(end_time - start_time) + "s")
