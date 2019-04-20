import numpy as np
import cv2

img = cv2.imread('test.jpg')
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 30
ret, label, center = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)

height, width, channel = img.shape

label = label.flatten()

counts = [0] * K

for i in range(len(label)):
  counts[label[i]] += 1

start_position = []
result = np.zeros((Z.shape))

for i in range(len(counts)):
  if i == 0:
    start_position.append(0)
  else:
    start_position.append(start_position[i - 1] + counts[i - 1])

position_tracker = start_position.copy()
for i in range(len(result)):
  index = position_tracker[label[i]]
  result[index][0] = Z[i][0]
  result[index][1] = Z[i][1]
  result[index][2] = Z[i][2]
  position_tracker[label[i]] = index + 1

result = np.uint8(result.reshape(img.shape))

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
