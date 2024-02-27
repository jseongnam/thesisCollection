import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread('http://matplotlib.org/3.1.0/_images/stinkbug.png')

gray_img = np.dot(img[...,:3],[0.2989, 0.5870, 0.1140])
plt.imshow(gray_img, cmap="gray")
plt.show()
# 물체의 외각을 구하고, 원본 이미지와 합치기. -> 외각 : 무조건 검은색 -> 주위를 다 봤을 때, 모든 외각이 검은색이면 -> 검은색, 아니면 하얀색 
filter = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
temp_img = np.zeros(gray_img.shape)
for i in range(1,gray_img.shape[0]-1) :
    for j in range(1,gray_img.shape[1]-1):
        result = 0
        for i_ in range(3): # 0 1 2 
            for j_ in range(3):
                result += filter[i_][j_] * gray_img[i+i_-1][j+j_-1]
        if result > 0.3 :
            temp_img[i][j] = 1
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]) :
        gray_img[i][j] = min(gray_img[i][j], temp_img[i][j])
plt.imshow(gray_img, cmap="gray")
plt.show()

