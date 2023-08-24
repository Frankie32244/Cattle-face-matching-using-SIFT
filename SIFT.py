import cv2
import pickle
import matplotlib.pyplot as plt
import time 

sift = cv2.SIFT_create()

# 数据集-> 001.jpg 002.jpg ... 103.jpg
imageList = []

# 测试集-> 001.jpg 002.jpg ... 050.jpg
image_test_List = []

# 数据集文件路径-> ./data/images/xxx.jpg
imagesPath = []

# 测试集文件路径-> ./data/test/xxx.jpg
images_test_Path = []

#利用computeSIFT()算出数据集的keypoints和descriptors 存放到keypoints[]和descriptors[]数组中
keypoints = []
descriptors = []

# 利用computeSIFT()算出测试集的descriptors and keypoints 存放到keypoints[]和descriptors[]数组中
test_keypoints = []
test_descriptors = []

#score_array 表示./data/test 文件夹下的图片和images文件夹下的图片匹配所得分数
score_array = []


# 利用cv2.resize()对图像进行缩放
def imageResizeTrain(image):
    maxD = 1024
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image


def imageResizeTest(image):
    maxD = 1024
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image


def computeSIFT(image):
    return sift.detectAndCompute(image, None)


def initImageList():
    for i in range(1, 104):
        filename = f"{i:03d}.jpg"
        imageList.append(filename)


def initImageTestList():
    for i in range(1, 51):
        filename = f"{i:03d}.jpg"
        image_test_List.append(filename)


def initImagePath():
    for imageName in imageList:
        imagePath = "data/images/" + str(imageName)
        imagesPath.append(imageResizeTrain(cv2.imread(imagePath,0))) # flag 0 means grayscale


def initImageTestPath():
    for imageName in image_test_List:
        imageTestPath = "data/test/" + str(imageName)
        images_test_Path.append(imageResizeTrain(cv2.imread(imageTestPath,0))) # flag 0 means grayscale


def getImagesKeypointsAndDes():
    for i,image in enumerate(imagesPath):
        keypointTemp, descriptorTemp = computeSIFT(image)
        keypoints.append(keypointTemp)
        descriptors.append(descriptorTemp)


def getTestImagesKeypointsAndDes():
    for i,image in enumerate(images_test_Path):
        keypointTemp, descriptorTemp = computeSIFT(image)
        test_keypoints.append(keypointTemp)
        test_descriptors.append(descriptorTemp)





# -----------------------------------------------------------------------------------------------------------
bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults


def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))


def getPlotFor(i,j,keypoint1,keypoint2,matches):
    image1 = imageResizeTest(cv2.imread("data/images/" + imageList[i]))
    image2 = imageResizeTest(cv2.imread("data/test/" + image_test_List[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)


def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255,255,255],
        flags=2
    )
    return matchPlot


def getScore(i,j):
    keypoint1 = keypoints[i]
    descriptor1 = descriptors[i]
    keypoint2 = test_keypoints[j]
    descriptor2 = test_descriptors[j]
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    return score


#  i 表示数据集的图片 0 -> 001.jpg  j 表示测试集的图片 0 -> 001.jpg
def getMaxScore(j):
    maxScore = 0
    global index
    for i in range(0,87):
        score = getScore(i,j)
        score_array.append(score)
        index  =  i if score > maxScore  else index
        # print(score,index)
        maxScore = max(maxScore,score)
    return maxScore

def calculateResultsFor(i,j):
    keypoint1 = keypoints[i]
    descriptor1 = descriptors[i]
    keypoint2 = test_keypoints[j]
    descriptor2 = test_descriptors[j]
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    plot = getPlotFor(i,j,keypoint1,keypoint2,matches)
    print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    print(score)
    plt.imshow(plot),plt.show()




if __name__ == '__main__':
    total_time = 0
    for i in range(0,50):
        start_time = time.time()
        print(getMaxScore(i)) 
        end_time = time.time()
        running_time = end_time - start_time
        total_time = total_time + running_time
        print("Running time:", running_time, "seconds")
        calculateResultsFor(index,i)
    print(total_time/50)
     
