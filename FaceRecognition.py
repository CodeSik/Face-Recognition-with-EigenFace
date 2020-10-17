from __future__ import print_function
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt

#TODO: 데이터를 바꿔야함.
#testface 6 : 아예 교체체
def createDataMatrix(images, mean):
    print("Creating data matrix", end=" ... ")

    numImages = len(images)
    sz = images[0].shape


    data = np.zeros(( sz[0] * sz[1], numImages ), dtype=np.float32)
    for i in range(0, numImages):
        #print("images[i]")
        #print(images[i].shape)
        image = images[i] - mean
        image = np.array(image).flatten()
        data[ :, i] = image

    print("DONE")

    return np.dot(data,data.T)



def readImages(path):
    print("Reading images from " + path, end="...")

    images = []

    average = np.zeros((1024, 1), dtype=np.float32)
    i = 0

    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".pgm"]:

            # Add to array of images
            imagePath = os.path.join(path, filePath)
            src = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(src,dsize=(32,32),interpolation=cv2.INTER_AREA)

            if im is None:
                print("image:{} not read properly".format(imagePath))
            elif i == 50000:
                break

            else:

                im = np.float32(im)

                im = np.array(im).flatten()
                im = im[:,None]

                average = np.add(average,im)
                images.append(im)

            i=i+1
    numImages = len(images)
    average = average / len(images)
    images = np.array(images)
    if numImages == 0:
        print("No images found")
        sys.exit(0)
    print(str(numImages) + " files read.")

    return images, average



def createFace(eigenfaces,testimage,mu):

    ck = []
    total_result =[]

    for img in testimage: #테스트 이미지 각각에 대해서
        result = np.zeros((1024, 1), dtype=np.float32)
        cn = []
        for eigenface in eigenfaces: #한 이미지에 대한 계수를 구하기 위해 이중포문, 각 eigenface에 해당하는 계수를 구함.
            imgnp = np.array(img)
            imgnp = imgnp-mu
            eigenfacenp = np.array(eigenface).reshape((1024,1))
            c = np.dot(eigenface.T,imgnp)
            weightedEigenface = eigenfacenp * c
            result = np.add(result,weightedEigenface)
            cn.append(c) #구하고 cn에 추가. cn은 행벡터로써 ck의 한 요소로 추가할 것.

        total_result.append(result)
        ck.append(cn) #추가하고 다음 이미지에서는 cn을 초기화

    total_result = np.array(total_result)
    ck = np.array(ck)
    return total_result, ck


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

if __name__ == '__main__':


    NUM_EIGEN_FACES = 64

    dir = "faces"
    images, mu = readImages(dir)
    data = createDataMatrix(images,mu)

    sz = images[0].shape




    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
    print("DONE")

    eigenFaces = [];

    for eigenVector in eigenVectors:
        eigenFace = np.array(eigenVector).reshape((1024,1))
        eigenFaces.append(eigenVector)

    eigenFaces = np.array(eigenFaces)



    dir1 = "testface1"
    testface1, mu1 = readImages(dir1)
    total_result1, ck1= createFace(eigenFaces, testface1,mu)

    dir2 = "testface2"
    testface2, mu2 = readImages(dir2)
    total_result2, ck2 = createFace(eigenFaces, testface2,mu)

    dir3 = "testface3"
    testface3, mu3 = readImages(dir3)
    total_result3, ck3 = createFace(eigenFaces, testface3,mu)

    dir4 = "testface4"
    testface4, mu4 = readImages(dir4)
    total_result4, ck4 = createFace(eigenFaces, testface4,mu)

    dir5 = "testface5"
    testface5, mu5 = readImages(dir5)
    total_result5, ck5 = createFace(eigenFaces, testface5,mu)

    dir6 = "testface6"
    testface6, mu6 = readImages(dir6)
    total_result6, ck6 = createFace(eigenFaces, testface6,mu)

    dir7 = "testface7"
    testface7, mu7 = readImages(dir7)
    total_result7, ck7 = createFace(eigenFaces, testface7,mu)

    dir8 = "testface8"
    testface8, mu8 = readImages(dir8)
    total_result8, ck8 = createFace(eigenFaces, testface8,mu)

    dir9 = "testface9"
    testface9, mu9 = readImages(dir9)
    total_result9, ck9 = createFace(eigenFaces, testface9,mu)

    dir10 = "testface10"
    testface10, mu10 = readImages(dir10)
    total_result10, ck10 = createFace(eigenFaces, testface10,mu)



    fig = plt.figure(1)
    row = 8
    col = 8
    i = 1


    for eigenFace in eigenFaces:
        eigenFace1 =eigenFace.reshape((32,32))
        ax = fig.add_subplot(row, col, i)
        ax.imshow(eigenFace1*128,'gray')
        ax.set_xlabel(xlabel=i)
        i = i + 1



    fig = plt.figure(2)
    row1 = 5
    col1 = 2
    i = 0
    j = 1
    for result in total_result1:
        correct = ""
        testface = np.array(testface1[i]).reshape((1024,1))

        testfaceToShow= np.array(testface1[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i,"번째 사진의 유사도: ",similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("1번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck1)):
        for j in range(i+1,len(ck1)):
                sim = cos_sim(ck1[j].T,ck1[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("1번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(3)
    i = 0
    j = 1
    for result in total_result2:
        correct = ""
        testface = np.array(testface2[i]).reshape((1024,1))
        testfaceToShow= np.array(testface2[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("2번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck2)):
        for j in range(i+1,len(ck2)):
                sim = cos_sim(ck2[j].T,ck2[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("2번째 Face에 대한 최종 유사도",total_similarity/cnt)


    print("-------------------------------------")

    fig = plt.figure(4)
    i = 0
    j = 1
    for result in total_result3:
        correct = ""
        testface = np.array(testface3[i]).reshape((1024,1))
        testfaceToShow= np.array(testface3[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("3번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck3)):
        for j in range(i+1,len(ck3)):
                sim = cos_sim(ck3[j].T,ck3[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("3번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(5)
    i = 0
    j = 1
    for result in total_result4:
        correct = ""
        testface = np.array(testface4[i]).reshape((1024,1))

        testfaceToShow= np.array(testface4[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("4번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck4)):
        for j in range(i+1,len(ck4)):
                sim = cos_sim(ck4[j].T,ck4[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("4번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(6)
    i = 0
    j = 1
    for result in total_result5:
        correct = ""
        testface = np.array(testface5[i]).reshape((1024,1))

        testfaceToShow= np.array(testface5[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("5번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck5)):
        for j in range(i+1,len(ck5)):
                sim = cos_sim(ck5[j].T,ck5[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("5번째 Face에 대한 최종 유사도",total_similarity/cnt)


    print("-------------------------------------")

    fig = plt.figure(7)
    i = 0
    j = 1
    for result in total_result6:
        correct = ""
        testface = np.array(testface6[i]).reshape((1024,1))

        testfaceToShow= np.array(testface6[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("6번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck6)):
        for j in range(i+1,len(ck6)):
                sim = cos_sim(ck6[j].T,ck6[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("6번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(8)
    i = 0
    j = 1
    for result in total_result7:
        correct = ""
        testface = np.array(testface7[i]).reshape((1024,1))

        testfaceToShow= np.array(testface7[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("7번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck7)):
        for j in range(i+1,len(ck7)):
                sim = cos_sim(ck7[j].T,ck7[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("7번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(9)
    i = 0
    j = 1
    for result in total_result8:
        correct = ""
        testface = np.array(testface8[i]).reshape((1024,1))

        testfaceToShow= np.array(testface8[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("8번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck8)):
        for j in range(i+1,len(ck8)):
                sim = cos_sim(ck8[j].T,ck8[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("8번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(10)

    i = 0
    j = 1
    for result in total_result9:
        correct = ""
        testface = np.array(testface9[i]).reshape((1024,1))

        testfaceToShow= np.array(testface9[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("9번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck9)):
        for j in range(i+1,len(ck9)):
                sim = cos_sim(ck9[j].T,ck9[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("9번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    fig = plt.figure(11)
    i = 0
    j = 1
    for result in total_result10:
        correct = ""
        testface = np.array(testface10[i]).reshape((1024,1))

        testfaceToShow= np.array(testface10[i]).reshape((32,32))
        resultToShow = result.reshape((32, 32))

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(testfaceToShow, 'gray')

        j=j+1

        ax = fig.add_subplot(row1, col1, j)
        ax.imshow(resultToShow, 'gray')

        similarity = cos_sim(testface.T,result)

        if similarity >= 0.4:
            correct = "correct"
        else:
            correct = "incorrect"
        print(i, "번째 사진의 유사도: ", similarity)
        ax.set_title(correct,fontsize = 13)

        i = i + 1
        j=j+1


    print("10번째 Face 5개 이미지 동일 확률: ")

    total_similarity = 0
    cnt = 0
    for i in range(0,len(ck10)):
        for j in range(i+1,len(ck10)):
                sim = cos_sim(ck10[j].T,ck10[i])
                print(i,"와",j,"의 유사도",sim)
                total_similarity = total_similarity + sim
                cnt = cnt+1
    print("10번째 Face에 대한 최종 유사도",total_similarity/cnt)

    print("-------------------------------------")

    plt.show()

