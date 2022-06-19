import os
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore
from skimage.filters import sobel,gaussian
from skimage.feature import peak_local_max
import warnings
warnings.filterwarnings("ignore")



class traitClass:

    def __init__(self, path):
        self.path = path
        self.img = None
        self.image = None


    def change(self):
        self.image = cv2.imread(self.path)
        # print(self.image)


    def traitement(self, img):
        try:
            height, width, byteValue = img.shape
        except Exception:
            height, width = img.shape
            byteValue = 1

        if byteValue == 3:
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(311, 311, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap(pixmap4)

    def egalisation_traitement(self):

        image = self.image
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.histeq(imag)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        self.img = img
        img_finale = self.traitement(img)
        return img_finale

    def usee(self):
        img_finale = self.traitement(self.img)
        self.image = self.img
        return img_finale

    def histeq(self, image):

        h = self.imhist(image)
        cdf = np.array(self.cumsum(h))
        sk = np.uint8(255 * cdf)
        s1, s2 = image.shape
        Y = np.zeros_like(image)
        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[image[i, j]]
        return Y

    def imhist(self, im):
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[im[i, j]] += 1
        return np.array(h) / (m * n)

    def cumsum(self, h):
        return [sum(h[:i + 1]) for i in range(len(h))]


    def etirement_traitement(self):
        image = self.image
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        MaxV = np.max(imag)
        MinV = np.min(imag)
        Y = np.zeros_like(imag)
        m = imag.shape
        for i in range(m[0]):
            for j in range(m[1]):
                Y[i, j] = (255 / (MaxV - MinV) * imag[i, j] - MinV)
        img = Y

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_finale = self.traitement(img)
        return img_finale

    def openFile_traitement(self, nom_fichier):
        pathx = nom_fichier[0]
        pixmap = QtGui.QPixmap(pathx)
        pixmap4 = pixmap.scaled(311, 311, QtCore.Qt.KeepAspectRatio)
        img_finale = QtGui.QPixmap(pixmap4)
        self.change()
        return img_finale

    def negatif_traitement(self):
        image = self.image
        img = 255 - image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        img_finale = self.traitement(img)
        return img_finale

    def histogram_traitement(self):
        image = self.image
        ###########ADDED###########
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ###########################
        k = 0
        try:
            test = image.shape[2]
        except IndexError:
            k = 1
        if k == 1:
            h = self.histo(image)
            plt.subplot(1, 1, 1)
            plt.plot(h)
            plt.savefig('Y_X.png')
        else:
            for i in range(0, 3):
                h = self.histo(image[:, :, i])
                plt.subplot(1, 3, i + 1)
                plt.plot(h)
            plt.savefig('Y_X.png')

        img = cv2.imread("Y_X.png")
        self.img = img
        img_final = self.traitement(img)
        files = glob.glob('Y_X.png')
        for i in files:
            os.remove(i)
        return img_final

    def histo(self, image):
        h = np.zeros(256)
        s = image.shape
        for j in range(s[0]):
            for i in range(s[1]):
                valeur = image[j, i]
                h[valeur] += 1
        return h

    def rotate_image_traitement(self, angle):
        image = self.image
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        img = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def redimentionnage_traitement(self, pourcentage):
        image = self.image
        scale_percent = pourcentage
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def binarisationM_traitement(self, s):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Seuillage(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Seuillage(self, image, s):
        imageX = image.copy()
        for i in range(1, imageX.shape[0]):
            for j in range(1, imageX.shape[1]):
                if imageX[i, j] < s:
                    imageX[i, j] = 0
                else:
                    imageX[i, j] = 255
        return imageX

    def otsu_traitement(self):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Otsu(image)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Otsu(self, image):

        pixel_number = image.shape[0] * image.shape[1]
        mean_weigth = 1.0 / pixel_number
        his, bins = np.histogram(image, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)
        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth
            if pcb == 0:
                pcb = 1
            if pcf == 0:
                pcf = 1
            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = image.copy()
        final_img[image > final_thresh] = 255
        final_img[image < final_thresh] = 0

        return final_img

    def gradient_traitement(self,s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.gradient(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def gradient(self, image1, seuil):
        image = self.Otsu(image1)
        imageX = image.copy()
        imageY = image.copy()
        for i in range(0, image.shape[0] - 2):
            for j in range(0, image.shape[1] - 2):
                imageX[i, j] = image[i, j+1] - image[i, j]
                imageY[i, j] = image[i+1, j] - image[i, j]
        imageXY = image.copy()
        for i in range(0, image.shape[0] - 1):
            for j in range(0, image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def sobel_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Sobel(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Sobel(self, image1, seuil):

        image = self.Otsu(image1)
        imageX = image.copy()
        imageY = image.copy()
        for i in range(0, image.shape[0] - 2):
            for j in range(0, image.shape[1] - 2):
                imageY[i, j] = -image[i-1, j-1] - 2*image[i, j-1] - image[i+1, j-1] \
                    + image[i - 1, j + 1] + 2 * \
                    image[i, j + 1] + image[i + 1, j + 1]
                imageX[i, j] = image[i-1, j-1] + 2*image[i-1, j] + image[i - 1, j + 1]\
                    - image[i+1, j-1] - 2 * \
                    image[i+1, j] - image[i + 1, j + 1]
        imageXY = image.copy()
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def robert_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Robert(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_final = self.traitement(img)
        return img_final

    def Robert(self, image1, seuil):

        image = self.Otsu(image1)

        imageX = image.copy()
        imageY = image.copy()
        for i in range(0, image.shape[0] - 1):
            for j in range(0, image.shape[1] - 1):
                imageX[i, j] = image[i , j + 1] - image[i + 1, j]

                imageY[i, j] = image[i , j ] - image[i + 1, j+1]
        imageXY = image.copy()
        for i in range(0, image.shape[0] - 1):
            for j in range(0, image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY



    def laplacien_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Laplacien(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final


    def Laplacien(self, image1, seuil):

        image = self.Otsu(image1)
        imageXY = image.copy()
        for i in range(1, image.shape[0] - 2):
            for j in range(1, image.shape[1] - 1):
                imageXY[i, j] = -4*image[i, j] + image[i-1, j] + image[i+1, j] \
                    + image[i, j - 1] + image[i, j + 1]
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY


    def dilatation_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.dilatation(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final




    def dilatation(self,image, H):

        imagecopy = image.copy()
        for i in range(1, image.shape[0] - 2):
            for j in range(1, image.shape[1] - 2):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 0):
                    imagecopy[i][j] = 0
                else:
                    imagecopy[i][j] = 255
        kernel = np.array(H, "uint8")
        imagecopy = cv2.dilate(image, kernel, iterations=1)
        return imagecopy

    def erosion_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Erosion(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final


    def Erosion(self,image, H):

        imagecopy = image.copy()

        for i in range(1, image.shape[0] - 2):
            for j in range(1, image.shape[1] - 2):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 2295):
                    imagecopy[i][j] = 255
                else:
                    imagecopy[i][j] = 0

        kernel = np.array(H, "uint8")
        imagecopy = cv2.erode(image, kernel, iterations=1)

        return imagecopy

    def ouverture_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Ouverture(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Ouverture(self,image, H):
        img = self.Erosion(image, H)
        image1 = self.dilatation(img, H)
        return image1

    def fermeture_traitement(self, s):

        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Fermeture(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Fermeture(self,image, H):
        img = self.dilatation(image, H)
        image1 = self.Erosion(img, H)
        return image1

    def gaussien_traitement(self, s):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.gaussien(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def gaussien(self, image, v):

        imagefiltrage = image.copy()
        x = 1
        for i in range(x, image.shape[0] - x):
            for j in range(x, image.shape[1] - x):
                s = 0
                for a in range(-x, x):
                    for b in range(-x, x):
                        s = s + self.h(a, b, v) * image[i + a, j + b]
                imagefiltrage[i, j] = s
                s = 0
        return imagefiltrage

    def h(self, x, y, v):
        x = (1 / (2 * math.pi * math.pow(v, 2))) * \
            (math.exp(-(math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(v, 2))))
        return x


    def median_traitement(self, s):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Median(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Median(self,image, taille):
        imagefiltrage = image.copy()
        x = int(taille/ 2)
        for i in range(x, image.shape[0] - x):
            for j in range(x, image.shape[1] - x):
                liste = []
                for ii in range(x, taille+x):
                    for jj in range(x, taille+x):
                        liste.append(image[i-ii][j-jj])
                liste.sort()
                median = liste[int(taille*taille/2) +1]
                imagefiltrage[i][j] = median
        return imagefiltrage

    def moyenne_traitement(self, s):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Moyenneur(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_final = self.traitement(img)
        self.img = img
        return img_final

    def Moyenneur(self,image, taille):
        imagefiltrage = image.copy()
        x = int(taille /2)
        for i in range(x, image.shape[0] - x):
            for j in range(x, image.shape[1] - x):
                s = 0
                for n in range(x, taille+x):
                    for m in range(x, taille+x):
                        s += image[i-n, j-m]
                imagefiltrage[i, j] = s/(taille*taille)
        return imagefiltrage
    def enregistrement_traitememt(self, fileName):

        cv2.imwrite(fileName, self.img)

    # def regionGrow_traitememt1(self):
    #     image = self.image
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     seeds = [[20, 100], [82, 150], [20, 250]]
    #     img = self.regionGrow(image, seeds, 10).astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #
    #     img_final = self.traitement(img)
    #     self.img = img
    #     return img_final
    #
    # def getGrayDiff(self,img, currentPoint, tmpPoint):
    #     return abs(int(img[currentPoint[0], currentPoint[1]]) - int(img[tmpPoint[0], tmpPoint[1]]))
    #
    # def selectConnects(self,p):
    #     if p != 0:
    #         connects = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
    #     else:
    #         connects = [[0, -1], [1, 0], [0, 1], [-1, 0]]
    #     return connects
    #
    # def regionGrow(self, img, seeds, thresh, p=1):
    #     try:
    #         height, weight = img.shape
    #         seedMark = np.zeros(img.shape)
    #         seedList = []
    #         for seed in seeds:
    #             seedList.append(seed)
    #         label = 255
    #         connects = self.selectConnects(p)
    #         while (len(seedList) > 0):
    #             currentPoint = seedList.pop(0)
    #             #print(currentPoint)
    #             seedMark[currentPoint[0], currentPoint[1]] = label
    #             for i in range(8):
    #                 tmpX = currentPoint[0] + connects[i][0]
    #                 tmpY = currentPoint[1] + connects[i][1]
    #                 if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
    #                     continue
    #                 grayDiff = self.getGrayDiff(img, currentPoint, [tmpX, tmpY])
    #                 if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
    #                     seedMark[tmpX, tmpY] = label
    #                     seedList.append([tmpX, tmpY])
    #         return seedMark
    #     except Exception as e:
    #         print(e)

    def kMeansSegmentation(self,img,k):
        try:
            h, w = img.shape
            image_2d = img.reshape(h * w, 1)
            pixel_vals = np.float32(image_2d)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))
            return segmented_image

        except Exception as e:
            print("kMeansSegmentation ERROR : ", e)

    def kmeans_traitement(self, k):
        image = self.image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = self.kMeansSegmentation(image, k)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_final = self.traitement(img)
        self.img = img
        return img_final

    def regionGrow_traitememt(self):

        img = self.Growing()

        img_final = self.traitement(img)
        self.img = img
        return img_final


    # def regionGrow1(self,image):
    #     img = image
    #     gray_img = gaussian(img, sigma=4)
    #     brgb = sobel(gray_img[:, :])
    #
    #     markers = peak_local_max(brgb.max()-brgb)
    #
    #     markers = peak_local_max(brgb.max() - brgb, threshold_rel=0.99, min_distance=50)
    #     (thresh, bin_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    #
    #     h = img.shape[0]
    #     w = img.shape[1]
    #
    #     out_img = np.zeros(shape=(gray_img.shape), dtype=np.uint8)
    #     print("markers: ", markers)
    #     seeds = markers.tolist()
    #     for seed in seeds:
    #         x = seed[0]
    #         y = seed[1]
    #         out_img[x][y] = 255
    #     directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    #     visited = np.zeros(shape=(gray_img.shape), dtype=np.uint8)
    #     while len(seeds):
    #         seed = seeds.pop(0)
    #         x = seed[0]
    #         y = seed[1]
    #         visited[x][y] = 1
    #
    #         for direct in directs:
    #             cur_x = x + direct[0]
    #             cur_y = y + direct[1]
    #             if cur_x < 0 or cur_y < 0 or cur_x >= h or cur_y >= w:
    #                 continue
    #             if (not visited[cur_x][cur_y]) and (bin_img[cur_x][cur_y] == bin_img[x][y]):
    #                 out_img[cur_x][cur_y] = 255
    #                 visited[cur_x][cur_y] = 1
    #                 seeds.append((cur_x, cur_y))
    #     print(visited)
    #     bake_img = img.copy()
    #     h = bake_img.shape[0]
    #     w = bake_img.shape[1]
    #     for i in range(h):
    #         for j in range(w):
    #             if out_img[i][j] != 255:
    #                 bake_img[i][j] = 0
    #     print(img)
    #     print(bake_img)
    #     return bake_img

    def Markers(self):
        img = self.image
        x, y, z = img.shape
        im_ = gaussian(img, sigma=4)
        if z == 3:
            br = sobel(im_[:, :, 0])
            bg = sobel(im_[:, :, 1])
            bb = sobel(im_[:, :, 2])
            brgb = br + bg + bb
        else:
            brgb = sobel(im_[:, :])

        markers = peak_local_max(brgb.max() - brgb, threshold_rel=0.99, min_distance=50)
        return markers

    def Growing(self):
        markers = self.Markers()
        img = self.image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, bin_img) = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        h = img.shape[0]
        w = img.shape[1]

        out_img = np.zeros(shape=(gray_img.shape), dtype=np.uint8)

        seeds = markers.tolist()
        for seed in seeds:
            x = seed[0]
            y = seed[1]
            out_img[x][y] = 255
        directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        visited = np.zeros(shape=(gray_img.shape), dtype=np.uint8)
        while len(seeds):
            seed = seeds.pop(0)
            x = seed[0]
            y = seed[1]
            visited[x][y] = 1

            for direct in directs:
                cur_x = x + direct[0]
                cur_y = y + direct[1]
                if cur_x < 0 or cur_y < 0 or cur_x >= h or cur_y >= w:
                    continue
                if (not visited[cur_x][cur_y]) and (bin_img[cur_x][cur_y] == bin_img[x][y]):
                    out_img[cur_x][cur_y] = 255
                    visited[cur_x][cur_y] = 1
                    seeds.append((cur_x, cur_y))
        bake_img = img.copy()
        h = bake_img.shape[0]
        w = bake_img.shape[1]
        for i in range(h):
            for j in range(w):
                if out_img[i][j] != 255:
                    bake_img[i][j][0] = 255
                    bake_img[i][j][1] = 255
                    bake_img[i][j][2] = 255

        return bake_img

    def Division_Judge(self, img, h0, w0, h, w):
        area = img[h0: h0 + h, w0: w0 + w]
        mean = np.mean(area)
        std = np.std(area, ddof=1)

        total_points = 0
        operated_points = 0

        for row in range(area.shape[0]):
            for col in range(area.shape[1]):
                if (area[row][col] - mean) < 2 * std:
                    operated_points += 1
                total_points += 1

        if operated_points / total_points >= 0.95:
            return True
        else:
            return False

    def Merge(self, img, h0, w0, h, w):

        for row in range(h0, h0 + h):
            for col in range(w0, w0 + w):
                if img[row, col] > 100 and img[row, col] < 200:
                    img[row, col] = 0
                else:
                    img[row, col] = 255

    def Recursion(self, img, h0, w0, h, w):
        # If the splitting conditions are met, continue to split
        if not self.Division_Judge(img, h0, w0, h, w) and min(h, w) > 5:
            # Recursion continues to determine whether it can continue to split
            # Top left square
            self.Division_Judge(img, h0, w0, int(h0 / 2), int(w0 / 2))
            # Upper right square
            self.Division_Judge(img, h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
            # Lower left square
            self.Division_Judge(img, h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
            # Lower right square
            self.Division_Judge(img, h0 + int(h0 / 2), w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
        else:
            # Merge
            self.Merge(img, h0, w0, h, w)

    def partition_traitememt(self):
        img = self.image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        segemented_img = img_gray.copy()
        self.Recursion(segemented_img, 0, 0, segemented_img.shape[0], segemented_img.shape[1])
        img = cv2.cvtColor(segemented_img, cv2.COLOR_GRAY2RGB)

        img_final = self.traitement(img)
        self.img = img
        return img_final

    def hit_or_miss_traitememt(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        img = self.hitOrMiss(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = self.Otsu(img)
        img_final = self.traitement(img)
        return img_final

    def hitOrMiss(self, image):
        try:
            input_image = image
            kernel = np.array((
                [0,  1, 0],
                [1, -1, 1],
                [0,  1, 0]), dtype="int")

            output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
            return output_image


        except Exception as e:
            print("hit or miss ERROR = ", e)

