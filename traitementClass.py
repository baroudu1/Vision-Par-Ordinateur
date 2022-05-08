import os
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore


import warnings
warnings.filterwarnings("ignore")

class traitClass:

    def __init__(self, path):
        self.path = path
        self.img = None

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

        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.histeq(imag)
        self.img = img
        img_finale = self.traitement(img)
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
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        MaxV = np.max(imag)
        MinV = np.min(imag)
        Y = np.zeros_like(imag)
        m = imag.shape
        for i in range(m[0]):
            for j in range(m[1]):
                Y[i, j] = (255 / (MaxV - MinV) * imag[i, j] - MinV)
        img = Y
        self.img = img
        img_finale = self.traitement(img)
        return img_finale

    def openFile_traitement(self, nom_fichier):
        pathx = nom_fichier[0]
        pixmap = QtGui.QPixmap(pathx)
        pixmap4 = pixmap.scaled(311, 311, QtCore.Qt.KeepAspectRatio)
        img_finale = QtGui.QPixmap(pixmap4)
        return img_finale

    def negatif_traitement(self):
        image = cv2.imread(self.path)
        img = 255 - image
        self.img = img
        img_finale = self.traitement(img)
        return img_finale

    def histogram_traitement(self):
        image = cv2.imread(self.path)
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
        image = cv2.imread(self.path)
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
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def redimentionnage_traitement(self, pourcentage):
        image = cv2.imread(self.path)
        scale_percent = pourcentage
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def binarisationM_traitement(self, s):
        image = cv2.imread(self.path)
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

        image = cv2.imread(self.path)
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
        # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
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
            # print mub, muf
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = image.copy()
        final_img[image > final_thresh] = 255
        final_img[image < final_thresh] = 0

        return final_img

    def gradient_traitement(self,s):

        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.grad(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def grad(self, image, seuil):

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

        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Sobel(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Sobel(self, image, seuil):
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

    def laplacien_traitement(self, s):

        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Laplacien(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final


    def Laplacien(self, image, seuil):
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

        image = cv2.imread(self.path)
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
        return imagecopy

    def erosion_traitement(self, s):

        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Erosion(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final


    def Erosion(self,image, H):

        imagecopy = image.copy()

        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if (image[i][j] > 128):
                    image[i][j] = 255
                else:
                    image[i][j] = 0

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
        return imagecopy

    def ouverture_traitement(self, s):

        image = cv2.imread(self.path)
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

        image = cv2.imread(self.path)
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
        image = cv2.imread(self.path)
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
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Median(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.img = img
        img_final = self.traitement(img)
        return img_final

    def Median(self,image, taille):
        imagefiltrage = image.copy()
        x = int((taille - 1) / 2)
        for i in range(x, image.shape[0] - x):
            for j in range(x, image.shape[1] - x):
                liste = []
                if imagefiltrage[i, j] == 0 or imagefiltrage[i, j] == 255:
                    for n in range(-x, x):
                        for m in range(-x, x):
                            liste.append(imagefiltrage[i + n, j + m])
                    liste.sort()
                    imagefiltrage[i, j] = liste[x + 1]
                    while len(liste) > 0:
                        liste.pop()
        return imagefiltrage

    def moyenne_traitement(self, s):
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.Moyenneur(image, s)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_final = self.traitement(img)
        self.img = img
        return img_final

    def Moyenneur(self,image, taille):
        imagefiltrage = image.copy()
        x = int((taille - 1)/2)
        for i in range(x, image.shape[0] - x):
            for j in range(x, image.shape[1] - x):
                s = 0
                for n in range(-x, x):
                    for m in range(-x, x):
                        s += image[i+n, j+m]/(taille*taille)
                imagefiltrage[i, j] = s
        return imagefiltrage
    def enregistrement_traitememt(self, fileName):

        cv2.imwrite(fileName, self.img)