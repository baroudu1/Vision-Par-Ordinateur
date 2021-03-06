

from PyQt5 import QtCore, QtWidgets
from traitementClass import traitClass
from PyQt5.QtWidgets import QFileDialog


class Ui_MainWindow(object):
    def __init__(self):
        self.object = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(955, 487)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.widget_test = QtWidgets.QWidget(self.centralwidget)
        self.widget_test.setGeometry(QtCore.QRect(-1, -1, 955, 487))
        self.widget_test.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.widget_test.setObjectName("widget_test")

        self.labeltest = QtWidgets.QLabel(self.widget_test)
        self.labeltest.setGeometry(QtCore.QRect(300, 90, 400, 40))
        self.labeltest.setStyleSheet("font: italic 24pt \"Monotype Corsiva\";\n")
        self.labeltest.setObjectName("labeltest")

        self.labeltest_2 = QtWidgets.QLabel(self.widget_test)
        self.labeltest_2.setGeometry(QtCore.QRect(200, 170, 800, 40))
        self.labeltest_2.setStyleSheet("font: 14pt \"Modern No. 20\";\n"
                                       "color: rgb(255, 0, 0);")
        self.labeltest_2.setObjectName("labeltest_2")

        self.ImportButton = QtWidgets.QPushButton(self.widget_test)
        self.ImportButton.setGeometry(QtCore.QRect(390, 250, 150, 40))
        self.ImportButton.setStyleSheet("font: italic 16pt \"Monotype Corsiva\";\n"
                                        "background-color: rgb(230, 230, 230);")
        self.ImportButton.setObjectName("ImportButton")

        self.asideRotation = QtWidgets.QWidget(self.centralwidget)
        self.asideRotation.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideRotation.setStyleSheet("")
        self.asideRotation.setObjectName("asideRotation")

        self.asideRedimensionnage = QtWidgets.QWidget(self.centralwidget)
        self.asideRedimensionnage.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideRedimensionnage.setStyleSheet("")
        self.asideRedimensionnage.setObjectName("asideRedimensionnage")

        self.asideBinarisationM = QtWidgets.QWidget(self.centralwidget)
        self.asideBinarisationM.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideBinarisationM.setStyleSheet("")
        self.asideBinarisationM.setObjectName("asideBinarisationM")

        self.asideFiltrageGauss = QtWidgets.QWidget(self.centralwidget)
        self.asideFiltrageGauss.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideFiltrageGauss.setStyleSheet("")
        self.asideFiltrageGauss.setObjectName("asideFiltrageGauss")

        self.asideFiltrageMoyenne = QtWidgets.QWidget(self.centralwidget)
        self.asideFiltrageMoyenne.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideFiltrageMoyenne.setStyleSheet("")
        self.asideFiltrageMoyenne.setObjectName("asideFiltrageMoyenne")

        self.asideKMeans = QtWidgets.QWidget(self.centralwidget)
        self.asideKMeans.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideKMeans.setStyleSheet("")
        self.asideKMeans.setObjectName("asideKMeans")



        self.asideFiltrageMediane = QtWidgets.QWidget(self.centralwidget)
        self.asideFiltrageMediane.setGeometry(QtCore.QRect(10, 110, 231, 191))
        self.asideFiltrageMediane.setStyleSheet("")
        self.asideFiltrageMediane.setObjectName("asideFiltrageMediane")


        self.functionName = QtWidgets.QLabel(self.centralwidget)
        self.functionName.setGeometry(QtCore.QRect(50, 130, 170, 41))
        self.functionName.setStyleSheet("color: rgb(0, 0, 0);\n"
                                        "font: 12pt \"MS Shell Dlg 2\";\n"
                                        "font: 12pt \"8514oem\";")
        self.functionName.setWordWrap(False)
        self.functionName.setOpenExternalLinks(False)
        self.functionName.setObjectName("functionName")

        self.rotationLabel = QtWidgets.QLabel(self.asideRotation)
        self.rotationLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.rotationLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                         "font: 9pt \"MS Reference Sans Serif\";")
        self.rotationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.rotationLabel.setWordWrap(False)
        self.rotationLabel.setOpenExternalLinks(False)
        self.rotationLabel.setObjectName("rotationLabel")
        self.rotationInput = QtWidgets.QLineEdit(self.asideRotation)
        self.rotationInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.rotationInput.setAlignment(QtCore.Qt.AlignCenter)
        self.rotationInput.setObjectName("rotationInput")
        self.rotationButton = QtWidgets.QPushButton(self.asideRotation)
        self.rotationButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.rotationButton.setObjectName("rotationButton")

        self.RedimensionnageLabel = QtWidgets.QLabel(self.asideRedimensionnage)
        self.RedimensionnageLabel.setGeometry(QtCore.QRect(10, 80, 150, 25))
        self.RedimensionnageLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                                "font: 9pt \"MS Reference Sans Serif\";")
        self.RedimensionnageLabel.setWordWrap(False)
        self.RedimensionnageLabel.setOpenExternalLinks(False)
        self.RedimensionnageLabel.setObjectName("RedimensionnageLabel")
        self.RedimensionnageInput = QtWidgets.QLineEdit(self.asideRedimensionnage)
        self.RedimensionnageInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.RedimensionnageInput.setAlignment(QtCore.Qt.AlignCenter)
        self.RedimensionnageInput.setObjectName("RedimensionnageInput")
        self.RedimensionnageButton = QtWidgets.QPushButton(self.asideRedimensionnage)
        self.RedimensionnageButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.RedimensionnageButton.setObjectName("RedimensionnageButton")


        self.BinarisationMLabel = QtWidgets.QLabel(self.asideBinarisationM)
        self.BinarisationMLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.BinarisationMLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                              "font: 9pt \"MS Reference Sans Serif\";")
        self.BinarisationMLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.BinarisationMLabel.setWordWrap(False)
        self.BinarisationMLabel.setOpenExternalLinks(False)
        self.BinarisationMLabel.setObjectName("BinarisationMLabel")
        self.BinarisationMInput = QtWidgets.QLineEdit(self.asideBinarisationM)
        self.BinarisationMInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.BinarisationMInput.setAlignment(QtCore.Qt.AlignCenter)
        self.BinarisationMInput.setObjectName("BinarisationMInput")
        self.BinarisationMButton = QtWidgets.QPushButton(self.asideBinarisationM)
        self.BinarisationMButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.BinarisationMButton.setObjectName("BinarisationMButton")

        self.FiltrageGaussLabel = QtWidgets.QLabel(self.asideFiltrageGauss)
        self.FiltrageGaussLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.FiltrageGaussLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                              "font: 9pt \"MS Reference Sans Serif\";")
        self.FiltrageGaussLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageGaussLabel.setWordWrap(False)
        self.FiltrageGaussLabel.setOpenExternalLinks(False)
        self.FiltrageGaussLabel.setObjectName(" FiltrageGaussLabel")
        self.FiltrageGaussInput = QtWidgets.QLineEdit(self.asideFiltrageGauss)
        self.FiltrageGaussInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.FiltrageGaussInput.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageGaussInput.setObjectName(" FiltrageGaussInput")
        self.FiltrageGaussButton = QtWidgets.QPushButton(self.asideFiltrageGauss)
        self.FiltrageGaussButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.FiltrageGaussButton.setObjectName(" FiltrageGaussButton")

        self.FiltrageKMeansLabel = QtWidgets.QLabel(self.asideKMeans)
        self.FiltrageKMeansLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.FiltrageKMeansLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                                "font: 9pt \"MS Reference Sans Serif\";")
        self.FiltrageKMeansLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageKMeansLabel.setWordWrap(False)
        self.FiltrageKMeansLabel.setOpenExternalLinks(False)
        self.FiltrageKMeansLabel.setObjectName(" FiltrageKMeansLabel")

        self.FiltrageKMeansInput = QtWidgets.QLineEdit(self.asideKMeans)
        self.FiltrageKMeansInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.FiltrageKMeansInput.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageKMeansInput.setObjectName(" FiltrageKMeansInput")
        self.FiltrageKMeansButton = QtWidgets.QPushButton(self.asideKMeans)
        self.FiltrageKMeansButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.FiltrageKMeansButton.setObjectName(" FiltrageKMeansButton")







        self.FiltrageMoyenneLabel = QtWidgets.QLabel(self.asideFiltrageMoyenne)
        self.FiltrageMoyenneLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.FiltrageMoyenneLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                                "font: 9pt \"MS Reference Sans Serif\";")
        self.FiltrageMoyenneLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageMoyenneLabel.setWordWrap(False)
        self.FiltrageMoyenneLabel.setOpenExternalLinks(False)
        self.FiltrageMoyenneLabel.setObjectName(" FiltrageMoyenneLabel")
        self.FiltrageMoyenneInput = QtWidgets.QLineEdit(self.asideFiltrageMoyenne)
        self.FiltrageMoyenneInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.FiltrageMoyenneInput.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageMoyenneInput.setObjectName(" FiltrageMoyenneInput")
        self.FiltrageMoyenneButton = QtWidgets.QPushButton(self.asideFiltrageMoyenne)
        self.FiltrageMoyenneButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.FiltrageMoyenneButton.setObjectName(" FiltrageMoyenneButton")

        self.FiltrageMedianeLabel = QtWidgets.QLabel(self.asideFiltrageMediane)
        self.FiltrageMedianeLabel.setGeometry(QtCore.QRect(10, 80, 71, 25))
        self.FiltrageMedianeLabel.setStyleSheet("color: rgb(0, 0, 0);\n"
                                                "font: 9pt \"MS Reference Sans Serif\";")
        self.FiltrageMedianeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageMedianeLabel.setWordWrap(False)
        self.FiltrageMedianeLabel.setOpenExternalLinks(False)
        self.FiltrageMedianeLabel.setObjectName(" FiltrageMedianeLabel")
        self.FiltrageMedianeInput = QtWidgets.QLineEdit(self.asideFiltrageMediane)
        self.FiltrageMedianeInput.setGeometry(QtCore.QRect(110, 80, 91, 25))
        self.FiltrageMedianeInput.setAlignment(QtCore.Qt.AlignCenter)
        self.FiltrageMedianeInput.setObjectName(" FiltrageMedianeInput")
        self.FiltrageMedianeButton = QtWidgets.QPushButton(self.asideFiltrageMediane)
        self.FiltrageMedianeButton.setGeometry(QtCore.QRect(50, 130, 100, 25))
        self.FiltrageMedianeButton.setObjectName(" FiltrageMedianeButton")



        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(360, 390, 141, 41))
        self.label.setStyleSheet("color: rgb(0, 0, 0);\n"
                                 "font: 12pt \"8514oem\";")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(710, 390, 141, 41))
        self.label_2.setStyleSheet("color: rgb(0, 0, 0);\n"
                                   "font: 12pt \"8514oem\";")
        self.label_2.setObjectName("label_2")
        self.imageInitiale = QtWidgets.QLabel(self.centralwidget)
        self.imageInitiale.setGeometry(QtCore.QRect(280, 70, 311, 311))
        self.imageInitiale.setText("")
        self.imageInitiale.setObjectName("imageInitiale")

        self.usee = QtWidgets.QPushButton(self.centralwidget)
        self.usee.setGeometry(QtCore.QRect(720, 40, 150, 25))
        self.usee.setObjectName("usee")



        self.imageTraitee = QtWidgets.QLabel(self.centralwidget)
        self.imageTraitee.setGeometry(QtCore.QRect(620, 70, 311, 311))
        self.imageTraitee.setText("")
        self.imageTraitee.setObjectName("imageTraitee")





        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 955, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAnalyse_Elementaire = QtWidgets.QMenu(self.menubar)
        self.menuAnalyse_Elementaire.setObjectName("menuAnalyse_Elementaire")
        self.menuBinarisation = QtWidgets.QMenu(self.menubar)
        self.menuBinarisation.setObjectName("menuBinarisation")
        self.menuFiltrage = QtWidgets.QMenu(self.menubar)
        self.menuFiltrage.setObjectName("menuFiltrage")
        self.menuContour = QtWidgets.QMenu(self.menubar)
        self.menuContour.setObjectName("menuContour")
        self.menuMorphologie = QtWidgets.QMenu(self.menubar)
        self.menuMorphologie.setObjectName("menuMorphologie")
        self.menuSegmentaion = QtWidgets.QMenu(self.menubar)
        self.menuSegmentaion.setObjectName("menuSegmentaion")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionimporter_fichier = QtWidgets.QAction(MainWindow)
        self.actionimporter_fichier.setObjectName("actionimporter_fichier")
        self.actionsauvegarder = QtWidgets.QAction(MainWindow)
        self.actionsauvegarder.setObjectName("actionsauvegarder")
        self.actionNegatif = QtWidgets.QAction(MainWindow)
        self.actionNegatif.setObjectName("actionNegatif")
        self.actionHistogramme = QtWidgets.QAction(MainWindow)
        self.actionHistogramme.setObjectName("actionHistogramme")
        self.actionEgalisation = QtWidgets.QAction(MainWindow)
        self.actionEgalisation.setObjectName("actionEgalisation")
        self.actionEtirement = QtWidgets.QAction(MainWindow)
        self.actionEtirement.setObjectName("actionEtirement")
        self.actionManuelle = QtWidgets.QAction(MainWindow)
        self.actionManuelle.setObjectName("actionManuelle")
        self.actionOtsu = QtWidgets.QAction(MainWindow)
        self.actionOtsu.setObjectName("actionOtsu")
        self.actionMedian = QtWidgets.QAction(MainWindow)
        self.actionMedian.setObjectName("actionMedian")
        self.actionMoyenneur = QtWidgets.QAction(MainWindow)
        self.actionMoyenneur.setObjectName("actionMoyenneur")
        self.actionGaussien = QtWidgets.QAction(MainWindow)
        self.actionGaussien.setObjectName("actionGaussien")
        self.actionRotation = QtWidgets.QAction(MainWindow)
        self.actionRotation.setObjectName("actionRotation")
        self.actionRedimensionner = QtWidgets.QAction(MainWindow)
        self.actionRedimensionner.setObjectName("actionRedimensionner")
        self.actionGradient = QtWidgets.QAction(MainWindow)
        self.actionGradient.setObjectName("actionGradient")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionRobert = QtWidgets.QAction(MainWindow)
        self.actionRobert.setObjectName("actionRobert")
        self.actionLaplacien = QtWidgets.QAction(MainWindow)
        self.actionLaplacien.setObjectName("actionLaplacien")
        self.actionErosion = QtWidgets.QAction(MainWindow)
        self.actionErosion.setObjectName("actionErosion")
        self.actionHitOrMiss = QtWidgets.QAction(MainWindow)
        self.actionHitOrMiss.setObjectName("actionHitOrMiss")

        self.actionGrowingRegion = QtWidgets.QAction(MainWindow)
        self.actionGrowingRegion.setObjectName("actionGrowingRegion")
        self.actionKMeans = QtWidgets.QAction(MainWindow)
        self.actionKMeans.setObjectName("actionKMeans")
        self.actionPartition = QtWidgets.QAction(MainWindow)
        self.actionPartition.setObjectName("actionPartition")



        self.actionDilatation = QtWidgets.QAction(MainWindow)
        self.actionDilatation.setObjectName("actionDilatation")
        self.actionFermeture = QtWidgets.QAction(MainWindow)
        self.actionFermeture.setObjectName("actionFermeture")
        self.actionOuverture = QtWidgets.QAction(MainWindow)
        self.actionOuverture.setObjectName("actionOuverture")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionimporter_fichier)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionsauvegarder)
        self.menuFile.addSeparator()
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionNegatif)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionHistogramme)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionEgalisation)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionEtirement)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionRotation)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuAnalyse_Elementaire.addAction(self.actionRedimensionner)
        self.menuAnalyse_Elementaire.addSeparator()
        self.menuBinarisation.addSeparator()
        self.menuBinarisation.addAction(self.actionManuelle)
        self.menuBinarisation.addSeparator()
        self.menuBinarisation.addAction(self.actionOtsu)
        self.menuBinarisation.addSeparator()
        self.menuFiltrage.addSeparator()
        self.menuFiltrage.addAction(self.actionMedian)
        self.menuFiltrage.addSeparator()
        self.menuFiltrage.addAction(self.actionMoyenneur)
        self.menuFiltrage.addSeparator()
        self.menuFiltrage.addAction(self.actionGaussien)
        self.menuFiltrage.addSeparator()
        self.menuContour.addSeparator()
        self.menuContour.addAction(self.actionGradient)
        self.menuContour.addSeparator()
        self.menuContour.addAction(self.actionSobel)
        self.menuContour.addSeparator()
        self.menuContour.addAction(self.actionRobert)
        self.menuContour.addSeparator()
        self.menuContour.addAction(self.actionLaplacien)
        self.menuContour.addSeparator()
        self.menuMorphologie.addSeparator()
        self.menuMorphologie.addAction(self.actionErosion)
        self.menuMorphologie.addSeparator()
        self.menuMorphologie.addAction(self.actionDilatation)
        self.menuMorphologie.addSeparator()
        self.menuMorphologie.addAction(self.actionFermeture)
        self.menuMorphologie.addSeparator()
        self.menuMorphologie.addAction(self.actionOuverture)
        self.menuMorphologie.addSeparator()
        self.menuMorphologie.addAction(self.actionHitOrMiss)
        self.menuMorphologie.addSeparator()

        self.menuSegmentaion.addSeparator()
        self.menuSegmentaion.addAction(self.actionGrowingRegion)
        self.menuSegmentaion.addSeparator()
        self.menuSegmentaion.addAction(self.actionKMeans)
        self.menuSegmentaion.addSeparator()
        self.menuSegmentaion.addAction(self.actionPartition)
        self.menuSegmentaion.addSeparator()



        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAnalyse_Elementaire.menuAction())
        self.menubar.addAction(self.menuBinarisation.menuAction())
        self.menubar.addAction(self.menuFiltrage.menuAction())
        self.menubar.addAction(self.menuContour.menuAction())
        self.menubar.addAction(self.menuMorphologie.menuAction())
        self.menubar.addAction(self.menuSegmentaion.menuAction())

        self.retranslateUi(MainWindow)
        self.hideAll()
        self.hideAll2()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Traitement d\'Images"))
        self.functionName.setText(_translate("MainWindow", ""))
        self.rotationLabel.setText(_translate("MainWindow", "Angle :"))
        self.rotationInput.setText(_translate("MainWindow", "0"))
        self.rotationButton.setText(_translate("MainWindow", "Appliquer"))
        self.RedimensionnageLabel.setText(_translate("MainWindow", "Pourcentage :"))
        self.RedimensionnageButton.setText(_translate("MainWindow", "Appliquer"))
        self.BinarisationMInput.setText(_translate("MainWindow", ""))
        self.BinarisationMLabel.setText(_translate("MainWindow", "Sueil :"))
        self.BinarisationMButton.setText(_translate("MainWindow", "Appliquer"))
        self.ImportButton.setText(_translate("MainWindow", "Importer"))
        self.usee.setText(_translate("MainWindow", "Utiliser image traiter"))

        self.labeltest.setText(_translate("MainWindow", "Traitement d\'Images"))
        self.labeltest_2.setText(_translate("MainWindow", "Importer Votre Image Pour Commencer Les Testes"))

        self.FiltrageGaussInput.setText(_translate("MainWindow", "0.0"))
        self.FiltrageGaussLabel.setText(_translate("MainWindow", "Sueil :"))
        self.FiltrageGaussButton.setText(_translate("MainWindow", "Appliquer"))

        self.FiltrageMoyenneInput.setText(_translate("MainWindow", ""))
        self.FiltrageMoyenneLabel.setText(_translate("MainWindow", "Taille :"))
        self.FiltrageMoyenneButton.setText(_translate("MainWindow", "Appliquer"))

        self.FiltrageKMeansInput.setText(_translate("MainWindow", ""))
        self.FiltrageKMeansLabel.setText(_translate("MainWindow", "K :"))
        self.FiltrageKMeansButton.setText(_translate("MainWindow", "Appliquer"))

        self.FiltrageMedianeInput.setText(_translate("MainWindow", ""))
        self.FiltrageMedianeLabel.setText(_translate("MainWindow", "Taille :"))
        self.FiltrageMedianeButton.setText(_translate("MainWindow", "Appliquer"))

        self.label.setText(_translate("MainWindow", "Image Initiale"))
        self.label_2.setText(_translate("MainWindow", "Image Traitee"))
        self.menuFile.setTitle(_translate("MainWindow", "Fichier"))
        self.menuAnalyse_Elementaire.setTitle(_translate("MainWindow", "Analyse Elementaire"))
        self.menuBinarisation.setTitle(_translate("MainWindow", "Binarisation"))
        self.menuFiltrage.setTitle(_translate("MainWindow", "Filtrage"))
        self.menuContour.setTitle(_translate("MainWindow", "Contour"))
        self.menuMorphologie.setTitle(_translate("MainWindow", "Morphologie"))
        self.menuSegmentaion.setTitle(_translate("MainWindow", "Segmentaion"))
        self.actionimporter_fichier.setText(_translate("MainWindow", "Importer"))
        self.actionsauvegarder.setText(_translate("MainWindow", "Sauvegarder"))
        self.actionNegatif.setText(_translate("MainWindow", "Negatif"))
        self.actionHistogramme.setText(_translate("MainWindow", "Histogramme"))
        self.actionEgalisation.setText(_translate("MainWindow", "Egalisation"))
        self.actionEtirement.setText(_translate("MainWindow", "Etirement"))
        self.actionManuelle.setText(_translate("MainWindow", "Manuelle"))
        self.actionOtsu.setText(_translate("MainWindow", "Otsu"))
        self.actionMedian.setText(_translate("MainWindow", "Median"))
        self.actionMoyenneur.setText(_translate("MainWindow", "Moyenneur"))
        self.actionGaussien.setText(_translate("MainWindow", "Gaussien"))
        self.actionRotation.setText(_translate("MainWindow", "Rotation"))
        self.actionRedimensionner.setText(_translate("MainWindow", "Redimensionner"))
        self.actionGradient.setText(_translate("MainWindow", "Gradient"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionRobert.setText(_translate("MainWindow", "Robert"))
        self.actionLaplacien.setText(_translate("MainWindow", "Laplacien"))
        self.actionErosion.setText(_translate("MainWindow", "Erosion"))
        self.actionHitOrMiss.setText(_translate("MainWindow", "Hit or Miss"))
        self.actionDilatation.setText(_translate("MainWindow", "Dilatation"))
        self.actionFermeture.setText(_translate("MainWindow", "Fermeture"))
        self.actionOuverture.setText(_translate("MainWindow", "Ouverture"))


        self.actionGrowingRegion.setText(_translate("MainWindow", "Croissant Region"))
        self.actionKMeans.setText(_translate("MainWindow", "K-Means"))
        self.actionPartition.setText(_translate("MainWindow", "Partition Regions"))
####################################Button##################################################

        self.actionsauvegarder.triggered.connect(self.enregistrement)
        self.actionimporter_fichier.triggered.connect(self.openFile)
        self.ImportButton.clicked.connect(self.openFile)
        self.actionNegatif.triggered.connect(self.negatif)
        self.actionEtirement.triggered.connect(self.etirement)
        self.actionEgalisation.triggered.connect(self.egalisation)
        self.actionHistogramme.triggered.connect(self.histogram)
        self.rotationButton.clicked.connect(self.rotate_image)
        self.RedimensionnageButton.clicked.connect(self.redimentionnage)
        self.BinarisationMButton.clicked.connect(self.binarisationM)
        self.actionOtsu.triggered.connect(self.otsu)
        self.actionGradient.triggered.connect(self.gradient)
        self.actionSobel.triggered.connect(self.sobel)
        self.actionRobert.triggered.connect(self.robert)
        self.actionLaplacien.triggered.connect(self.Laplacien)
        self.actionErosion.triggered.connect(self.erosion)
        self.actionHitOrMiss.triggered.connect(self.hitOrMiss)
        self.actionDilatation.triggered.connect(self.dilatation)
        self.actionOuverture.triggered.connect(self.ouverture)
        self.actionFermeture.triggered.connect(self.fermeture)
        self.FiltrageGaussButton.clicked.connect(self.gaussien)
        self.FiltrageMedianeButton.clicked.connect(self.median)
        self.FiltrageMoyenneButton.clicked.connect(self.moyenne)
        self.FiltrageKMeansButton.clicked.connect(self.Kmeans)

        self.actionMedian.triggered.connect(self.show_FiltrageMedian)
        self.actionMoyenneur.triggered.connect(self.show_FiltrageMoyenne)
        self.actionGaussien.triggered.connect(self.show_FiltrageGauss)
        self.actionRotation.triggered.connect(self.show_rotate_image)
        self.actionRedimensionner.triggered.connect(self.show_redimentionnage)
        self.actionManuelle.triggered.connect(self.show_binarisationM)
        self.actionKMeans.triggered.connect(self.show_KMeans)
        self.actionPartition.triggered.connect(self.show_Partition)
        self.actionGrowingRegion.triggered.connect(self.show_Growing)
        self.usee.clicked.connect(self.useee)


####################################Functions##################################################
    def hideAll(self):
        self.asideRedimensionnage.hide()
        self.asideRotation.hide()
        self.asideBinarisationM.hide()
        self.asideFiltrageGauss.hide()
        self.asideFiltrageMoyenne.hide()
        self.asideFiltrageMediane.hide()
        self.asideKMeans.hide()

    def hideAll2(self):
        self.imageInitiale.hide()
        self.imageTraitee.hide()
        self.usee.hide()
        self.label.hide()
        self.label_2.hide()
        self.menubar.hide()


    def showAll(self):
        self.widget_test.hide()
        self.imageInitiale.show()
        self.imageTraitee.show()
        self.usee.show()

        self.label.show()
        self.label_2.show()
        self.menubar.show()

    def useee(self):
        self.hideAll()
        img_final = self.object.usee()
        self.imageInitiale.setPixmap(img_final)

    def enregistrement(self):
        x = "Image" + self.functionName.text()
        fileName = QFileDialog.getSaveFileName(
            None, 'some text', x+".png", "Image Files (*.jpg *.gif *.bmp *.png)")
        self.object.enregistrement_traitememt(fileName[0])
        self.functionName.setText("Enregistrement")

    def openFile(self):
        self.hideAll()
        nom_fichier = QFileDialog.getOpenFileName()
        self.object = traitClass(nom_fichier[0])
        img_finale = self.object.openFile_traitement(nom_fichier)
        self.imageInitiale.setPixmap(img_finale)
        self.functionName.setText("Importing")
        self.showAll()

    def negatif(self):
        self.hideAll()
        img_final = self.object.negatif_traitement()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Negatif")

    def etirement(self):
        self.hideAll()
        img_final = self.object.etirement_traitement()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Etirement")

    def egalisation(self):
        self.hideAll()
        img_final = self.object.egalisation_traitement()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Egalisation")

    def histogram(self):
        self.hideAll()
        img_final = self.object.histogram_traitement()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Histogram")

    def rotate_image(self):
        anglevalue = int(self.rotationInput.text())
        img_final = self.object.rotate_image_traitement(anglevalue)
        self.imageTraitee.setPixmap(img_final)

    def show_rotate_image(self):
        self.hideAll()
        self.asideRotation.show()
        self.functionName.setText("Rotation")

    def redimentionnage(self):
        pourcentage = int(self.RedimensionnageInput.text())
        img_final = self.object.redimentionnage_traitement(pourcentage)
        self.imageTraitee.setPixmap(img_final)

    def show_redimentionnage(self):
        self.hideAll()
        self.asideRedimensionnage.show()
        self.functionName.setText("Redimensionnage")

    def binarisationM(self):
        sueil = int(self.BinarisationMInput.text())
        img_final = self.object.binarisationM_traitement(sueil)
        self.imageTraitee.setPixmap(img_final)

    def show_binarisationM(self):
        self.hideAll()
        self.asideBinarisationM.show()
        self.functionName.setText("BinarisationM")

    def otsu(self):
        self.hideAll()
        img_final = self.object.otsu_traitement()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Otsu")

    def gradient(self):
        self.hideAll()
        img_final = self.object.gradient_traitement(20)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Gradient")

    def sobel(self):
        self.hideAll()
        img_final = self.object.sobel_traitement(50)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Sobel")

    def robert(self):
        self.hideAll()
        img_final = self.object.robert_traitement(50)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Robert")

    def Laplacien(self):
        self.hideAll()
        img_final = self.object.laplacien_traitement(50)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Laplacien")

    def erosion(self):
        self.hideAll()
        h = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

        img_final = self.object.erosion_traitement(h)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Erosion")

    def dilatation(self):
        self.hideAll()
        h = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        img_final = self.object.dilatation_traitement(h)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Dilatation")


    def fermeture(self):
        self.hideAll()
        h = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        img_final = self.object.fermeture_traitement(h)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Fermeture")

    def ouverture(self):
        self.hideAll()
        h = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        img_final = self.object.ouverture_traitement(h)
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Ouverture")

    def gaussien(self):
        v = float(self.FiltrageGaussInput.text())
        img_final = self.object.gaussien_traitement(v)
        self.imageTraitee.setPixmap(img_final)

    def show_FiltrageGauss(self):
        self.hideAll()
        self.asideFiltrageGauss.show()
        self.functionName.setText("Gaussien")

    def median(self):
        v = int(self.FiltrageMedianeInput.text())
        img_final = self.object.median_traitement(v)
        self.imageTraitee.setPixmap(img_final)

    def show_FiltrageMedian(self):
        self.hideAll()
        self.asideFiltrageMediane.show()
        self.functionName.setText("Median")

    def moyenne(self):
        v = int(self.FiltrageMoyenneInput.text())
        img_final = self.object.moyenne_traitement(v)
        self.imageTraitee.setPixmap(img_final)

    def show_FiltrageMoyenne(self):
        self.hideAll()
        self.asideFiltrageMoyenne.show()
        self.functionName.setText("Moyenneur")

    def show_KMeans(self):
        self.hideAll()
        self.asideKMeans.show()
        self.functionName.setText("K-means")

    def show_Growing(self):
        self.hideAll()
        img_final = self.object.regionGrow_traitememt()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Croissant Region")

    def show_Partition(self):
        self.hideAll()
        img_final = self.object.partition_traitememt()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Partition Region")

    def Kmeans(self):
        k = int(self.FiltrageKMeansInput.text())
        img_final = self.object.kmeans_traitement(k)
        self.imageTraitee.setPixmap(img_final)

    def hitOrMiss(self):
        self.hideAll()
        img_final = self.object.hit_or_miss_traitememt()
        self.imageTraitee.setPixmap(img_final)
        self.functionName.setText("Hit-Or-Miss")
