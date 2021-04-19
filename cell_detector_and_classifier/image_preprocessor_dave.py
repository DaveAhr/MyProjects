# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:00:52 2021

@author: dave-
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
import keras
import os, sys
from skimage import io
import xml.etree.ElementTree as ET
from xml.etree import cElementTree as ElementTree
#from xml_to_string import XmlDictConfig
import skimage
from skimage import util
from skimage.transform import resize
from PIL import Image



path_images = r'C:\Users\dave-\Documents\python_übungen\deep_learning_kurs\abschlussprojekt\cells_control'
names_images = os.listdir(path_images) #Liste der Bildnamen
path_images_new = r'C:\Users\dave-\Documents\python_übungen\deep_learning_kurs\abschlussprojekt\cells_control\\'
path_scaled_img = r'C:\Users\dave-\Documents\python_übungen\deep_learning_kurs\abschlussprojekt\images_control\\'

image_height = 600
image_width = 600
dpi = 96
figsize = (image_height/dpi, image_height/dpi) #umrechnung in inch



for i in names_images: #i sind Datei Namen
    j = names_images.index(i) 
    image = io.imread(path_images_new + str(names_images[j]))
    
    #Bilder croppen:
    left = 345
    top = 17
    right = 1528
    bottom = 1195
    image = image[17:1195, 345:1528, :]

    #Bilder skalieren:
    image = util.img_as_ubyte(skimage.transform.resize(image, (image_height, image_width)))
    
    #show Image without frames + save it
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    
    #neuer Bildname
    #aus Gründen der Reihenfolge der Bildnamen wird bei 100 gestartet
    new_name = path_scaled_img + str(j+100)+"_control"
    
    fig.savefig(new_name, dpi='figure', bbox_inches='tight', pad_inches=0)

















