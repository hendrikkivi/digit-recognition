import pandas as pd
import matplotlib.pyplot as plt
import os

dirName = "ex"

files = os.listdir(dirName)
files1 = files[:14]
files2 = files[14:]


def drawDigits(files, fileName):
    f, axarr = plt.subplots(14, 2, figsize=(25, 25))
    for index, file in enumerate(files):
        file = "ex/" + file
        data = pd.read_csv(file).as_matrix()
        xPoints = data[0:, 0]
        yPoints = data[0:, 3]
        
        y = index % 2
        x = int(index/2)
        
        axarr[x, y].axis('off')
        axarr[x, y].scatter(xPoints, yPoints)
        axarr[x, y].invert_yaxis()
        
    plt.savefig(fileName)
    
drawDigits(files, "test.png")

#drawDigits(files1, "test1.png")
#drawDigits(files2, "test2.png")