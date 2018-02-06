
# The following file will download, extract and split the Standford Dog Breed dataset between
# a test and training set, where numtest is the number of images you chose to keep for your test
# set per dog breed
import urllib.request
import tarfile
import os
import shutil
# Folder paths for our training and test dataset
inputpath = 'Images_train'
outputpath = 'Images_test'
# Number of images per breed for the test set
numtest = 30

print('This can take a few minutes')
# Download the Images dataset from the Standford website
if not os.path.isfile('Images.tar'):
    print('Beginning file download with urllib2...')
    url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'  
    urllib.request.urlretrieve(url, './Images.tar') 
    print('Done')

# Extract the files from the Images.tar dataset    
if not os.path.isdir('Images'):
    print('Beginning file extraction...')
    tar = tarfile.open("Images.tar")
    tar.extractall()
    tar.close()
    print('Done')

# Rename Images folder as Images_train, we will split its contents afterward
os.rename('Images', 'Images_train')

# Create an Images_test folder and fill it with the same folder architecture as Images_train
if not os.path.isdir('Images_test'):
    print('Beginning train/test splitting...')
    os.mkdir('Images_test')
for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = outputpath + dirpath[len(inputpath):]
    if not os.path.isdir(structure):
        os.mkdir(structure)    
        
# Move the 30 first images from each subfolders of Images_train to the corresponding 
# subfolder in Images_test
for folderName, subfolders, filenames in os.walk(inputpath):
    for subfolder in subfolders:
        for _, _, filename in os.walk(inputpath+'/'+subfolder):
            for i in range(numtest):
                os.rename(inputpath +'/'+ subfolder + '/'+ filename[i], outputpath + '/' + subfolder + '/' + filename[i])
print('Done')
print("The data set is ready to use.")
        

