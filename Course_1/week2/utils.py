from os import listdir
from os import path
import warnings

def getAllImageInfo(dir, extensions = ['.jpg', '.png'], recursive = False):
    """
    This function collect all image files in directory.
    
    Argument:
    dir -- root directory
    extensions -- image file extensions to search
    recursive -- whether include child directories recursively
    
    Returns:
    allImageNames -- list of image file names exist in directory
    allImageFullPath -- list of full path of image files
    allImageNamesWithExtension -- list of image file names with extension
    """
    allImageNames = []
    allImageFullPath = []
    allImageNamesWithExtension = []

    for f in listdir(dir):
        fileName, fileExtension = path.splitext(f)
        fullPath = path.join(dir, f)
        if path.isfile(fullPath):
            if fileExtension in extensions:
                allImageNames.append(fileName)
                allImageFullPath.append(fullPath)
                allImageNamesWithExtension.append(f)
        elif recursive:
            namesInChildDir, allImageFullPathInChildDir, allImageNamesWithExtension = getAllImageInfo(fullPath, extensions, recursive)
            allImageNames += namesInChildDir
            allImageFullPath += allImageFullPathInChildDir
            allImageNamesWithExtension += allImageNamesWithExtension
    return allImageNames, allImageFullPath, allImageNamesWithExtension

def save2File(object, filePath):
    import pickle
    with open(filePath, 'wb') as f: pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
def loadFromeFile(filePath):
    import pickle
    if not path.isfile(filePath):
        warnings.warn("file not find at [{0}]".format(filePath))
        return None
    else:
        with open(filePath, 'rb') as f: return pickle.load(f)

def strProgress(current, total, displayBlockCnt = 50):
    import math
    current = total if current > total else current
    completedBlockCnt = math.floor(current / total * displayBlockCnt)
    resultStr = '['
    for i in range(completedBlockCnt): resultStr += '#'
    for i in range(displayBlockCnt - completedBlockCnt): resultStr += '.'
    return resultStr + ']'