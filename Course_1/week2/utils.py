from os import listdir
from os import path

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