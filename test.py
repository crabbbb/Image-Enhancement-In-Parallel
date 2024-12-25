import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt

SERIALEXE = "serial.exe"
OMPEXE = "omp.exe"
CUDAEXE = "cuda.exe"

EXE_LOCATION = r"exe/"

TXT_LOCATION = r"resource/timetaken/"

# number of Loops 
N = 10

def executeExe(exeLocation) : 
    runCmd = [f"{exeLocation}"]
    print(f"\nRunning {exeLocation}...")

    try:
        result = subprocess.run(
            runCmd,
            capture_output=True,
            text=True,
            check=True
        )

        print("Platform finished successfully")

        if result.stdout:
            print("Platform output : ", result.stdout)
        if result.stderr:
            print("Platform error/warning output : ", result.stderr)
        
    except subprocess.CalledProcessError as e:
        print("platform failed with return code : ", e.returncode)
        print("platform error output : ", e.stderr)

def runOMP():
    # opencv path 
    opencvInclude = r"C:/opencv-mingw/install/include"
    opencvLibLocation = r"C:/opencv-mingw/install/x64/mingw/lib"
    opencvBinLocation = r"C:/opencv-mingw/install/x64/mingw/bin"

    # library use 
    libs = [
        "-lopencv_core4100",
        "-lopencv_imgproc4100",
        "-lopencv_highgui4100",
        "-lopencv_imgcodecs4100",
    ]

    ompBase = r"OpenMP/ImageEnhancementWithGaussianHPFilter/ImageEnhancementWithGaussianHPFilter/"

    # all the file need to be compile together with main, because have include 
    sourceFiles = [
        f"{ompBase}main.cpp", 
        f"{ompBase}FastFourierTransform.cpp",
        f"{ompBase}GaussianHPFilter.cpp",
        f"{ompBase}Utils.cpp",
        f"{ompBase}convertGrayscale.cpp"
    ]

    # construct a command for g++ compiler 
    compileCmd = [
        "g++", 
        "-std=c++17",               # c++ version
        "-fopenmp",                 # enable omp
        f"-I{opencvInclude}",       # the include file location of opencv
        f"-L{opencvLibLocation}",   # library location of opencv
        *sourceFiles,               # all the file want to compile
        *libs,                      # all the library use 
        "-o", f"{EXE_LOCATION}{OMPEXE}"
    ]

    print("OpenMP code compilation command :")
    print(" ".join(compileCmd))

    # run the compile command via subprocess
    try:
        result = subprocess.run(
            compileCmd,
            capture_output=True,
            text=True,
            check=True
        )

        print("Compilation success")

        if result.stdout:
            # if success
            print(result.stdout)
        if result.stderr:
            # if error
            print("Warnings/Errors :", result.stderr)

    except subprocess.CalledProcessError as e:
        print("Compilation failed with return code : ", e.returncode)
        print("Compiler error output : ", e.stderr)
        exit()

    # update path to let the system can find the needed opencv dll file from the location 
    newPath = opencvBinLocation + os.pathsep + os.environ["PATH"]
    os.environ["PATH"] = newPath

    # run the exe
    executeExe(f"{EXE_LOCATION}{OMPEXE}")

def readByLine(filePath) :
    with open(filePath, 'r') as file:
        return [line.strip() for line in file]

def readAllTxtFile() :
    # list of platform name and image Name
    # platform = ["serial", "omp", "cuda"]
    platform = ["omp"]
    imName = ["lena", "wolf"]

    # dataframe 0,1,2,...,10, imageName, platform (col)
    columns = []
    
    for i in range(1, 11) :
        columns.append(i)

    columns.append("imageName")
    columns.append("platform")

    # row serial, omp, cuda 
    df = pd.DataFrame(columns=columns)

    # read data from txt file 
    i = 0
    for p in platform :
        for im in imName :
            # get txt fileName
            filePath = f"{TXT_LOCATION}{p}_{im}.txt"

            # read all the data add into dataframe 
            df.loc[i] = readByLine(filePath) + [im, p]
            i += 1
    
    return df

if __name__ == "__main__":
    df = readAllTxtFile()