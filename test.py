import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json

SERIALEXE = "serial.exe"
OMPEXE = "omp.exe"
CUDAEXE = "cuda.exe"

EXE_LOCATION = r"exe/"

TXT_LOCATION = r"resource/timetaken/"

# number of Loops 
N = 10

# opencv path for MinGW 
OPEN_CV_INCLUDE = r"C:/opencv-mingw/install/include"
OPEN_LIB_LOCATION = r"C:/opencv-mingw/install/x64/mingw/lib"
OPEN_BIN_LOCATION = r"C:/opencv-mingw/install/x64/mingw/bin"

# update path to let the system can find the needed opencv dll file from the location 
newPath = OPEN_BIN_LOCATION + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = newPath

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

def compileSerial() : 
    # library use 
    libs = [
        "-lopencv_core4100",
        "-lopencv_imgproc4100",
        "-lopencv_highgui4100",
        "-lopencv_imgcodecs4100",
    ]

    serialBase = r"Serial/Serial-ImageEnhancementWithGaussianHPFilter/Serial-ImageEnhancementWithGaussianHPFilter/"

    # all the file need to be compile together with main, because have include 
    sourceFiles = [
        f"{serialBase}main.cpp", 
        f"{serialBase}FastFourierTransform.cpp",
        f"{serialBase}InverseFastFourierTransform.cpp",
        f"{serialBase}GaussianHPFilter.cpp",
        f"{serialBase}Utils.cpp",
        f"{serialBase}convertGrayscale.cpp",
    ]

    # construct a command for g++ compiler 
    compileCmd = [
        "g++", 
        "-std=c++17",               # c++ version
        f"-I{OPEN_CV_INCLUDE}",     # the include file location of opencv
        f"-L{OPEN_LIB_LOCATION}",   # library location of opencv
        *sourceFiles,               # all the file want to compile
        *libs,                      # all the library use 
        "-o", f"{EXE_LOCATION}{SERIALEXE}"
    ]

    print("Serial code compilation command :")
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

def compileOMP() :
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
        f"-I{OPEN_CV_INCLUDE}",       # the include file location of opencv
        f"-L{OPEN_LIB_LOCATION}",   # library location of opencv
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

def compileCUDA() : 
    # -------------------- change this to nvcc path -----------------
    NVCC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe"

    # CUDA paths
    CUDA_INCLUDE = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"
    CUDA_LIB_LOCATION = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"

    # opencv 
    CUDA_OPEN_CV_INCLUDE = "C:/opencv-mingw/install/include"
    CUDA_OPEN_CV_LIB_LOCATION = "C:/opencv-mingw/install/lib"

    # library use 
    cudaLibs = [
        "-lcudart", # cuda runtime library 
    ]

    opencvLibs = [
        "-lopencv_core4100",
        "-lopencv_imgproc4100",
        "-lopencv_highgui4100",
        "-lopencv_imgcodecs4100",
    ]

    cudaBase = r"CUDA/CUDA-ImageEnhancementWithGaussianHPFilter/CUDA-ImageEnhancementWithGaussianHPFilter/"

    # all the file need to be compile together with main, because have include 
    sourceFiles = [
        f"{cudaBase}CUDA-main.cpp", 
        f"{cudaBase}CUDA-FastFourierTransform.cpp",
        f"{cudaBase}CUDA-GaussianHPFilter.cpp",
        f"{cudaBase}CUDA-Utils.cpp",
    ]

    # construct a command for g++ compiler 
    compileCmd = [
        NVCC, 
        "-std=c++17",               # c++ version
        "-O2",
        "-arch=sm_75",              
        f"-I{CUDA_OPEN_CV_INCLUDE}",       # the include file location of opencv
        f"-I{CUDA_INCLUDE}",
        f"-L{CUDA_OPEN_CV_LIB_LOCATION}",   # library location of opencv
        f"-L{CUDA_LIB_LOCATION}",
        *cudaLibs,
        *opencvLibs,
        *sourceFiles,               # all the file want to compile
        "-o", f"{EXE_LOCATION}{CUDAEXE}"
    ]

    print("CUDA code compilation command :")
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

def runAll() :
    # run the exe
    executeExe(f"{EXE_LOCATION}{SERIALEXE}")
    executeExe(f"{EXE_LOCATION}{OMPEXE}")
    executeExe(f"{EXE_LOCATION}{CUDAEXE}")

def readByLine(filePath) :
    with open(filePath, 'r') as file:
        return [line.strip() for line in file]

def createRuntimeDF(platform, imName) :
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

def drawLineGraph(df, imName) :
    dfImage = df[df["imageName"] == imName]
    
    plt.figure(figsize=(9,6))
    
    # melt the dataframe so columns 1 - 10 become a single 'X' col, time usage go into 'Y'
    dfMelted = dfImage.melt(
        id_vars=['imageName', 'platform'],
        var_name='X',          # was 1 - 10
        value_name='Y'         # time usage
    )
    
    sns.lineplot(
        data=dfMelted,
        x='X',
        y='Y',
        hue='platform',
        marker='o'
    )

    # Optionally label points
    for i in range(len(dfMelted)):
        xVal = dfMelted.loc[i, 'X']
        yVal = dfMelted.loc[i, 'Y']
        plt.text(xVal+0.1, yVal, str(yVal), fontsize=8)
    
    plt.title("Line Plot: Serial vs. OMP vs. CUDA Runtimes")
    plt.xlabel("Column (1 to 10)")
    plt.ylabel("Time Used")
    plt.legend(title="Platform")
    plt.savefig(f'{imName}_runtime.png')

def calculateAverage(filePath) :
    result = readByLine(filePath=filePath)
    total = 0
    for i in result : 
        total += int(i)
    
    # divide by N
    return math.ceil((total / N) * 10) / 10

def calculateSpeedUp(numThreads) : 
    # calculate Serial 
    

def resultGenerate() :
    # list of platform name and image Name
    # platform = ["serial", "omp", "cuda"]
    platform = ["serial", "omp"]
    imName = ["lena", "wolf"]
    overall = "overall"

    basePath = r"resource/outputForDisplay/"

    result = {}

    for p in platform : 
        for im in imName : 
            # calculate overall result first
            timeOverallPath = f"{basePath}{p}_{im}_{overall}.txt"
            avgOverall = calculateAverage(timeOverallPath)
            result["average"][f"{p}_{m}_{overall}"] = avgOverall

            timeParallelPath = f"{basePath}{p}_{im}.txt"
            avgParallel = calculateAverage(timeOverallPath)
            result["average"][f"{p}_{m}"] = avgParallel
    
    # speed up


    # store into json file 
    with open(f"{basePath}performanceResult.json", "w") as file :
        json.dump(result, file, indent=4)



if __name__ == "__main__" :
    compileCUDA()
    executeExe(f"{EXE_LOCATION}{CUDAEXE}")
