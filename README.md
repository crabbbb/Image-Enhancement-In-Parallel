# Image-Enhancement-In-Parallel
Distributed System and Parallel Computing Course Assignment, which required to use CUDA and OpenMP to compare the code performance between Serial, Parallel in CPU and Parallel in GPU

- Image Enhancement with using Gaussian High Pass Filter
- Transfer Image from Spatial domain to Frequency Domain with using Fast-Fourier Transform ( FFT ) 


# TODO
- [ ] Learn How to use Named Pipe
- [ ] Learn How to Display image on colab 
- [ ] FFT
- [ ] IFFT
- [ ] Gaussian

# OpenCV Installation Step 
1. Download OpenCV Window .exe file [[Link]](https://github.com/opencv/opencv/releases/tag/4.10.0)<br/>
    ![alt text](readmeImage/image.png)
2. Execute .exe extract the opencv folder to <code><b>C:\ </b></code>
3. Window open edit the system environment variable > Environment Variable > Advannced > Under System Variable find Path > Double Click > Add a new Path <code>**C:\opencv\build\x64\vc16\bin**</code> > Click OK and close the app
4. At Visual Studio 2022, Right Click Project > Properties 
    - C/C++ > General > Additional Include Directories > Enter <code>**C:\opencv\build\include**</code>
    - Linker > General > Additional Library Directories > Enter <code>**C:\opencv\build\x64\vc16\lib**</code>
    - Linker > Input > Additional Dependencies > Enter <code>**opencv_world4100d.lib;opencv_world4100.lib;**</code>
        - *d.lib is for debug purpose 
        - .lib is for release use 
5. Execute the code below, console should be able to prompt the "Hello World!"
    ```
    #include <opencv2/opencv.hpp>
    #include <iostream>

    int main()
    {
        std::string image_path = "C:\\path-to-your-resource\\resource\\raw\\lena.jpeg";
        cv::Mat rgb_image = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::imshow("RGB Image", rgb_image);
        // must put, else will have error 
        cv::waitKey(0);
    }
    ```
6. If Debug facing the error below, restart the Visual Studio 2022 <br/>
    ![alt text](readmeImage/image-1.png)

# Project Setup and Usage Guide

This part provides step-by-step instructions for setting up a Python environment, installing the required dependencies, and running the project's main script.

## 1. Create and Activate a Python Environment

You can choose to use either a virtual environment (`venv`) or Conda.

### Option A: Virtualenv / venv

1. Create a new environment with Python 3.12.8:
   ```bash
   python -m venv myenv
   ```

2. Activate the environment:
   - On Unix/macOS:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```cmd
     myenv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Conda

1. Create a new environment with Python 3.12.8:
   ```bash
   conda create --name myenv python=3.12.8
   ```

2. Activate the environment:
   ```bash
   conda activate myenv
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Navigate to the Project Directory

Change into the project directory:
```bash
cd path/to/project
```

## 3. Run the Script

Once the environment is activated and dependencies are installed, run the script:
```bash
python run_file_and_notebook.py
```

## 4. Deactivating the Environment (Optional)

- **venv**:
  ```bash
  deactivate
  ```

- **Conda**:
  ```bash
  conda deactivate
  ```

---

## Troubleshooting

- Ensure that you are using **Python 3.12.8**.
- Double-check that the `requirements.txt` file is in the project directory before installing dependencies.
- If you encounter permission issues on Unix-based systems, you may need to adjust file permissions (e.g., `chmod +x`) or use `sudo` cautiously (though it is not recommended for Python environments).
