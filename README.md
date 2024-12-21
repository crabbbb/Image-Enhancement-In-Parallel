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
2. Execute .exe extract the opencv folder to C:\opencv
3. Window open edit the system environment variable > Environment Variable > Advannced > Under System Variable find Path > Double Click > Add a new Path "C:\opencv\build\x64\vc16\bin" > Click OK and close the app
4. At Visual Studio 2022, Right Click Project > Properties 
    - C/C++ > General > Additional Include Directories > Enter "C:\opencv\build\include"
    - Linker > General > Additional Library Directories > Enter "C:\opencv\build\x64\vc16\lib"
    - Linker > Input > Additional Dependencies > Enter "opencv_world4100d.lib;opencv_world4100.lib;"
        - *d.lib is for debug purpose 
        - .lib is for release use 
5. Execute the code below, console should be able to prompt the "Hello World!"
    ```
    #include <opencv2/opencv.hpp>
    #include <iostream>
    #include "convertGrayscale.hpp"

    int main()
    {
        std::string image_path = "C:\\path-to-your-resource\\resource\\raw\\lena.jpeg";
        cv::Mat rgb_image = cv::imread(image_path, cv::IMREAD_COLOR);
        std::cout << "Hello World!\n";
    }
    ```
6. If Debug facing the error below, restart the Visual Studio 2022 <br/>
    ![alt text](readmeImage/image-1.png)