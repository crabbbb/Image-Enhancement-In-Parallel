import os, sys, subprocess
import threading
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt

def tee_pipe(pipe, out):
    for line in pipe:
        #print(line.decode('utf-8'), end='')
        #print(line.decode('utf-8'), end='',  file=out)
        out.write(line.decode('utf-8'))

def execute(filename, *args, pipe_name='/tmp/my_pipe'):
    if not os.path.exists(pipe_name):
        os.mkfifo(pipe_name)

    # Start the subprocess. The -u option is to force the Python subprocess
    # to flush its output everytime it prints.
    proc = subprocess.Popen(
            [filename, '-p', pipe_name, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
    )
    # Create threads to capture and print stdout and stderr
    t1 = threading.Thread(target=tee_pipe, args=(proc.stdout, sys.stdout))
    t2 = threading.Thread(target=tee_pipe, args=(proc.stderr, sys.stdout))
    t1.start()
    t2.start()

    with open(pipe_name, "rb") as pipe:
        while True:
            # Read the image size from the pipe
            # The 1st 4 byte is column size
            # The 2nd 4 byte is row size
            # The 3rd 4 byte is channel size
            img_header = pipe.read(12)
            if not img_header:
                break
            image_size = np.frombuffer(img_header, dtype=np.uint32)
            # Read the image data for all channels
            frame_data = pipe.read(image_size[0] * image_size[1] * image_size[2])
            if not frame_data:
                break
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((image_size[1], image_size[0], image_size[2]))
            # Display the received frame
            cv2_imshow(frame)

            # # check if the image is grayscale or color
            # if frame.shape[2] == 1:  
            #     # grayscale
            #     plt.imshow(frame, cmap='gray')
            # else: 
            #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  

            # plt.axis('off')
            # plt.show()

    proc.wait()                 # Wait for subprocess to exit
    os.remove(pipe_name)        # Clean up the named pipe
    cv2.destroyAllWindows()