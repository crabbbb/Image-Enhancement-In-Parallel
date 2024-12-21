#include "cv_pipe.h"

static int fd = -1;

int open_named_pipe(char *pipe_name) {
    fd = open(pipe_name, O_WRONLY); // open a pipe
    if(fd < 0) {
        std::cerr << "Error: failed to open the named pipe: "
                    << pipe_name << std::endl;
    }
    return fd;
}

int cv_imshow(cv::Mat &image) {
    if(fd < 0) {
        std::cerr << "Error: no named pipe available." << std::endl;
        return -1;
    }
    // Send image size as a header
    int img_size[3] = {image.cols, image.rows, image.channels()};
    write(fd, img_size, sizeof(img_size));
    // Send the image data
    write(fd, image.data, image.total() * image.elemSize());
    return 0;
}

int init_cv_pipe_comm(int argc, char *argv[], bool verbose) {
    int c;
    char *pipe_path = NULL;

    if(verbose) {
        // Print all input arguments
        for(int i = 0; i < argc; i++) {
            std::cout << "[" << i << "] " << argv[i] << std::endl;
        }
    }
    //opterr = 0;       // Do not print error to stderr
    while ((c = getopt(argc, argv, ":p:")) != -1) {
        switch(c) {
            case 'p':
                pipe_path = optarg;
                break;
            case ':':
                std::cerr << "Error: option -" << static_cast<char>(optopt)
                            << " requires an argument.\n";
                return -1;
            case '?':
                // Ignore all unknown options; let the main program handles it.
                break;
        }
    }
    if(!pipe_path) {
        std::cerr << "Error: expect a pipe name but none found. Try the "
                    << "following:\n\t" << argv[0] << " -p my_pipe\n";
        return -1;
    }

    fd = open_named_pipe(pipe_path);
    return fd;
}

int finalize_cv_pipe_comm() {
    close(fd);        // Close the write end of the pipe
    return 0;
}