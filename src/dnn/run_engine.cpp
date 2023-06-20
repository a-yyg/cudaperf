#include <cstdlib>
#include <cstdio>
#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <fstream>
#include <iostream>
#include <memory>

#include <numeric>

#include <opencv2/opencv.hpp>

#include <chrono>

#include "common/timelogger.hpp"

YuriPerf::TimeLogger g_logger;

#define INPUT_NAME "data"
#define OUTPUT_NAME "resnetv24_dense0_fwd"

#define BENCHMARK 1

#define CHECK_CUDA(call) { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        std::exit(1); \
    } \
}

using namespace nvinfer1;
using namespace nvonnxparser;

void print_usage(char* argv[]) {
    printf("Program to run a TensorRT engine for image classification.\n");
    printf("Usage: %s <trt model> <image folder> <batch size> <iterations>\n", argv[0]);
}

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            printf("%s\n", msg);
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
 
template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;


int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv);
        return 1;
    }

    std::vector<char> trtModelStream_;
    size_t size{0};

    std::ifstream file(argv[1], std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << argv[1] << " failed" << std::endl;
        return -1;
    }

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream_.resize(size);
    file.read(trtModelStream_.data(), size);
    file.close();
    
    TRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger)};
    if (runtime == nullptr) {
        std::cerr << "createInferRuntime failed" << std::endl;
        return -1;
    }

    TRTUniquePtr<ICudaEngine> engine{runtime->deserializeCudaEngine(trtModelStream_.data(), size)};

    TRTUniquePtr<IExecutionContext> context{engine->createExecutionContext()};

    // read image
    std::string image_folder = argv[2];
    std::vector<std::string> image_names;
    std::ifstream infile(image_folder + "/images.txt");
    std::string line;
    while (std::getline(infile, line)) {
        image_names.push_back(line);
    }
    infile.close();

    // read label
    std::vector<std::string> labels;
    infile.open(image_folder + "/labels.txt");
    while (std::getline(infile, line)) {
        labels.push_back(line);
    }

    // use opencv to read image
    cv::Mat img;
    std::vector<uchar> prob(1000);

    uchar* d_input{nullptr};
    uchar* d_output{nullptr};
    CHECK_CUDA(cudaMalloc((void**)&d_input, 3 * 224 * 224 * sizeof(uchar)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, 1000 * sizeof(uchar)));
    void* const buffers[] = {
        reinterpret_cast<void*>(d_input),
        reinterpret_cast<void*>(d_output)
    };

    int iterations = argc == 5 ? atoi(argv[4]) : 1;
#if (BENCHMARK == 1)
    g_logger.setActive(true);
#else
    g_logger.setActive(false);
#endif

    for (uint i = 0; i < image_names.size(); ++i) {
        img = cv::imread(image_folder + "/" + image_names[i]);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << image_names[i] << std::endl;
            return -1;
        }

        cv::resize(img, img, cv::Size(224, 224));

        imwrite("input.jpg", img);
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // img.convertTo(img, CV_32FC3);
        // img = (img - 127.5) / 127.5;

        double t_in = 0.0;
        double t_proc = 0.0;
        double t_out = 0.0;
        for (int j = 0; j < iterations; ++j) {
#if (BENCHMARK == 1)
            g_logger.startRecording("h2d");
#endif

            CHECK_CUDA(cudaMemcpy(d_input, img.data, 3 * 224 * 224 * sizeof(uchar), cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();

#if (BENCHMARK == 1)
            g_logger.stopRecording();
            g_logger.startRecording("infer");
#endif

            // run inference
            context->execute(1, buffers);
            cudaDeviceSynchronize();

#if (BENCHMARK == 1)
            g_logger.stopRecording();
            g_logger.startRecording("d2h");
#endif

            // postprocess
            CHECK_CUDA(cudaMemcpy(prob.data(), d_output, 1000 * sizeof(uchar), cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();

#if (BENCHMARK == 1)
            g_logger.stopRecording();
#endif
        }

        // std::cout << "input time: " << t_in / iterations << std::endl;
        // std::cout << "process time: " << t_proc / iterations << std::endl;
        // std::cout << "output time: " << t_out / iterations << std::endl;
#if (BENCHMARK == 1)
        g_logger.print();
        g_logger.writeCSV("resnet50.csv");
#endif

        for (int j = 0; j < 10; ++j) {
            std::cout << labels[j] << ": " << prob[j] << std::endl;
        }

        // calculate softmax
        std::transform(prob.begin(), prob.end(), prob.begin(), [](float val) { return std::exp(val); });
        auto sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
                
        std::transform(prob.begin(), prob.end(), prob.begin(), [sum](float val) { return val / sum; });

        // find max prob
        auto max = std::max_element(prob.begin(), prob.end());
        auto index = std::distance(prob.begin(), max);

        std::cout << "image: " << image_names[i] << ", label: " << labels[index] << ", prob: " << *max << std::endl;
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
