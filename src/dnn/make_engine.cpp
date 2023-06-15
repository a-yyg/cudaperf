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

#define INPUT_NAME "data"
#define OUTPUT_NAME "resnetv24_dense0_fwd"

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
    printf("Program to generate TensorRT engine from ONNX model.\n");
    printf("Usage: %s <onnx model> <batch size> <trt engine name>\n", argv[0]);
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

    std::string model_name = argv[1];
    int batch_size = std::atoi(argv[2]);
    std::string engine_name = argv[3];

    TRTUniquePtr<IBuilder> builder {createInferBuilder(gLogger)};
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<INetworkDefinition> network {builder->createNetworkV2(flag)};

    TRTUniquePtr<IBuilderConfig> config {builder->createBuilderConfig()};

    TRTUniquePtr<IParser> parser {createParser(*network, gLogger)};
    parser->parseFromFile(model_name.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        printf("%s\n", parser->getError(i)->desc());
    }

    if (!builder->platformHasFastInt8()) {
        printf("Platform doesn't support int8 mode\n");
        return 1;
    }

    config->setMaxWorkspaceSize(1 << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kINT8); // force int8 mode
    config->setInt8Calibrator(nullptr); // use default calibrator

    IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("data", OptProfileSelector::kMIN, Dims4(batch_size, 3, 224, 224));
    profile->setDimensions("data", OptProfileSelector::kOPT, Dims4(batch_size, 3, 224, 224));
    profile->setDimensions("data", OptProfileSelector::kMAX, Dims4(batch_size, 3, 224, 224));
    profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kMIN, Dims2(batch_size, 1000));
    profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kOPT, Dims2(batch_size, 1000));
    profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kMAX, Dims2(batch_size, 1000));

    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(batch_size);

    // setup network layer precision

    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto layer = network->getLayer(i);

        switch(layer->getType()) {
        case LayerType::kCONSTANT:
        case LayerType::kCONCATENATION:
        case LayerType::kSHAPE:
            break;
        default:
            layer->setPrecision(DataType::kINT8);
            break;
        }

        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            if (layer->getOutput(j)->isExecutionTensor()) {
                layer->setOutputType(j, DataType::kINT8);
            }
        }
    }

    // set custom dynamic range
    std::ifstream input("resnet50_per_tensor_dynamic_range.txt");

    std::map<std::string, float> ranges;

    {
        std::string line;
        char delim = ':';
        while (std::getline(input, line)) {
            std::istringstream iss(line);
            std::string token;
            std::getline(iss, token, delim);
            std::string name = token;
            std::getline(iss, token, delim);
            float max = std::stof(token);
            ranges[name] = max;
        }
    }

    input.close();

    for (int i = 0; i < network->getNbInputs(); ++i) {
        std::string name = network->getInput(i)->getName();
        if (ranges.find(name) != ranges.end()) {
            float max = ranges[name];
            float min = -max;
            if (!network->getInput(i)->setDynamicRange(min, max)) {
                printf("set dynamic range failed\n");
                return 1;
            }
        } else {
            printf("can't find dynamic range for %s\n", name.c_str());
            // return 1;
        }
    }

    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto lyr = network->getLayer(i);
        for (int j = 0, e = lyr->getNbOutputs(); j < e; ++j) {
            std::string name = lyr->getOutput(j)->getName();
            if (ranges.find(name) != ranges.end()) {
                float max = ranges[name];
                float min = -max;
                if (!lyr->getOutput(j)->setDynamicRange(min, max)) {
                    printf("set dynamic range failed\n");
                    return 1;
                }
            } else if (lyr->getType() == LayerType::kCONSTANT) {
                IConstantLayer* const_lyr = static_cast<IConstantLayer*>(lyr);
                auto wts = const_lyr->getWeights();
                double max = std::numeric_limits<double>::min();
                printf("computing dynamic range for %s\n", name.c_str());
                for (int64_t wb = 0, we = wts.count; wb < we; ++wb) {
                    double val{};
                    switch (wts.type) {
                    case DataType::kFLOAT: val = static_cast<const float*>(wts.values)[wb]; break;
                    case DataType::kBOOL: val = static_cast<const bool*>(wts.values)[wb]; break;
                    case DataType::kINT8: val = static_cast<const int8_t*>(wts.values)[wb]; break;
                    // case DataType::kHALF: val = static_cast<const half*>(wts.values)[wb]; break;
                    case DataType::kINT32: val = static_cast<const int32_t*>(wts.values)[wb]; break;
                    // case DataType::kUINT8: val = static_cast<uint8_t const*>(wts.values)[wb]; break;
                    default: printf("unsupported weight type\n"); return 1;
                    }
                    max = std::max(max, std::abs(val));
                }

                if (!lyr->getOutput(j)->setDynamicRange(-max, max)) {
                    printf("set dynamic range failed\n");
                    return 1;
                }
            } else {
                printf("can't find dynamic range for %s\n", name.c_str());
                // return 1;
            }
        }
    }

    TRTUniquePtr<IHostMemory> plan {builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        printf("build engine failed\n");
        return 1;
    }

    TRTUniquePtr<IRuntime> runtime {createInferRuntime(gLogger)};
    if (!runtime) {
        printf("create runtime failed\n");
        return 1;
    }

    TRTUniquePtr<ICudaEngine> engine {runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr)};
    if (!engine) {
        printf("deserialize engine failed\n");
        return 1;
    }

    TRTUniquePtr<IExecutionContext> context {engine->createExecutionContext()};

    FILE* fp = fopen(engine_name.c_str(), "wb");
    fwrite(plan->data(), 1, plan->size(), fp);

    fclose(fp);

    return 0;
}