#include <cstdlib>
#include <cstdio>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "BatchStream.h"
#include "EntropyCalibrator.h"

#include <dirent.h>

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
    printf("Usage: %s <onnx model> <calibration data> <batch size> <trt engine name>\n", argv[0]);
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
#if NV_TENSORRT_MAJOR < 8
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;
#else
using TRTUniquePtr = std::unique_ptr< T >;
#endif

#define INPUT_SIZE 640
#define INPUT_W 640
#define INPUT_H 640

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void blobFromImage(cv::Mat& img, float *blob){
    // float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    // return blob;
}

/* Class to load files from a directory and prepare them
   for use with Int8Calibrator.                          */
class DirectoryBatchStream : public IBatchStream
{
public:
    DirectoryBatchStream(int batchSize, int maxBatches, std::string dataDir)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDataDir(dataDir)
    {
        // Load all files into mFileNames
        readDataFiles(mDataDir);
    }

    void reset(int firstBatch) override
    {
        mCurrentBatch = firstBatch;
    }    

    bool next() override
    {
        if (mCurrentBatch >= mNbBatches - 1)
        {
            return false;
        }
        mCurrentBatch++;
        return true;
    }

    void skip(int skipCount) override
    {
        mCurrentBatch += skipCount;
    }

    float* getBatch() override
    {
        // Return pointer to mBatchSize images
        return &mData[mCurrentBatch * mBatchSize * 3 * INPUT_SIZE * INPUT_SIZE];
    }

    float* getLabels() override
    {
        return nullptr;
    }

    int getBatchesRead() const override { return mCurrentBatch; }

    int getBatchSize() const override { return mBatchSize; }

    nvinfer1::Dims getDims() const override { return nvinfer1::Dims3(3, INPUT_SIZE, INPUT_SIZE); }
private:

    void readDataFiles(std::string dirName)
    {
        // Get list of files in the directory
        std::vector<std::string> fileNames;
        DIR* dir = opendir(dirName.c_str());
        if (dir == nullptr)
        {
            std::cerr << "Error opening directory " << dirName << std::endl;
            return;
        }

        struct dirent* ent;
        while ((ent = readdir(dir)) != nullptr)
        {
            if (ent->d_type == DT_REG)
            {
                fileNames.push_back(ent->d_name);
            }
        }

        closedir(dir);

        // Sort files alphabetically
        std::sort(fileNames.begin(), fileNames.end());

        // Add full path to files
        for (auto& fileName : fileNames)
        {
            mFileNames.push_back(dirName + "/" + fileName);
        }

        // Calculate number of batches
        mNbBatches = mFileNames.size() / mBatchSize;
        if (mMaxBatches != 0)
        {
            mNbBatches = std::min(mNbBatches, mMaxBatches);
        }

        // Resize mFileNames to actual batch size
        mFileNames.resize(mNbBatches * mBatchSize);

        // Shuffle file names if required
        if (mShuffle)
        {
            std::random_shuffle(mFileNames.begin(), mFileNames.end());
        }

        // Print information
        std::cout << "Read " << mFileNames.size() << " images in " << mNbBatches << " batches of size " << mBatchSize << std::endl;

        // Open all files
        mData.resize(mFileNames.size() * 3 * INPUT_SIZE * INPUT_SIZE);
        // std::vector<float> data(mBatchSize * 3 * INPUT_SIZE * INPUT_SIZE);

        for (int i = 0; i < mFileNames.size(); i++)
        {
            std::string fileName = mFileNames[i];
            cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);

            if (image.empty())
            {
                std::cerr << "Could not read image " << fileName << std::endl;
                return;
            }

            cv::Mat resized_image = static_resize(image);
            blobFromImage(resized_image, mData.data() + i * 3 * INPUT_SIZE * INPUT_SIZE);
        }
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mNbBatches{0};
    int mCurrentBatch{0};
    bool mShuffle{true};
    std::string mDataDir;
    std::vector<std::string> mFileNames;
    std::vector<float> mData;
};

//! \class Int8MinMaxCalibrator
//!
//! \brief Implements Int8MinMaxCalibrator interface
//!  CalibrationAlgoType is kMINMAX_CALIBRATION
//!
template <typename TBatchStream, typename Calibrator>
class Int8Calibrator : public Calibrator
{
public:
    Int8Calibrator(
        TBatchStream stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }

    int getBatchSize() const noexcept override
    {
        return mImpl.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) noexcept override
    {
        return mImpl.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override
    {
        mImpl.writeCalibrationCache(cache, length);
    }

private:
    EntropyCalibratorImpl<TBatchStream> mImpl;
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv);
        return 1;
    }

    std::string model_name = argv[1];
    std::string calibration_data = argv[2];
    int batch_size = std::atoi(argv[3]);
    std::string engine_name = argv[4];

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
    // config->setInt8Calibrator(nullptr); // use default calibrator

    DirectoryBatchStream calibrationStream(batch_size, 0, calibration_data);

    // TRTUniquePtr<Int8EntropyCalibrator2<DirectoryBatchStream>> calibrator {
    //     new Int8EntropyCalibrator2<DirectoryBatchStream>(calibrationStream, 0, "calibration.table", "images")};

    TRTUniquePtr<Int8Calibrator<DirectoryBatchStream, IInt8MinMaxCalibrator>> calibrator {
        new Int8Calibrator<DirectoryBatchStream, IInt8MinMaxCalibrator>(calibrationStream, 0, "calibration.table", "images")};

    config->setInt8Calibrator(calibrator.get());

    // IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions("data", OptProfileSelector::kMIN, Dims4(batch_size, 3, 224, 224));
    // profile->setDimensions("data", OptProfileSelector::kOPT, Dims4(batch_size, 3, 224, 224));
    // profile->setDimensions("data", OptProfileSelector::kMAX, Dims4(batch_size, 3, 224, 224));
    // profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kMIN, Dims2(batch_size, 1000));
    // profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kOPT, Dims2(batch_size, 1000));
    // profile->setDimensions("resnetv24_dense0_fwd", OptProfileSelector::kMAX, Dims2(batch_size, 1000));

    // config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(batch_size);

#if 0
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
            printf("can't find dynamic range for %s\n, setting dummy values", name.c_str());
            if (!network->getInput(i)->setDynamicRange(-1, 1)) {
                printf("set dynamic range failed\n");
                return 1;
            }
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
                printf("can't find dynamic range for %s\n, setting dummy values", name.c_str());
                if (!lyr->getOutput(j)->setDynamicRange(-1, 1)) {
                    printf("set dynamic range failed\n");
                    return 1;
                }
                // return 1;
            }
        }
    }
#endif

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
