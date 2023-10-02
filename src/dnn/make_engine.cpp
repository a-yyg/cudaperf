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
    printf("Usage: %s <onnx model> <calibration data> <batch size> <trt engine name> <input size> <int8/fp32>\n", argv[0]);
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

// #define INPUT_SIZE 640
#define INPUT_SIZE 416

cv::Mat static_resize(cv::Mat& img, int width = INPUT_SIZE, int height = INPUT_SIZE) {
    float r = std::min(width / (img.cols*1.0), height / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
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
    DirectoryBatchStream(int batchSize, int maxBatches, std::string dataDir, size_t imageSize = INPUT_SIZE)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDataDir(dataDir)
        , mImageSize(imageSize)
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
        return &mData[mCurrentBatch * mBatchSize * 3 * mImageSize * mImageSize];
    }

    float* getLabels() override
    {
        return nullptr;
    }

    int getBatchesRead() const override { return mCurrentBatch; }

    int getBatchSize() const override { return mBatchSize; }

    nvinfer1::Dims getDims() const override { return nvinfer1::Dims3(3, mImageSize, mImageSize); }
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
        mData.resize(mFileNames.size() * 3 * mImageSize * mImageSize);
        // std::vector<float> data(mBatchSize * 3 * mImageSize * mImageSize);

        for (int i = 0; i < mFileNames.size(); i++)
        {
            std::string fileName = mFileNames[i];
            cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);

            if (image.empty())
            {
                std::cerr << "Could not read image " << fileName << std::endl;
                return;
            }

            cv::Mat resized_image = static_resize(image, mImageSize, mImageSize);
            blobFromImage(resized_image, mData.data() + i * 3 * mImageSize * mImageSize);
        }
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mNbBatches{0};
    int mCurrentBatch{0};
    bool mShuffle{true};
    size_t mImageSize{0};
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
    if (argc < 6) {
        print_usage(argv);
        return 1;
    }

    std::string model_name = argv[1];
    std::string calibration_data = argv[2];
    int batch_size = std::atoi(argv[3]);
    std::string engine_name = argv[4];
    int input_size = std::atoi(argv[5]);
    std::string mode = argv[6];

    if (mode != "int8" && mode != "fp32") {
        print_usage(argv);
        return 1;
    }

    TRTUniquePtr<IBuilder> builder {createInferBuilder(gLogger)};
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<INetworkDefinition> network {builder->createNetworkV2(flag)};

    TRTUniquePtr<IBuilderConfig> config {builder->createBuilderConfig()};

    TRTUniquePtr<IParser> parser {createParser(*network, gLogger)};
    parser->parseFromFile(model_name.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        printf("%s\n", parser->getError(i)->desc());
    }

    config->setMaxWorkspaceSize(1 << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    if (mode == "int8") {
        if (!builder->platformHasFastInt8()) {
            printf("Platform doesn't support int8 mode\n");
            return 1;
        }

        config->setFlag(BuilderFlag::kINT8); // force int8 mode

        DirectoryBatchStream calibrationStream(batch_size, 0, calibration_data, input_size);

        TRTUniquePtr<Int8Calibrator<DirectoryBatchStream, IInt8MinMaxCalibrator>> calibrator {
            new Int8Calibrator<DirectoryBatchStream, IInt8MinMaxCalibrator>(calibrationStream, 0, "calibration.table", "images")};

        config->setInt8Calibrator(calibrator.get());
    }

    builder->setMaxBatchSize(batch_size);

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
