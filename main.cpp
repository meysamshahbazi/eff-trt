#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>

using namespace std;
// utilities ----------------------------------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// get classes names
std::vector<std::string> getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

// preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims)
{
    // read input image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    // normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

// post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    auto classes = getClassNames("../imagenet_classes.txt");

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    // float sum = 1.0f;
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims) * batch_size);
    std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch


    unique_ptr<nvinfer1::IHostMemory,TRTDestroy> serializedModel{builder->buildSerializedNetwork(*network, *config)};

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);


    engine.reset(runtime->deserializeCudaEngine( serializedModel->data(), serializedModel->size()) );
    context.reset(engine->createExecutionContext());
    return;
}

// main pipeline ------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx image.jpg\n";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    int batch_size = 1;

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context);

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }

    // auto t1=std::chrono::system_clock::now();
    int64 tick_counter = 0;
    int frame_idx = 1;
    
    // preprocess input data
    // preprocessImage(image_path, (float *) buffers[0], input_dims[0]);
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return -1;
    }

    cv::resize(frame,frame,cv::Size(224,224));
    int data_idx = 0;
    int q_size = 224;
    float blob[1*3*q_size*q_size];

    //     resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    // cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    // cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    for (int i = 0; i < frame.rows; ++i)
    {
        uchar* pixel = frame.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < frame.cols; ++j)
        {
            blob[data_idx] = ( (*pixel++)/255.f-0.485f)/0.229f;
            blob[data_idx+q_size*q_size] = ( (*pixel++)/255.f-0.456f)/0.224f;
            blob[data_idx+2*q_size*q_size] = ( (*pixel++)/255.f-0.406f)/0.225f;
            data_idx++;
        }
    }
    int64 t1 = cv::getTickCount();
    cudaMemcpyAsync(buffers[0], blob, 1*3*q_size*q_size*sizeof(float), cudaMemcpyHostToDevice);

    // inference
    
    // context->executeV2(buffers.data());
    context->enqueueV2(buffers.data(), 0, nullptr);
    cudaStreamSynchronize(0);
    // postprocess results
    int64 t2 = cv::getTickCount();
    postprocessResults((float *) buffers[1], output_dims[0], batch_size);

    
    tick_counter += t2 - t1;
    // auto t2 = std::chrono::system_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    // float fps= 1000/(duration*1e-3);

    // std::cout << "Network FPS = " << std::to_string(fps) << std::endl;

    cout << "FPS: " << ((double)(1)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }
    return 0;
}

