#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Custom deleter for TensorRT objects
struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const {
        delete p;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// TensorRT logger
class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
};

// Buffer info for a single engine I/O tensor
struct BufferInfo {
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t byte_size;
    void* device_ptr = nullptr;  // GPU memory
};

// Wraps a single TensorRT engine: load, allocate buffers, run inference
class TrtEngine {
  public:
    TrtEngine() = default;
    ~TrtEngine();

    // Non-copyable
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Load a serialized .trt engine file
    bool load(const std::string& engine_path);

    // Free GPU resources (engine + buffers)
    void unload();

    // Whether engine is loaded
    bool is_loaded() const { return context_ != nullptr; }

    // Get buffer info for inputs/outputs
    const std::vector<BufferInfo>& inputs() const { return inputs_; }
    const std::vector<BufferInfo>& outputs() const { return outputs_; }

    // Copy data to an input buffer (host → device)
    bool set_input(const std::string& name, const void* host_data, size_t bytes);

    // Run inference (synchronous)
    bool infer();

    // Copy output data from device → host
    bool get_output(const std::string& name, void* host_data, size_t bytes);

  private:
    bool allocate_buffers();
    void free_buffers();

    static size_t dtype_size(nvinfer1::DataType dt);
    static size_t volume(const nvinfer1::Dims& dims);

    TrtLogger logger_;
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;

    std::vector<BufferInfo> inputs_;
    std::vector<BufferInfo> outputs_;
    std::unordered_map<std::string, BufferInfo*> buffer_map_;

    cudaStream_t stream_ = nullptr;
};
