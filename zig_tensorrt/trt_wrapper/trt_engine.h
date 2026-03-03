// TensorRT engine wrapper — copied from tensor_rt/src/engine.h with the class
// renamed to TrtEngineImpl to avoid collision with the C wrapper's opaque type.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
};

struct BufferInfo {
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t byte_size;
    void* device_ptr = nullptr;
};

class TrtEngineImpl {
  public:
    TrtEngineImpl() = default;
    ~TrtEngineImpl();

    TrtEngineImpl(const TrtEngineImpl&) = delete;
    TrtEngineImpl& operator=(const TrtEngineImpl&) = delete;

    bool load(const std::string& engine_path);
    void unload();
    bool is_loaded() const { return context_ != nullptr; }

    const std::vector<BufferInfo>& inputs() const { return inputs_; }
    const std::vector<BufferInfo>& outputs() const { return outputs_; }

    bool set_input_shape(const std::string& name, const nvinfer1::Dims& dims);
    bool set_input(const std::string& name, const void* host_data, size_t bytes);
    bool infer();
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
