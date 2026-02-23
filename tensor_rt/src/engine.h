#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TensorRT objects (IRuntime, ICudaEngine, etc.) use `delete` for cleanup,
// not a custom destructor, so we wrap them in unique_ptr with this deleter.
struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const {
        delete p;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// Routes TensorRT internal log messages to stderr.
class TrtLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
};

// Metadata for a single engine I/O tensor (input or output).
// Each tensor has a name, shape, data type, and a GPU buffer.
struct BufferInfo {
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    size_t byte_size;
    void* device_ptr = nullptr;
};

// Wraps a single TensorRT engine file (.trt).
//
// A TensorRT engine is a neural network that has been compiled and optimized
// for a specific GPU. It contains fused layers, selected CUDA kernels, and
// pre-allocated memory layouts — making inference much faster than running
// the original ONNX graph directly.
//
// Usage:
//   TrtEngine engine;
//   engine.load("model.trt");
//   engine.set_input_shape("input", dims);   // for dynamic-shape engines
//   engine.set_input("input", data, bytes);  // copy input to GPU
//   engine.infer();                           // run the network
//   engine.get_output("output", data, bytes); // copy result back to CPU
class TrtEngine {
  public:
    TrtEngine() = default;
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    bool load(const std::string& engine_path);
    void unload();
    bool is_loaded() const { return context_ != nullptr; }

    const std::vector<BufferInfo>& inputs() const { return inputs_; }
    const std::vector<BufferInfo>& outputs() const { return outputs_; }

    // Set the concrete shape for a dynamic input (engines built with
    // optimization profiles have -1 in some dims until this is called).
    bool set_input_shape(const std::string& name, const nvinfer1::Dims& dims);

    // Copy host data into the named input's GPU buffer.
    bool set_input(const std::string& name, const void* host_data, size_t bytes);

    // Execute the network synchronously (blocks until GPU finishes).
    bool infer();

    // Copy the named output's GPU buffer back to host memory.
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
    // Fast name→buffer lookup (pointers into inputs_/outputs_).
    std::unordered_map<std::string, BufferInfo*> buffer_map_;

    cudaStream_t stream_ = nullptr;
};
