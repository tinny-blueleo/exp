#include "engine.h"

#include <fstream>
#include <iostream>

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

TrtEngine::~TrtEngine() { unload(); }

bool TrtEngine::load(const std::string& engine_path) {
    // Read the entire .trt file into memory. These files contain the compiled
    // network weights and execution plan — typically 100MB-1.6GB for SD models.
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine: " << engine_path << std::endl;
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        std::cerr << "Failed to read engine: " << engine_path << std::endl;
        return false;
    }
    file.close();

    // Deserialize: reconstruct the engine from its serialized form.
    // This loads weights into GPU memory and prepares CUDA kernels.
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) return false;

    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine: " << engine_path << std::endl;
        return false;
    }

    // An execution context holds the runtime state for one inference call.
    // Multiple contexts can share one engine for concurrent inference.
    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "Failed to allocate buffers" << std::endl;
        return false;
    }

    std::cout << "Loaded engine: " << engine_path
              << " (" << size / 1024 / 1024 << " MB)" << std::endl;
    return true;
}

void TrtEngine::unload() {
    free_buffers();
    context_.reset();
    engine_.reset();
    runtime_.reset();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool TrtEngine::allocate_buffers() {
    int num_io = engine_->getNbIOTensors();
    int num_profiles = engine_->getNbOptimizationProfiles();

    // Reserve vectors upfront so push_back won't reallocate and invalidate
    // the pointers we store in buffer_map_.
    int num_inputs = 0, num_outputs = 0;
    for (int i = 0; i < num_io; i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            num_inputs++;
        else
            num_outputs++;
    }
    inputs_.reserve(num_inputs);
    outputs_.reserve(num_outputs);

    for (int i = 0; i < num_io; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);
        auto mode = engine_->getTensorIOMode(name);

        // Dynamic-shape engines report -1 for variable dimensions.
        // We allocate based on the max profile shape so the buffer is large
        // enough for any valid input size.
        bool is_dynamic = false;
        for (int d = 0; d < dims.nbDims; d++) {
            if (dims.d[d] == -1) { is_dynamic = true; break; }
        }

        nvinfer1::Dims alloc_dims = dims;
        if (is_dynamic && num_profiles > 0) {
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                alloc_dims = engine_->getProfileShape(
                    name, 0, nvinfer1::OptProfileSelector::kMAX);
            } else {
                // Output shapes depend on inputs and can't be queried from the
                // profile directly. Use conservative upper bounds for our
                // known model structure (SD 512x512).
                for (int d = 0; d < alloc_dims.nbDims; d++) {
                    if (alloc_dims.d[d] == -1) {
                        if (d == 0) alloc_dims.d[d] = 1;        // batch
                        else if (d == 1 && alloc_dims.nbDims == 4)
                            alloc_dims.d[d] = 4;                 // channels
                        else if (alloc_dims.nbDims == 4)
                            alloc_dims.d[d] = 512;               // spatial
                        else alloc_dims.d[d] = 768;              // embedding
                    }
                }
            }
        }

        size_t bytes = volume(alloc_dims) * dtype_size(dtype);

        BufferInfo info;
        info.name = name;
        info.dims = alloc_dims;
        info.dtype = dtype;
        info.byte_size = bytes;

        if (cudaMalloc(&info.device_ptr, bytes) != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for " << name
                      << " (" << bytes << " bytes)" << std::endl;
            return false;
        }
        cudaMemset(info.device_ptr, 0, bytes);

        // Tell TensorRT where this tensor lives in GPU memory.
        context_->setTensorAddress(name, info.device_ptr);

        // Set an initial shape for dynamic inputs so the context is valid.
        if (is_dynamic && mode == nvinfer1::TensorIOMode::kINPUT
            && num_profiles > 0) {
            nvinfer1::Dims opt = engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kOPT);
            context_->setInputShape(name, opt);
            info.dims = opt;
        }

        if (mode == nvinfer1::TensorIOMode::kINPUT)
            inputs_.push_back(info);
        else
            outputs_.push_back(info);
    }

    // Build name→pointer map after all pushes (vector is now stable).
    for (auto& buf : inputs_) buffer_map_[buf.name] = &buf;
    for (auto& buf : outputs_) buffer_map_[buf.name] = &buf;

    return true;
}

void TrtEngine::free_buffers() {
    for (auto& buf : inputs_) {
        if (buf.device_ptr) { cudaFree(buf.device_ptr); buf.device_ptr = nullptr; }
    }
    for (auto& buf : outputs_) {
        if (buf.device_ptr) { cudaFree(buf.device_ptr); buf.device_ptr = nullptr; }
    }
    inputs_.clear();
    outputs_.clear();
    buffer_map_.clear();
}

bool TrtEngine::set_input_shape(const std::string& name,
                                 const nvinfer1::Dims& dims) {
    auto it = buffer_map_.find(name);
    if (it == buffer_map_.end()) {
        std::cerr << "Input tensor not found: " << name << std::endl;
        return false;
    }
    it->second->dims = dims;
    return context_->setInputShape(name.c_str(), dims);
}

bool TrtEngine::set_input(const std::string& name, const void* host_data,
                           size_t bytes) {
    auto it = buffer_map_.find(name);
    if (it == buffer_map_.end()) {
        std::cerr << "Input tensor not found: " << name << std::endl;
        return false;
    }
    BufferInfo* buf = it->second;
    if (bytes > buf->byte_size) {
        std::cerr << "Input size mismatch for " << name << ": " << bytes
                  << " > " << buf->byte_size << std::endl;
        return false;
    }
    return cudaMemcpyAsync(buf->device_ptr, host_data, bytes,
                           cudaMemcpyHostToDevice, stream_) == cudaSuccess;
}

bool TrtEngine::infer() {
    bool ok = context_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);
    return ok;
}

bool TrtEngine::get_output(const std::string& name, void* host_data,
                            size_t bytes) {
    auto it = buffer_map_.find(name);
    if (it == buffer_map_.end()) {
        std::cerr << "Output tensor not found: " << name << std::endl;
        return false;
    }
    BufferInfo* buf = it->second;
    if (bytes > buf->byte_size) {
        std::cerr << "Output size mismatch for " << name << ": " << bytes
                  << " > " << buf->byte_size << std::endl;
        return false;
    }
    return cudaMemcpyAsync(host_data, buf->device_ptr, bytes,
                           cudaMemcpyDeviceToHost, stream_) == cudaSuccess &&
           cudaStreamSynchronize(stream_) == cudaSuccess;
}

size_t TrtEngine::dtype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBF16:  return 2;
        default: return 4;
    }
}

size_t TrtEngine::volume(const nvinfer1::Dims& dims) {
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; i++) vol *= dims.d[i];
    return vol;
}
