// C wrapper implementation — delegates every call to TrtEngineImpl.
#include "trt_wrapper.h"
#include "trt_engine.h"

#include <cstring>

// Helper: fill a TrtTensorInfo from a BufferInfo.
static void fill_info(const BufferInfo& buf, TrtTensorInfo* out) {
    out->name = buf.name.c_str();
    out->dtype = (int32_t)buf.dtype;
    out->num_dims = buf.dims.nbDims;
    for (int i = 0; i < buf.dims.nbDims; i++)
        out->dims[i] = buf.dims.d[i];
    for (int i = buf.dims.nbDims; i < 8; i++)
        out->dims[i] = 0;
    out->byte_size = buf.byte_size;
}

// Helper: find a buffer by name in a vector.
static const BufferInfo* find_by_name(const std::vector<BufferInfo>& vec,
                                       const char* name) {
    for (const auto& b : vec)
        if (b.name == name) return &b;
    return nullptr;
}

extern "C" {

TrtEngine* trt_engine_create(void) {
    return reinterpret_cast<TrtEngine*>(new TrtEngineImpl());
}

void trt_engine_destroy(TrtEngine* engine) {
    delete reinterpret_cast<TrtEngineImpl*>(engine);
}

bool trt_engine_load(TrtEngine* engine, const char* path) {
    return reinterpret_cast<TrtEngineImpl*>(engine)->load(path);
}

void trt_engine_unload(TrtEngine* engine) {
    reinterpret_cast<TrtEngineImpl*>(engine)->unload();
}

bool trt_engine_is_loaded(const TrtEngine* engine) {
    return reinterpret_cast<const TrtEngineImpl*>(engine)->is_loaded();
}

int trt_engine_num_inputs(const TrtEngine* engine) {
    return (int)reinterpret_cast<const TrtEngineImpl*>(engine)->inputs().size();
}

int trt_engine_num_outputs(const TrtEngine* engine) {
    return (int)reinterpret_cast<const TrtEngineImpl*>(engine)->outputs().size();
}

bool trt_engine_get_input_info(const TrtEngine* engine, int index,
                                TrtTensorInfo* info) {
    auto* impl = reinterpret_cast<const TrtEngineImpl*>(engine);
    if (index < 0 || index >= (int)impl->inputs().size()) return false;
    fill_info(impl->inputs()[index], info);
    return true;
}

bool trt_engine_get_output_info(const TrtEngine* engine, int index,
                                 TrtTensorInfo* info) {
    auto* impl = reinterpret_cast<const TrtEngineImpl*>(engine);
    if (index < 0 || index >= (int)impl->outputs().size()) return false;
    fill_info(impl->outputs()[index], info);
    return true;
}

bool trt_engine_find_input(const TrtEngine* engine, const char* name,
                            TrtTensorInfo* info) {
    auto* impl = reinterpret_cast<const TrtEngineImpl*>(engine);
    auto* buf = find_by_name(impl->inputs(), name);
    if (!buf) return false;
    fill_info(*buf, info);
    return true;
}

bool trt_engine_find_output(const TrtEngine* engine, const char* name,
                             TrtTensorInfo* info) {
    auto* impl = reinterpret_cast<const TrtEngineImpl*>(engine);
    auto* buf = find_by_name(impl->outputs(), name);
    if (!buf) return false;
    fill_info(*buf, info);
    return true;
}

bool trt_engine_set_input_shape(TrtEngine* engine, const char* name,
                                 const int64_t* dims, int num_dims) {
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = num_dims;
    for (int i = 0; i < num_dims && i < 8; i++)
        trt_dims.d[i] = dims[i];
    return reinterpret_cast<TrtEngineImpl*>(engine)->set_input_shape(name, trt_dims);
}

bool trt_engine_set_input(TrtEngine* engine, const char* name,
                           const void* data, size_t bytes) {
    return reinterpret_cast<TrtEngineImpl*>(engine)->set_input(name, data, bytes);
}

bool trt_engine_infer(TrtEngine* engine) {
    return reinterpret_cast<TrtEngineImpl*>(engine)->infer();
}

bool trt_engine_get_output(TrtEngine* engine, const char* name,
                            void* data, size_t bytes) {
    return reinterpret_cast<TrtEngineImpl*>(engine)->get_output(name, data, bytes);
}

} // extern "C"
