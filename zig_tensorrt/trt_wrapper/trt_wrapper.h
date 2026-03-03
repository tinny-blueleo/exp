// C API wrapping the C++ TensorRT engine class.
//
// TensorRT is a pure C++ API (virtual classes, no C ABI). Zig cannot call
// C++ directly, so this header provides a flat C interface using opaque
// pointers. The implementation (trt_wrapper.cpp) delegates to TrtEngineImpl.
#ifndef TRT_WRAPPER_H
#define TRT_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the C++ TrtEngineImpl class.
typedef struct TrtEngine TrtEngine;

// Mirrors nvinfer1::DataType integer values exactly.
enum TrtDataType {
    TRT_FLOAT  = 0,
    TRT_HALF   = 1,
    TRT_INT8   = 2,
    TRT_INT32  = 3,
    TRT_BOOL   = 4,
    TRT_UINT8  = 5,
    TRT_FP8    = 6,
    TRT_BF16   = 7,
    TRT_INT64  = 8,
};

// Flat C struct for tensor metadata.
// The name pointer is borrowed from the C++ side — valid while engine is loaded.
typedef struct {
    const char* name;
    int32_t dtype;       // TrtDataType value
    int32_t num_dims;    // 1-8
    int64_t dims[8];     // shape (matches TensorRT Dims64.d[MAX_DIMS])
    size_t byte_size;    // total allocated GPU bytes
} TrtTensorInfo;

// --- Lifecycle ---
TrtEngine* trt_engine_create(void);
void trt_engine_destroy(TrtEngine* engine);

// --- Load / Unload ---
bool trt_engine_load(TrtEngine* engine, const char* path);
void trt_engine_unload(TrtEngine* engine);
bool trt_engine_is_loaded(const TrtEngine* engine);

// --- Tensor enumeration ---
int trt_engine_num_inputs(const TrtEngine* engine);
int trt_engine_num_outputs(const TrtEngine* engine);
bool trt_engine_get_input_info(const TrtEngine* engine, int index,
                                TrtTensorInfo* info);
bool trt_engine_get_output_info(const TrtEngine* engine, int index,
                                 TrtTensorInfo* info);
bool trt_engine_find_input(const TrtEngine* engine, const char* name,
                            TrtTensorInfo* info);
bool trt_engine_find_output(const TrtEngine* engine, const char* name,
                             TrtTensorInfo* info);

// --- Operations ---
bool trt_engine_set_input_shape(TrtEngine* engine, const char* name,
                                 const int64_t* dims, int num_dims);
bool trt_engine_set_input(TrtEngine* engine, const char* name,
                           const void* data, size_t bytes);
bool trt_engine_infer(TrtEngine* engine);
bool trt_engine_get_output(TrtEngine* engine, const char* name,
                            void* data, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // TRT_WRAPPER_H
