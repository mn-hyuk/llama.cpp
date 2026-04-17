#pragma once

#if defined(GGML_USE_CUDA) && (defined(__linux__) || defined(__APPLE__))
#include <array>
#include <dlfcn.h>
#define MTMD_HAS_NVTX 1

using mtmd_nvtx_range_push_fn = int (*)(const char *);
using mtmd_nvtx_range_pop_fn = int (*)();

struct mtmd_nvtx_api {
    mtmd_nvtx_range_push_fn push = nullptr;
    mtmd_nvtx_range_pop_fn pop = nullptr;
};

static inline mtmd_nvtx_api mtmd_load_nvtx_api() {
    mtmd_nvtx_api api;

    void * handle = nullptr;

    static const std::array<const char *, 6> candidate_paths = {{
        "libnvToolsExt.so.1",
        "libnvToolsExt.so",
        "/opt/ohpc/pub/cuda/11.8.0/targets/x86_64-linux/lib/libnvToolsExt.so.1",
        "/opt/ohpc/pub/cuda/11.8.0/targets/x86_64-linux/lib/libnvToolsExt.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so.1",
        "/usr/local/cuda/lib64/libnvToolsExt.so.1",
    }};

    for (const char * path : candidate_paths) {
        handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
        if (handle != nullptr) {
            break;
        }
    }

#if defined(__APPLE__)
    if (handle == nullptr) {
        handle = dlopen("libnvToolsExt.dylib", RTLD_LAZY | RTLD_LOCAL);
    }
#endif

    if (handle != nullptr) {
        api.push = reinterpret_cast<mtmd_nvtx_range_push_fn>(dlsym(handle, "nvtxRangePushA"));
        api.pop = reinterpret_cast<mtmd_nvtx_range_pop_fn>(dlsym(handle, "nvtxRangePop"));
        if (api.push == nullptr || api.pop == nullptr) {
            api = {};
        }
    }

    return api;
}

static inline const mtmd_nvtx_api & mtmd_get_nvtx_api() {
    static const mtmd_nvtx_api api = mtmd_load_nvtx_api();
    return api;
}
#else
#define MTMD_HAS_NVTX 0
#endif

struct mtmd_nvtx_range {
    explicit mtmd_nvtx_range(const char * name) {
#if MTMD_HAS_NVTX
        const auto & api = mtmd_get_nvtx_api();
        if (name != nullptr && api.push != nullptr) {
            api.push(name);
        }
#else
        (void) name;
#endif
    }

    ~mtmd_nvtx_range() {
#if MTMD_HAS_NVTX
        const auto & api = mtmd_get_nvtx_api();
        if (api.pop != nullptr) {
            api.pop();
        }
#endif
    }

    mtmd_nvtx_range(const mtmd_nvtx_range &) = delete;
    mtmd_nvtx_range & operator=(const mtmd_nvtx_range &) = delete;
};
