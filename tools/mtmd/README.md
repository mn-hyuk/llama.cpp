# Multimodal Support in llama.cpp

This directory provides multimodal capabilities for `llama.cpp`. Initially intended as a showcase for running LLaVA models, its scope has expanded significantly over time to include various other vision-capable models. As a result, LLaVA is no longer the only multimodal architecture supported.

> [!IMPORTANT]
>
> Multimodal support can be viewed as a sub-project within `llama.cpp`. It is under **very heavy development**, and **breaking changes are expected**.

The naming and structure related to multimodal support have evolved, which might cause some confusion. Here's a brief timeline to clarify:

- [#3436](https://github.com/ggml-org/llama.cpp/pull/3436): Initial support for LLaVA 1.5 was added, introducing `llava.cpp` and `clip.cpp`. The `llava-cli` binary was created for model interaction.
- [#4954](https://github.com/ggml-org/llama.cpp/pull/4954): Support for MobileVLM was added, becoming the second vision model supported. This built upon the existing `llava.cpp`, `clip.cpp`, and `llava-cli` infrastructure.
- **Expansion & Fragmentation:** Many new models were subsequently added (e.g., [#7599](https://github.com/ggml-org/llama.cpp/pull/7599), [#10361](https://github.com/ggml-org/llama.cpp/pull/10361), [#12344](https://github.com/ggml-org/llama.cpp/pull/12344), and others). However, `llava-cli` lacked support for the increasingly complex chat templates required by these models. This led to the creation of model-specific binaries like `qwen2vl-cli`, `minicpmv-cli`, and `gemma3-cli`. While functional, this proliferation of command-line tools became confusing for users.
- [#12849](https://github.com/ggml-org/llama.cpp/pull/12849): `libmtmd` was introduced as a replacement for `llava.cpp`. Its goals include providing a single, unified command-line interface, improving the user/developer experience (UX/DX), and supporting both audio and image inputs.
- [#13012](https://github.com/ggml-org/llama.cpp/pull/13012): `mtmd-cli` was added, consolidating the various model-specific CLIs into a single tool powered by `libmtmd`.

## Pre-quantized models

See the list of pre-quantized model [here](../../docs/multimodal.md)

## How it works and what is `mmproj`?

Multimodal support in `llama.cpp` works by encoding images into embeddings using a separate model component, and then feeding these embeddings into the language model.

This approach keeps the multimodal components distinct from the core `libllama` library. Separating these allows for faster, independent development cycles. While many modern vision models are based on Vision Transformers (ViTs), their specific pre-processing and projection steps can vary significantly. Integrating this diverse complexity directly into `libllama` is currently challenging.

Consequently, running a multimodal model typically requires two GGUF files:
1.  The standard language model file.
2.  A corresponding **multimodal projector (`mmproj`)** file, which handles the image encoding and projection.

## What is `libmtmd`?

As outlined in the history, `libmtmd` is the modern library designed to replace the original `llava.cpp` implementation for handling multimodal inputs.

Built upon `clip.cpp` (similar to `llava.cpp`), `libmtmd` offers several advantages:
- **Unified Interface:** Aims to consolidate interaction for various multimodal models.
- **Improved UX/DX:** Features a more intuitive API, inspired by the `Processor` class in the Hugging Face `transformers` library.
- **Flexibility:** Designed to support multiple input types (text, audio, images) while respecting the wide variety of chat templates used by different models.

## Layerwise mmproj offload and runtime swap

Layer-based vision encoders can use two different `mmproj` placement modes:

### 1. Static layer placement

Keep a fixed subset of encoder layers on the GPU and leave the rest on the CPU:

```bash
./build/bin/llama-mtmd-cli \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --image /path/to/image.jpg \
  --mmproj-offload \
  --mmproj-n-gpu-layers 8
```

You can also select exact layers explicitly:

```bash
./build/bin/llama-mtmd-cli \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --image /path/to/image.jpg \
  --mmproj-offload \
  --mmproj-gpu-layers 0-3,8,10-12
```

- `--mmproj-n-gpu-layers N`: keeps the last `N` encoder layers on the GPU.
- `--mmproj-gpu-layers LAYERS`: selects exact encoder layers to place on the GPU.
- `--mmproj-gpu-layers` overrides `--mmproj-n-gpu-layers`.

### 2. Runtime swap / streaming

Keep encoder weights on the CPU and stream only a chunk of encoder layers through the GPU at execution time:

```bash
./build/bin/llama-mtmd-cli \
  -m /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --image /path/to/image.jpg \
  --mmproj-offload \
  --mmproj-runtime-swap \
  --mmproj-n-gpu-layers 4
```

In this mode:

- `--mmproj-n-gpu-layers N` means the number of encoder layers to execute per GPU chunk.
- encoder layer weights remain on the CPU and are streamed through the GPU during vision encoding.
- `--mmproj-gpu-layers` is not compatible with `--mmproj-runtime-swap`.

This mode is intended for GPU-constrained environments where full `mmproj` offload does not fit comfortably in VRAM, but a smaller execution chunk does.

### Current support

`--mmproj-runtime-swap` currently supports:

- Gemma 3 vision encoders
- Qwen3-VL vision encoders

For unsupported projectors, `llama.cpp` falls back to the existing static loading path.

### Notes

- `--mmproj-n-gpu-layers -1` keeps the legacy full-offload behavior when runtime swap is disabled.
- Runtime swap is only useful when `--mmproj-offload` can initialize a GPU backend.
- Warmup / graph reservation also follow the runtime-swap pre/chunk/post execution path, so VRAM sizing is based on the chunked execution flow instead of the full encoder graph.

## How to obtain `mmproj`

Multimodal projector (`mmproj`) files are specific to each model architecture.

For the following models, you can use `convert_hf_to_gguf.py` with `--mmproj` flag to get the `mmproj` file:
- [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) ; See the guide [here](../../docs/multimodal/gemma3.md) - Note: 1B variant does not have vision support
- SmolVLM (from [HuggingFaceTB](https://huggingface.co/HuggingFaceTB))
- SmolVLM2 (from [HuggingFaceTB](https://huggingface.co/HuggingFaceTB))
- [Pixtral 12B](https://huggingface.co/mistral-community/pixtral-12b) - only works with `transformers`-compatible checkpoint
- Qwen 2 VL and Qwen 2.5 VL (from [Qwen](https://huggingface.co/Qwen))
- [Mistral Small 3.1 24B](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)
- InternVL 2.5 and InternVL 3 from [OpenGVLab](https://huggingface.co/OpenGVLab) (note: we don't support conversion of `InternVL3-*-hf` model, only non-HF version is supported ; `InternLM2Model` **text** model is not supported)

For older models, please refer to the relevant guide for instructions on how to obtain or create them:

NOTE: conversion scripts are located under `tools/mtmd/legacy-models`

- [LLaVA](../../docs/multimodal/llava.md)
- [MobileVLM](../../docs/multimodal/MobileVLM.md)
- [GLM-Edge](../../docs/multimodal/glmedge.md)
- [MiniCPM-V 2.5](../../docs/multimodal/minicpmv2.5.md)
- [MiniCPM-V 2.6](../../docs/multimodal/minicpmv2.6.md)
- [MiniCPM-o 2.6](../../docs/multimodal/minicpmo2.6.md)
- [IBM Granite Vision](../../docs/multimodal/granitevision.md)
