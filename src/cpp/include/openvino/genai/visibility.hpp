// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/core/visibility.hpp"

#ifdef genai_EXPORTS
#    define openvino/genai_EXPORTS OPENVINO_CORE_EXPORTS
#else
#    define openvino/genai_EXPORTS OPENVINO_CORE_IMPORTS
#endif  // genai_EXPORTS
