// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    
    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(model_path, device);

    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = 100;
    config.do_sample = false;
    auto streamer = [](std::string subword){ std::cout << subword << std::flush; return false; };
    
    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    pipe.generate(prompt, config, streamer);
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
