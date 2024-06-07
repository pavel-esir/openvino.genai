// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    std::string model_path = argv[1];
    ov::genai::LLMPipeline pipe(model_path, "CPU");
    
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 10000;
    auto streamer = [](std::string word) -> bool { 
        std::cout << word << std::flush;
        // flag correspods whether generation should be stopped.
        // false means continue generation.
        return false;
    };

    pipe.start_chat();
    std::string prompt;
    for (;;) {
        std::cout << "question:\n";
        
        std::getline(std::cin, prompt);
        if (prompt == "Stop!") 
            break;

        pipe.generate(prompt, config, streamer);
        
        std::cout << "\n----------\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
