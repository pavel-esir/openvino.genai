// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "openvino/core/any.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

using StreamerVariant = std::variant<std::function<void (std::string)>, std::shared_ptr<StreamerBase>>;
using OptionalGenerationConfig = std::optional<GenerationConfig>;
using OptionalStreamerVariant = std::optional<StreamerVariant>;

/**
* @brief Structure to store resulting batched tokens and scores for each batch sequence
*
* @param tokens sequence of resulting tokens
* @param scores scores for each sequence
*/
class EncodedResults {
public:
    std::vector<std::vector<int64_t>> tokens;
    std::vector<float> scores;
};

/**
* @brief Structure to store resulting batched text outputs and scores for each batch
*
* @param texts vector of resulting sequences
* @param scores scores for each sequence
*/
class DecodedResults {
public:
    std::vector<std::string> texts;
    std::vector<float> scores;

     // @brief Convert DecodedResults to a vector of strings.
     // @return A std::vector<std::string> containing the texts from the DecodedResults object.
    operator std::vector<std::string>() const { 
        return texts; 
    }
    
     // @brief Overloads operator<< to enhance output the contents of DecodedResults.
     // @return A reference to the output stream with the concatenated texts.
    friend std::ostream& operator<<(std::ostream& os, const DecodedResults& dr) {
       for (size_t i = 0; i < dr.texts.size(); ++i) {
            os << dr.texts[i];
            if (i != dr.texts.size() - 1) {
                os << std::endl;
            }
        }
        return os;
    }
};

/**
* @brief This class is used for generation with LLMs.
 */
class OPENVINO_GENAI_EXPORTS LLMPipeline {
public:
    /**
    * @brief Constructs an LLMPipeline from xml/bin files, tokenizers and configuration in the same dir.
    *
    * @param model_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
    * @param device optional device
    * @param plugin_config optional plugin_config
    * @param ov_tokenizers_path optional path to an extension to add. Empty adds openvino_tokenizers from openvini_genai library folder.
    */
    LLMPipeline(const std::string& path, const std::string& device="CPU", 
                const ov::AnyMap& plugin_config={}, 
                const std::string& ov_tokenizers_path="");
    
    /**
    * @brief Constructs a LLMPipeline when ov::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param tokenizer manually initialized ov::Tokenizer 
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    LLMPipeline(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config = {}
    );
    
    ~LLMPipeline();

    /**
    * @brief High level generate for the input with a single prompt which encodes inputs and returns decoded output
    *
    * @param text input prompt
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return std::string decoded resulting text
    */
    std::string generate(std::string text, OptionalGenerationConfig generation_config=std::nullopt, OptionalStreamerVariant streamer=std::nullopt);
    
    template <typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> generate(
            std::string text,
            Properties&&... properties) {
        return generate(text, AnyMap{std::forward<Properties>(properties)...});
    }
    std::string generate(std::string text, const ov::AnyMap& config);

    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
        ov::Tensor input_ids,
        Properties&&... properties) {
        return generate(input_ids, AnyMap{std::forward<Properties>(properties)...});
    }
    EncodedResults generate(ov::Tensor input_ids, const ov::AnyMap& config);

    /**
    * @brief High level generate for batched prompts which encodes inputs and returns decoded outputs. 
    * Streamer cannot be used for multibatch inputs.
    *
    * @param text input prompt
    * @param generation_config optional GenerationConfig
    * @return DecodedResults a structure with resulting texts & scores
    */
    DecodedResults generate(const std::vector<std::string>& texts, OptionalGenerationConfig generation_config);

    /**
    * @brief Low level generate to be called with already encoded input_ids tokens.
    * Streamer cannot be used for multibatch inputs.
    *
    * @param input_ids encoded input prompt tokens
    * @param attention_mask optional attention_mask
    * @param generation_config optional GenerationConfig
    * @param streamer optional streamer
    * @return EncodedResults a structure with resulting tokens and scores
    * @throws Exception if the stremaer is set for inputs_ids with multiple batches
    */
    EncodedResults generate(ov::Tensor input_ids, 
                            std::optional<ov::Tensor> attention_mask, 
                            OptionalGenerationConfig generation_config=std::nullopt,
                            OptionalStreamerVariant streamer=std::nullopt);
    
    template <typename InputsType, typename... Properties>
    util::EnableIfAllStringAny<std::string, Properties...> operator()(
        InputsType text,
        Properties&&... properties) {
        return generate(text, AnyMap{std::forward<Properties>(properties)...});
    }
    
    DecodedResults operator()(const std::vector<std::string>& text, OptionalGenerationConfig generation_config=std::nullopt) {
        return generate(text, generation_config);
    }

    std::string operator()(
        std::string text, 
        OptionalGenerationConfig generation_config=std::nullopt, 
        OptionalStreamerVariant streamer=std::nullopt
    ) {
        return generate(text, generation_config, streamer);
    }
    
    ov::genai::Tokenizer get_tokenizer();
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    void start_chat();
    void finish_chat();
    void reset_state();
    std::string apply_chat_template(std::string prompt, std::string role = "user") const;
private:
    class LLMPipelineImpl;
    std::unique_ptr<LLMPipelineImpl> m_pimpl;
};

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(input_ids, ov::max_new_tokens(200), ov::temperature(1.0f),...)
 * pipe(text, ov::max_new_tokens(200), ov::temperature(1.0f),...)
*/
static constexpr ov::Property<size_t> max_new_tokens{"max_new_tokens"};
static constexpr ov::Property<size_t> max_length{"max_length"};
static constexpr ov::Property<bool> ignore_eos{"ignore_eos"};

static constexpr ov::Property<size_t> num_beam_groups{"num_beam_groups"};
static constexpr ov::Property<size_t> num_beams{"num_beams"};
static constexpr ov::Property<float> diversity_penalty{"diversity_penalty"};
static constexpr ov::Property<float> length_penalty{"length_penalty"};
static constexpr ov::Property<size_t> num_return_sequences{"num_return_sequences"};
static constexpr ov::Property<size_t> no_repeat_ngram_size{"no_repeat_ngram_size"};
static constexpr ov::Property<StopCriteria> stop_criteria{"stop_criteria"};

static constexpr ov::Property<float> temperature{"temperature"};
static constexpr ov::Property<float> top_p{"top_p"};
static constexpr ov::Property<int> top_k{"top_k"};
static constexpr ov::Property<bool> do_sample{"do_sample"};
static constexpr ov::Property<float> repetition_penalty{"repetition_penalty"};


static constexpr ov::Property<int64_t> pad_token_id{"pad_token_id"};
static constexpr ov::Property<int64_t> bos_token_id{"bos_token_id"};
static constexpr ov::Property<int64_t> eos_token_id{"eos_token_id"};
    
static constexpr ov::Property<std::string> bos_token{"bos_token"};
static constexpr ov::Property<std::string> eos_token{"eos_token"};

// only lambda streamer can be set via ov::streamer(),... syntaxic sugar,
// because std::variant<StremaerBase, std::function<>> can not be stored in AnyMap
static constexpr ov::Property<std::function<void (std::string)>> streamer{"streamer"};

}  // namespace genai
}  // namespace ov
