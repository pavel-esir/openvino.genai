// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <vector>

#include "generation_config_helper.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"


namespace {

struct TokenIdScore {
    int id;
    float score;

    TokenIdScore() = default;
    TokenIdScore(int id, float score) : id(id), score(score) {}

    bool operator<(const TokenIdScore& other) const {
        return score < other.score;
    }
    bool operator>(const TokenIdScore& other) const {
        return score > other.score;
    }
};

void sampling_softmax_inplace(TokenIdScore* first, TokenIdScore* last) {
    float max_score = std::max_element(first, last)->score;
    float sum = 0.f;
    for (TokenIdScore* p = first; p != last; p++) {
        float s = std::exp(p->score - max_score);
        p->score = s;
        sum += s;
    }
    float inv_sum = 1.f / sum;
    for (TokenIdScore* p = first; p != last; p++) {
        p->score *= inv_sum;
    }
}

void sampling_top_k(TokenIdScore* first, TokenIdScore* kth, TokenIdScore* last) {
    std::nth_element(first, kth, last, std::greater<TokenIdScore>());
}

TokenIdScore* sampling_top_p(TokenIdScore* first, TokenIdScore* last, float top_p) {
    // sort score
    std::sort(first, last, std::greater<TokenIdScore>());

    int vocab_size = last - first;
    std::vector<TokenIdScore> token_scores(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        token_scores[i] = first[i];
    }

    // calculate softmax
    sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

    float prefix_sum = 0.0f;

    // top_p
    for (int i = 0; i < vocab_size; i++) {
        prefix_sum += token_scores[i].score;
        if (prefix_sum >= top_p) {
            return first + (i + 1);
        }
    }

    return last;
}

void sampling_repetition_penalty(float* first, float* last, const std::vector<int64_t>& input_ids, float penalty) {
    const float inv_penalty = 1.f / penalty;
    const int vocab_size = last - first;
    std::vector<bool> occurrence(vocab_size, false);
    for (const int64_t id : input_ids) {
        if (!occurrence[id]) {
            first[id] *= (first[id] > 0) ? inv_penalty : penalty;
        }
        occurrence[id] = true;
    }
}

void sampling_temperature(float* first, float* last, float temp) {
    const float inv_temp = 1.f / temp;
    for (float* it = first; it != last; it++) {
        *it *= inv_temp;
    }
}

struct SamplingParameters {
    int top_k;
    float top_p;
    float temperature;
    float repetition_penalty;

    SamplingParameters(ov::GenerationConfig generation_config) {
        // parameters validation
        OPENVINO_ASSERT(generation_config.top_k > 0,
                        "top_k must be a strictly positive float, but got ",
                        generation_config.top_p);
        OPENVINO_ASSERT(generation_config.top_p > 0 || generation_config.top_p < 1.0f,
                        "top_p must be a positive float > 0 and < 1, but got ",
                        generation_config.top_p);
        OPENVINO_ASSERT(generation_config.temperature > 0,
                        "Temperature must be a strictly positive float, but got ",
                        generation_config.temperature);
        OPENVINO_ASSERT(generation_config.repetition_penalty > 0,
                        "Repetition penalty must be a strictly positive float, but got ",
                        generation_config.repetition_penalty);

        top_k = generation_config.top_k;
        top_p = generation_config.top_p;
        temperature = generation_config.temperature;
        repetition_penalty = generation_config.repetition_penalty;
    }
};

struct RandomSampling {
    SamplingParameters parameters;
    RandomSampling(SamplingParameters parameters) : parameters{std::move(parameters)} {}

    TokenIdScore get_out_token(float* logits, size_t vocab_size, std::vector<int64_t> tokens) {
        // logits pre-process
        if (parameters.repetition_penalty != 1.0f) {
            sampling_repetition_penalty(logits, logits + vocab_size, tokens, parameters.repetition_penalty);
        }

        if (parameters.temperature > 0) {
            sampling_temperature(logits, logits + vocab_size, parameters.temperature);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < parameters.top_k && parameters.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(),
                           token_scores.data() + parameters.top_k,
                           token_scores.data() + token_scores.size());
            token_scores.resize(parameters.top_k);
        }

        // top_p sampling
        if (0.f < parameters.top_p && parameters.top_p < 1.0f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), parameters.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        TokenIdScore out_token = token_scores[dist(gen)];

        return out_token;
    }
};
}  // namespace

namespace ov {

// todo: add batching
ov::EncodedResults multinominal_decoding(ov::InferRequest& m_model_runner,
                                         ov::Tensor input_ids,
                                         ov::Tensor attention_mask,
                                         ov::GenerationConfig generation_config,
                                         std::shared_ptr<StreamerBase> streamer) {
    ov::GenerationConfigHelper config_helper{generation_config};

    ov::Shape prompts_shape = input_ids.get_shape();
    size_t batch_size = prompts_shape[0];
    size_t prompt_len = prompts_shape[1];

    ov::EncodedResults results;
    results.scores.resize(batch_size);
    results.tokens.resize(batch_size);
    std::fill(results.scores.begin(), results.scores.end(), 0);

    // Initialize inputs
    m_model_runner.set_tensor("input_ids", input_ids);
    m_model_runner.set_tensor("attention_mask", attention_mask);

    ov::Tensor position_ids = m_model_runner.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), 0);

    m_model_runner.get_tensor("beam_idx").set_shape({batch_size});
    m_model_runner.get_tensor("beam_idx").data<int32_t>()[0] = 0;

    m_model_runner.infer();

    auto logits_tensor = m_model_runner.get_tensor("logits");

    int64_t sequence_offset = logits_tensor.get_shape().at(1) - 1;
    size_t vocab_size = logits_tensor.get_shape().back();

    float* logits = logits_tensor.data<float>() + (sequence_offset)*vocab_size;

    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    std::vector<int64_t> tokens{input_ids_data, input_ids_data + input_ids.get_size()};

    RandomSampling sampling{SamplingParameters{generation_config}};

    TokenIdScore out_token = sampling.get_out_token(logits, vocab_size, tokens);

    tokens.push_back(out_token.id);
    results.tokens[0].push_back(out_token.id);
    results.scores[0] += out_token.score;

    if (streamer) {
        streamer->put(out_token.id);
    }

    if (!generation_config.ignore_eos && out_token.id == generation_config.eos_token_id) {
        return results;
    }

    m_model_runner.get_tensor("input_ids").set_shape({batch_size, 1});
    m_model_runner.get_tensor("position_ids").set_shape({batch_size, 1});

    size_t max_new_tokens = config_helper.get_max_new_tokens(prompt_len);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        ov::generate_utils::update_position_ids(m_model_runner.get_tensor("position_ids"),
                                                m_model_runner.get_tensor("attention_mask"));
        m_model_runner.set_tensor("attention_mask",
                                  ov::generate_utils::extend_attention(m_model_runner.get_tensor("attention_mask")));

        m_model_runner.get_tensor("input_ids").data<int64_t>()[0] = out_token.id;

        m_model_runner.infer();

        logits = m_model_runner.get_tensor("logits").data<float>();
        out_token = sampling.get_out_token(logits, vocab_size, tokens);

        tokens.push_back(out_token.id);
        results.tokens[0].push_back(out_token.id);
        results.scores[0] += out_token.score;

        if (streamer) {
            streamer->put(out_token.id);
        }

        if (!generation_config.ignore_eos && out_token.id == generation_config.eos_token_id) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return results;
}
}  // namespace ov
