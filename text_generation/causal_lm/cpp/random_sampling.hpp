// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <openvino/openvino.hpp>
#include <random>
#include <regex>
#include <vector>

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
    OPENVINO_ASSERT(penalty > 0, "Penalty must be a positive float, but got ", penalty);

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
    std::vector<int64_t> tokens;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.1;
    bool do_sample = true;
};

// RandomSampling processes logits produced by a language model and randomly samples token from top_k or top_p
// distribution. If do_sample set to false arg_max token returned.
// todo: add batching
struct RandomSampling {
    SamplingParameters parameters;
    RandomSampling(SamplingParameters parameters) : parameters{std::move(parameters)} {}

    int64_t get_out_token(float* logits, size_t vocab_size) {
        if (!parameters.do_sample) {
            int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;
            return out_token;
        }

        // logits pre-process
        if (parameters.repeat_penalty != 1.f) {
            // todo: concatenate tokens only if repetition penalty enabled
            sampling_repetition_penalty(logits, logits + vocab_size, parameters.tokens, parameters.repeat_penalty);
        }

        if (parameters.temp > 0) {
            sampling_temperature(logits, logits + vocab_size, parameters.temp);
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
        if (0.f < parameters.top_p && parameters.top_p < 1.f) {
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
        int64_t out_token = token_scores[dist(gen)].id;

        parameters.tokens.push_back(out_token);

        return out_token;
    }
};
