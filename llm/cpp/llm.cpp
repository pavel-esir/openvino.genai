// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <utils.hpp>
#include <valarray>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int32_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int32_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}
}

int main(int argc, char* argv[]) try {
    if (argc != 5) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>'");
    }
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    constexpr size_t BATCH_SIZE = 1;
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, -1
        }}
    };
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    model->reshape(shapes);
    ov::preprocess::PrePostProcessor p3(model);
    p3.input("input_ids").tensor().set_element_type(ov::element::i32);  // cast to the type of tokenyzer's output
    p3.input("attention_mask").tensor().set_element_type(ov::element::i32);
    model = p3.build();
    ov::CompiledModel compiled = core.compile_model(model, "CPU", {ov::cache_dir("llm-cache")});
    ov::InferRequest ireq = compiled.create_infer_request();
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    ireq.get_tensor("attention_mask").set_shape(input_ids.get_shape());
    std::copy_n(input_ids.data<const int32_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int32_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int32_t>(), input_ids.get_size(), 1);
    ireq.infer();
    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    // float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    // int32_t out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, 1});
    ireq.get_tensor("attention_mask").data<int32_t>()[0] = 1;
    constexpr int32_t SPECIAL_EOS_TOKEN = 1;  // There's no way to extract the value from the tokenizer for now  // TODO: 2 for llama2
    constexpr size_t N_GROUPS = 2;
    constexpr size_t GROUP_SIZE = 2;
    float DIVERSITY_PENALTY = 1.0f;
    struct Beam {
        float log_prob;
        std::vector<size_t> tokens;
        ov::InferRequest ireq;
    };
    struct Group {
        std::vector<Beam> beams;  // TODO: one contigous array with all beams?
    };
    ov::Tensor logits_tensor = ireq.get_tensor("logits");
    std::valarray<float> logits{logits_tensor.data<const float>(), logits_tensor.get_size()};  // TODO: maybe use valarray<Token>
    float max_logit = logits.max();
    float log_sum = std::log((std::exp(logits - max_logit)).sum());  // TODO: log(softmax) only for topk logits
    std::valarray<float> log_prob = logits - max_logit - log_sum;
    struct Token {float log; size_t idx;
        bool operator<(Token indexed) {
            return log > indexed.log;  // greater, not less to pick most probable tokens
        }
    };
    std::vector<Token> topk;
    topk.reserve(log_prob.size());
    for (size_t idx = 0; idx < log_prob.size(); ++idx) {
        topk.push_back({log_prob[idx], idx});
    }
    std::vector<Group> groups{N_GROUPS};
    for (size_t group_idx = 0; group_idx < N_GROUPS; ++group_idx) {
        std::partial_sort(topk.begin(), topk.begin() + GROUP_SIZE, topk.end());
        for (size_t idx = 0; idx < GROUP_SIZE; ++idx) {
            groups[group_idx].beams.push_back(Beam{topk[idx].log, {topk[idx].idx}, compiled.create_infer_request()});
            topk[idx].log -= DIVERSITY_PENALTY;
            ov::InferRequest& beam_ireq = groups[group_idx].beams[idx].ireq;
            for (size_t tensor_idx = 2; tensor_idx < inputs.size(); ++tensor_idx) {
                beam_ireq.set_input_tensor(tensor_idx, ireq.get_output_tensor(tensor_idx - 1));
            }
            beam_ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
            beam_ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, 1});  // TODO all after fix is reelased
            beam_ireq.get_tensor("attention_mask").data<int32_t>()[0] = 1;
            beam_ireq.get_tensor("input_ids").data<int32_t>()[0] = topk[idx].idx;
            beam_ireq.start_async();
        }
    }

    for (;;) {
        for (size_t group_idx = 0; group_idx < N_GROUPS; ++group_idx) {
            for (size_t beam_idx = 0; beam_idx < GROUP_SIZE; ++beam_idx) {
                ov::InferRequest& beam_ireq = groups[group_idx].beams[beam_idx].ireq;
                beam_ireq.wait();
                ov::Tensor logits_tensor = ireq.get_tensor("logits");
                std::valarray<float> logits{logits_tensor.data<const float>(), logits_tensor.get_size()};  // TODO: maybe use valarray<Token>
                float max_logit = logits.max();
                float log_sum = std::log((std::exp(logits - max_logit)).sum());  // TODO: log(softmax) only for topk logits
                std::valarray<float> log_prob = logits - max_logit - log_sum;
                std::vector<Token> topk;
                topk.reserve(log_prob.size());
                for (size_t idx = 0; idx < log_prob.size(); ++idx) {
                    topk.push_back({log_prob[idx], idx});
                }
                for (size_t prev_group_idx = 0; prev_group_idx < group_idx; ++prev_group_idx) {  // TODO: range based for
                    for (size_t prev_beam_idx = 0; prev_beam_idx < GROUP_SIZE; ++prev_beam_idx) {
                        topk[groups[prev_group_idx].beams[prev_beam_idx].tokens.back()] -= DIVERSITY_PENALTY;

                    }
                }
                if ()



                std::partial_sort(topk.begin(), topk.begin() + GROUP_SIZE, topk.end());
            }

        }


    }



    // 5971, 25068

    // [ 1727, 29392, 25700, 18559]



    // while (out_token != SPECIAL_EOS_TOKEN) {
    //     for (size_t idx = 2; idx < inputs.size(); ++idx) {
    //          ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 1));
    //     }
    //     ireq.get_tensor("input_ids").data<int32_t>()[0] = out_token;
    //     ireq.start_async();
    //     print_token(detokenizer, out_token);
    //     ireq.wait();
    //     logits = ireq.get_tensor("logits").data<float>();
    //     out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
    // }
    std::cout << '\n';
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
