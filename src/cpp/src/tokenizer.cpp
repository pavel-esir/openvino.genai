// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include "openvino/genai/tokenizer.hpp"
#include "utils.hpp"
#include <fstream>

namespace {

// todo: remove when openvino-tokenizers will support left padding
ov::genai::TokenizedInputs pad_left(ov::Tensor& input_ids, ov::Tensor& attention_mask, int64_t pad_token_id) {
    const size_t batch_size = input_ids.get_shape()[0];
    const size_t sequence_length = input_ids.get_shape()[1];
    int64_t* inputs_data = input_ids.data<int64_t>();
    int64_t* attention_mask_data = attention_mask.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        const size_t batch_offset = batch * sequence_length;

        // last token in the sequence is not a PAD_TOKEN, skipping
        if (inputs_data[batch_offset + sequence_length - 1] != pad_token_id)
            continue;

        size_t pad_tokens_number = 0;
        for (int i = sequence_length - 1; i >= 0; i--) {
            const size_t token_offset = batch_offset + i;

            if (inputs_data[token_offset] == pad_token_id)
                continue;

            if (pad_tokens_number == 0)
                pad_tokens_number = sequence_length - i - 1;

            std::swap(inputs_data[token_offset], inputs_data[token_offset + pad_tokens_number]);
            std::swap(attention_mask_data[token_offset], attention_mask_data[token_offset + pad_tokens_number]);
        }
    }

    return {input_ids, attention_mask};
}

#ifdef _WIN32
#    include <windows.h>
#    define MAX_ABS_PATH _MAX_PATH
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
#else
#    include <dlfcn.h>
#    include <limits.h>
#    define MAX_ABS_PATH PATH_MAX
#    define get_absolute_path(result, path) realpath(path.c_str(), result)

std::string get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    std::ignore = get_absolute_path(&absolutePath[0], path);
    if (!absolutePath.empty()) {
        // on Linux if file does not exist or no access, function will return NULL, but
        // `absolutePath` will contain resolved path
        absolutePath.resize(absolutePath.find('\0'));
        return std::string(absolutePath);
    }
    std::stringstream ss;
    ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
    throw std::runtime_error(ss.str());
}
#endif

std::string get_ov_genai_library_path() {
    #ifdef _WIN32
        CHAR genai_library_path[MAX_PATH];
        HMODULE hm = NULL;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPSTR>(get_ov_genai_library_path),
                                &hm)) {
            std::stringstream ss;
            ss << "GetModuleHandle returned " << GetLastError();
            throw std::runtime_error(ss.str());
        }
        GetModuleFileNameA(hm, (LPSTR)genai_library_path, sizeof(genai_library_path));
        return std::string(genai_library_path);
    #elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
        Dl_info info;
        dladdr(reinterpret_cast<void*>(get_ov_genai_library_path), &info);
        return get_absolute_file_path(info.dli_fname).c_str();
    #else
    #    error "Unsupported OS"
    #endif  // _WIN32
}

std::filesystem::path with_openvino_tokenizers(const std::filesystem::path& path) {
    #ifdef _WIN32
        constexpr char tokenizers[] = "openvino_tokenizers.dll";
    #elif __linux__
        constexpr char tokenizers[] = "libopenvino_tokenizers.so";
    #elif __APPLE__
        constexpr char tokenizers[] = "libopenvino_tokenizers.dylib";
    #endif
        return path.parent_path() / tokenizers;
}
}  // namespace

namespace ov {
namespace genai {

class Tokenizer::TokenizerImpl {
public:
    ov::InferRequest m_tokenize_request;
    ov::InferRequest m_detokenizer_request;
    int64_t m_pad_token_id = -1;
    int64_t m_bos_token_id = -1;
    int64_t m_eos_token_id = -1;

    std::string m_pad_token = "";
    std::string m_bos_token = "";
    std::string m_eos_token = "";

    void spec_tokens_from_tokenizers_config_if_exists(const std::filesystem::path& tokenizers_config_path) {
        if (!std::filesystem::exists(tokenizers_config_path))
            return ;

        std::ifstream f(tokenizers_config_path);
        if (!f.is_open())
            return ;

        nlohmann::json data = nlohmann::json::parse(f);
        std::string spec_tokens_tag = "added_tokens_decoder";

        std::string pad_token_tag = "pad_token";
        std::string bos_token_tag = "bos_token";
        std::string eos_token_tag = "eos_token";

        if (!data.contains(spec_tokens_tag))
            return ;
        
        using ov::genai::utils::read_json_param;
        auto added_tokens_decoder = data[spec_tokens_tag];
        
        // special tokens string representation from json
        read_json_param(data, pad_token_tag, m_pad_token);
        read_json_param(data, bos_token_tag, m_bos_token);
        read_json_param(data, eos_token_tag, m_eos_token);
        
        for (auto& [key, value] : added_tokens_decoder.items()) {
            if (!value.contains("content"))
                continue;
            auto content = value["content"];
            if (content == m_pad_token)
                m_pad_token_id = std::stoi(key);
            if (content == m_bos_token)
                m_bos_token_id = std::stoi(key);
            if (content == m_eos_token)
                m_eos_token_id = std::stoi(key);
        }
    }

    TokenizerImpl() = default;
    TokenizerImpl(std::filesystem::path tokenizers_path) {
        ov::Core core;
        
        if (tokenizers_path.extension() == ".xml")
            OPENVINO_THROW("tokenizers_path should be a path to a dir not a xml file");

        const char* ov_tokenizers_path = getenv(ScopedVar::ENVIRONMENT_VARIABLE_NAME);
        if (ov_tokenizers_path) {
            core.add_extension(ov_tokenizers_path);
        } else {
            OPENVINO_THROW("openvino_tokenizers path is not set");
        }

        auto tokenizer_json_path = tokenizers_path / "tokenizer_config.json";
        spec_tokens_from_tokenizers_config_if_exists(tokenizer_json_path);

        if (m_pad_token_id == -1 || m_bos_token_id || m_eos_token_id) {
            using ov::genai::utils::get_special_tokens_from_config_json;
            auto config_path = tokenizers_path / "config.json";
            auto [pad_token_id, bos_token_id, eos_token_id] = get_special_tokens_from_config_json(config_path);
            if (m_pad_token_id == -1)
                m_pad_token_id = pad_token_id;
            if (m_bos_token_id == -1)
                m_bos_token_id = bos_token_id;
            if (m_eos_token_id == -1)
                m_eos_token_id = eos_token_id;
        }

        auto device = "CPU"; // currently openvino_tokenizer supports only CPU
        m_tokenize_request = core.compile_model(tokenizers_path / "openvino_tokenizer.xml", 
                                                device).create_infer_request();
        m_detokenizer_request = core.compile_model(tokenizers_path / "openvino_detokenizer.xml", 
                                                   device).create_infer_request();
    }

    TokenizedInputs encode(std::string prompt) {
        size_t batch_size = 1;
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {batch_size}, &prompt});
        m_tokenize_request.infer();
        return get_copied_results();
    }

    TokenizedInputs encode(std::vector<std::string>& prompts) {
        m_tokenize_request.set_input_tensor(ov::Tensor{ov::element::string, {prompts.size()}, prompts.data()});
        auto size_ = m_tokenize_request.get_input_tensor().get_shape();
        m_tokenize_request.infer();
       
        auto res = get_copied_results();
        pad_left(res.input_ids, res.attention_mask, m_pad_token_id);
        return {res.input_ids, res.attention_mask};
    }

    TokenizedInputs get_copied_results() {
        auto input_ids = m_tokenize_request.get_tensor("input_ids");
        auto attention_mask = m_tokenize_request.get_tensor("attention_mask");
        ov::Tensor input_ids_ = ov::Tensor(input_ids.get_element_type(), input_ids.get_shape());
        ov::Tensor attention_mask_ = ov::Tensor(attention_mask.get_element_type(), attention_mask.get_shape());
        input_ids.copy_to(input_ids_);
        attention_mask.copy_to(attention_mask_);

        return {input_ids_, attention_mask_};        
    }

    std::string decode(std::vector<int64_t> tokens) {
        size_t batch_size = 1;
        m_detokenizer_request.set_input_tensor(ov::Tensor{ov::element::i64, {batch_size, tokens.size()}, tokens.data()});
        m_detokenizer_request.infer();
        return m_detokenizer_request.get_output_tensor().data<std::string>()[0];
    }

    std::vector<std::string> decode(ov::Tensor tokens) {
        OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64, "tokens tensor element type should be an i64");
        OPENVINO_ASSERT(tokens.get_shape().size() == 2, "tokens tensor should of rank 2 with shape [batch_size, seq_len]");

        ov::genai::utils::print_tensor(tokens);

        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }

    std::vector<std::string> decode(std::vector<std::vector<int64_t>> lines) {
        auto compare_lengths = [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
            return a.size() < b.size();
        };
        size_t max_len = std::max_element(lines.begin(), lines.end(), compare_lengths)->size();

        ov::Tensor tokens = ov::Tensor{ov::element::i64, {lines.size(), max_len}};
        auto tokens_data = tokens.data<int64_t>();
        
        for (size_t i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            size_t line_len = line.size();
            std::copy(line.begin(), line.end(), tokens_data + i * max_len);
            std::fill(tokens_data + i * max_len + line_len, tokens_data + (i + 1) * max_len, m_pad_token_id);
        }

        m_detokenizer_request.set_input_tensor(tokens);
        m_detokenizer_request.infer();
        auto res = m_detokenizer_request.get_output_tensor();
        auto res_data = res.data<std::string>();
        return std::vector<std::string>(res_data, res_data + res.get_shape()[0]);
    }
};

Tokenizer::Tokenizer(const std::string& tokenizers_path) {
    ov::genai::ScopedVar env_manager(tokenizers_relative_to_genai().string());
    m_pimpl = std::make_shared<TokenizerImpl>(tokenizers_path);
}

TokenizedInputs Tokenizer::encode(const std::string prompt) {
    return m_pimpl->encode(std::move(prompt));
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::vector<std::string>&& prompts) {
    return m_pimpl->encode(prompts);
}

TokenizedInputs Tokenizer::encode(std::initializer_list<std::string>& text) {
    return encode(std::vector<std::string>(text.begin(), text.end()));
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(ov::Tensor tokens) {
    return m_pimpl->decode(tokens);
}

std::vector<std::string> Tokenizer::decode(std::vector<std::vector<int64_t>> lines) {
    return m_pimpl->decode(lines);
}

int64_t Tokenizer::get_bos_token_id() const {
    return m_pimpl->m_bos_token_id;
}

int64_t Tokenizer::get_eos_token_id() const {
    return m_pimpl->m_eos_token_id;
}

int64_t Tokenizer::get_pad_token_id() const {
    return m_pimpl->m_pad_token_id;
}

std::string Tokenizer::get_pad_token() const {
    return m_pimpl->m_pad_token;
}

std::string Tokenizer::get_bos_token() const {
    return m_pimpl->m_bos_token;
}

std::string Tokenizer::get_eos_token() const {
    return m_pimpl->m_eos_token;
}

Tokenizer::~Tokenizer() = default;

std::filesystem::path tokenizers_relative_to_genai() {
    return with_openvino_tokenizers(get_ov_genai_library_path());
}

ScopedVar::ScopedVar(const std::string& environment_variable_value) {
#ifdef _WIN32
    char* value = nullptr;
    size_t len = 0;
    _dupenv_s(&value, &len, ENVIRONMENT_VARIABLE_NAME);
    if (value == nullptr)
        _putenv_s(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str());
#else
    if (!getenv(ENVIRONMENT_VARIABLE_NAME))
        setenv(ENVIRONMENT_VARIABLE_NAME, environment_variable_value.c_str(), 1);
#endif
    else
        was_already_set = true;
}

ScopedVar::~ScopedVar() {
    if (!was_already_set) {
#ifdef _WIN32
        _putenv_s(ENVIRONMENT_VARIABLE_NAME, "");
#else
        unsetenv(ENVIRONMENT_VARIABLE_NAME);
#endif
    }
}
}  // namespace genai
}  // namespace ov
