#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// CLIP BPE tokenizer (byte-level, matching OpenAI's implementation)
class ClipTokenizer {
  public:
    // Load vocabulary from bpe_simple_vocab_16e6.txt
    bool load(const std::string& vocab_path);

    // Encode text to token IDs, padded/truncated to max_length
    std::vector<int32_t> encode(const std::string& text,
                                 int max_length = 77) const;

  private:
    // BPE merge pair
    using BpePair = std::pair<std::string, std::string>;

    // Convert text to list of Unicode byte tokens
    std::vector<std::string> byte_encode(const std::string& text) const;

    // Apply BPE merges to a word
    std::vector<std::string> bpe(const std::string& token) const;

    // Vocabulary: token string → ID
    std::unordered_map<std::string, int> encoder_;

    // BPE merge ranks: pair → rank (lower = merge first)
    std::unordered_map<std::string, int> bpe_ranks_;

    // Byte-to-unicode mapping
    std::unordered_map<uint8_t, std::string> byte_encoder_;

    int sot_token_ = 49406;  // <|startoftext|>
    int eot_token_ = 49407;  // <|endoftext|>
};
