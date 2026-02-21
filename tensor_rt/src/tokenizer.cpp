#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

// Build the byte-to-unicode mapping used by CLIP
static std::unordered_map<uint8_t, std::string> bytes_to_unicode() {
    std::unordered_map<uint8_t, std::string> result;

    // Printable ASCII and Latin-1 supplement ranges that map to themselves
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

    std::set<int> bs_set(bs.begin(), bs.end());

    // Map those bytes to single-char UTF-8 strings
    for (int b : bs) {
        // These bytes map to themselves as Unicode codepoints
        std::string s;
        if (b < 0x80) {
            s = std::string(1, (char)b);
        } else {
            // Encode as UTF-8
            s += (char)(0xC0 | (b >> 6));
            s += (char)(0x80 | (b & 0x3F));
        }
        result[(uint8_t)b] = s;
    }

    // Remaining bytes (0-32, 127-160, etc.) get mapped to codepoints 256+
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (bs_set.count(b) == 0) {
            int cp = 256 + n;
            // Encode codepoint as UTF-8
            std::string s;
            if (cp < 0x80) {
                s = std::string(1, (char)cp);
            } else if (cp < 0x800) {
                s += (char)(0xC0 | (cp >> 6));
                s += (char)(0x80 | (cp & 0x3F));
            }
            result[(uint8_t)b] = s;
            n++;
        }
    }

    return result;
}

bool ClipTokenizer::load(const std::string& vocab_path) {
    byte_encoder_ = bytes_to_unicode();

    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
        return false;
    }

    std::string line;

    // First line is a comment, skip it
    std::getline(file, line);

    // Read BPE merges — CLIP uses only the first 48894 merges
    // (matching OpenAI's: merges[1:49152-256-2+1])
    const int max_merges = 48894;
    int rank = 0;
    while (std::getline(file, line) && rank < max_merges) {
        if (line.empty()) continue;

        // Lines are "token1 token2" merge pairs
        auto space = line.find(' ');
        if (space == std::string::npos) continue;

        std::string first = line.substr(0, space);
        std::string second = line.substr(space + 1);

        std::string key = first + " " + second;
        bpe_ranks_[key] = rank++;
    }
    file.close();

    // Build encoder: the vocab file implicitly defines token IDs
    // First 256 entries are single byte tokens, then 256 byte+</w> tokens,
    // then the merge results in order, then special tokens
    std::vector<std::string> vocab;

    // Single byte tokens
    for (int b = 0; b < 256; b++) {
        auto it = byte_encoder_.find((uint8_t)b);
        if (it != byte_encoder_.end()) {
            vocab.push_back(it->second);
        }
    }

    // Byte + </w> tokens
    for (int b = 0; b < 256; b++) {
        auto it = byte_encoder_.find((uint8_t)b);
        if (it != byte_encoder_.end()) {
            vocab.push_back(it->second + "</w>");
        }
    }

    // Merge results (each merge creates a new token)
    // Re-read the file to get merge results in order, limited to max_merges
    std::ifstream file2(vocab_path);
    std::getline(file2, line);  // skip header

    int merge_count = 0;
    while (std::getline(file2, line) && merge_count < max_merges) {
        if (line.empty()) continue;
        auto space = line.find(' ');
        if (space == std::string::npos) continue;
        merge_count++;

        std::string merged = line.substr(0, space) + line.substr(space + 1);
        vocab.push_back(merged);
    }
    file2.close();

    // Special tokens
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");

    for (int i = 0; i < (int)vocab.size(); i++) {
        encoder_[vocab[i]] = i;
    }

    sot_token_ = encoder_["<|startoftext|>"];
    eot_token_ = encoder_["<|endoftext|>"];

    std::cout << "Loaded CLIP tokenizer: " << encoder_.size() << " tokens, "
              << bpe_ranks_.size() << " merges" << std::endl;
    return true;
}

std::vector<std::string> ClipTokenizer::byte_encode(
    const std::string& text) const {
    std::vector<std::string> result;
    for (unsigned char c : text) {
        auto it = byte_encoder_.find(c);
        if (it != byte_encoder_.end()) {
            result.push_back(it->second);
        }
    }
    return result;
}

std::vector<std::string> ClipTokenizer::bpe(const std::string& token) const {
    // Split token into individual characters (byte-encoded)
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size();) {
        // Handle UTF-8 multi-byte sequences
        int len = 1;
        unsigned char c = token[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        word.push_back(token.substr(i, len));
        i += len;
    }

    if (word.empty()) return word;

    // Add </w> to last token
    word.back() += "</w>";

    while (word.size() > 1) {
        // Find the pair with lowest merge rank
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (size_t i = 0; i + 1 < word.size(); i++) {
            std::string pair_key = word[i] + " " + word[i + 1];
            auto it = bpe_ranks_.find(pair_key);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;  // No more merges

        // Merge the best pair
        std::string merged = word[best_idx] + word[best_idx + 1];
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); i++) {
            if ((int)i == best_idx) {
                new_word.push_back(merged);
                i++;  // skip next
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = std::move(new_word);
    }

    return word;
}

std::vector<int32_t> ClipTokenizer::encode(const std::string& text,
                                            int max_length) const {
    std::vector<int32_t> tokens;
    tokens.push_back(sot_token_);

    // Lowercase and basic cleanup
    std::string clean;
    for (char c : text) {
        clean += std::tolower((unsigned char)c);
    }

    // Simple whitespace tokenization (CLIP uses a regex, but for basic prompts
    // splitting on whitespace + punctuation works)
    // Pattern: split on spaces, keep punctuation as separate tokens
    std::string word;
    for (size_t i = 0; i < clean.size(); i++) {
        char c = clean[i];
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!word.empty()) {
                // Byte-encode and BPE the word
                std::string encoded;
                for (unsigned char ch : word) {
                    auto it = byte_encoder_.find(ch);
                    if (it != byte_encoder_.end()) {
                        encoded += it->second;
                    }
                }
                auto bpe_tokens = bpe(encoded);
                for (const auto& t : bpe_tokens) {
                    auto it = encoder_.find(t);
                    if (it != encoder_.end()) {
                        tokens.push_back(it->second);
                    }
                }
                word.clear();
            }
        } else {
            word += c;
        }
    }

    // Process last word
    if (!word.empty()) {
        std::string encoded;
        for (unsigned char ch : word) {
            auto it = byte_encoder_.find(ch);
            if (it != byte_encoder_.end()) {
                encoded += it->second;
            }
        }
        auto bpe_tokens = bpe(encoded);
        for (const auto& t : bpe_tokens) {
            auto it = encoder_.find(t);
            if (it != encoder_.end()) {
                tokens.push_back(it->second);
            }
        }
    }

    tokens.push_back(eot_token_);

    // Pad or truncate to max_length
    if ((int)tokens.size() > max_length) {
        tokens.resize(max_length);
        tokens.back() = eot_token_;
    }
    while ((int)tokens.size() < max_length) {
        tokens.push_back(eot_token_);
    }

    return tokens;
}
