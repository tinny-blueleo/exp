#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

// Build the byte-to-unicode mapping used by CLIP's tokenizer.
//
// The problem: BPE operates on strings, but we need to handle arbitrary bytes
// (including control characters, null bytes, etc.). The solution is to map each
// byte value (0-255) to a unique Unicode character:
//
//   - Printable bytes (33-126, 161-172, 174-255) map to themselves
//     e.g., byte 65 ('A') → "A", byte 97 ('a') → "a"
//
//   - Non-printable bytes (0-32, 127-160, 173) map to codepoints 256+
//     e.g., byte 0 → U+0100 (Ā), byte 32 (space) → U+0120 (Ġ)
//
// This ensures every byte has a visible, unique representation in the
// vocabulary without collisions or ambiguity.
static std::unordered_map<uint8_t, std::string> bytes_to_unicode() {
    std::unordered_map<uint8_t, std::string> result;

    // Collect bytes that map to themselves (printable ranges).
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; i++) bs.push_back(i);      // ASCII printable
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);    // Latin-1 supplement
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);    // Latin-1 supplement

    std::set<int> bs_set(bs.begin(), bs.end());

    // Map printable bytes to their Unicode character (encoded as UTF-8).
    for (int b : bs) {
        std::string s;
        if (b < 0x80) {
            s = std::string(1, (char)b);
        } else {
            // 2-byte UTF-8: 110xxxxx 10xxxxxx
            s += (char)(0xC0 | (b >> 6));
            s += (char)(0x80 | (b & 0x3F));
        }
        result[(uint8_t)b] = s;
    }

    // Map remaining bytes to codepoints starting at 256.
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (bs_set.count(b) == 0) {
            int cp = 256 + n;
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

    // First line is a header comment, skip it.
    std::getline(file, line);

    // Read BPE merge rules. Each line is "token1 token2", meaning those two
    // tokens should be merged (in priority order — earlier = merge first).
    // CLIP uses exactly 48,894 merges (not the full 49,152 in the file).
    const int max_merges = 48894;
    int rank = 0;
    while (std::getline(file, line) && rank < max_merges) {
        if (line.empty()) continue;

        auto space = line.find(' ');
        if (space == std::string::npos) continue;

        std::string first = line.substr(0, space);
        std::string second = line.substr(space + 1);

        std::string key = first + " " + second;
        bpe_ranks_[key] = rank++;
    }
    file.close();

    // Build the vocabulary (token string → integer ID).
    //
    // The vocab is constructed in a specific order that determines token IDs:
    //   IDs 0-255:     single-byte tokens (one per byte value)
    //   IDs 256-511:   single-byte + "</w>" tokens (end-of-word variants)
    //   IDs 512-49405: merged tokens (one per merge rule, in merge order)
    //   ID 49406:      <|startoftext|>
    //   ID 49407:      <|endoftext|>
    std::vector<std::string> vocab;

    // Single-byte tokens (IDs 0-255).
    for (int b = 0; b < 256; b++) {
        auto it = byte_encoder_.find((uint8_t)b);
        if (it != byte_encoder_.end()) {
            vocab.push_back(it->second);
        }
    }

    // Byte + </w> tokens (IDs 256-511).
    for (int b = 0; b < 256; b++) {
        auto it = byte_encoder_.find((uint8_t)b);
        if (it != byte_encoder_.end()) {
            vocab.push_back(it->second + "</w>");
        }
    }

    // Merged tokens (IDs 512+). Each merge rule creates a new token by
    // concatenating the two tokens it merges.
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

    // Special tokens.
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

// Apply BPE merges to a single word.
//
// Starting from individual characters (byte-encoded), repeatedly find and
// merge the highest-priority pair until no more merges apply.
//
// Example for "space":
//   Initial:  ["s", "p", "a", "c", "e</w>"]
//   Merge 1:  ["sp", "a", "c", "e</w>"]       (rank of "s p" is lowest)
//   Merge 2:  ["sp", "a", "ce</w>"]            (rank of "c e</w>")
//   Merge 3:  ["sp", "ace</w>"]                (rank of "a ce</w>")
//   Merge 4:  ["space</w>"]                    (rank of "sp ace</w>")
//   Done: no more pairs have merge rules.
std::vector<std::string> ClipTokenizer::bpe(const std::string& token) const {
    // Split into individual UTF-8 characters.
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size();) {
        // Determine UTF-8 character length from first byte.
        int len = 1;
        unsigned char c = token[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;

        word.push_back(token.substr(i, len));
        i += len;
    }

    if (word.empty()) return word;

    // Mark the last character with </w> (end-of-word).
    // This distinguishes "space" (end of word) from "space" (prefix of
    // "spaceship"), which get different BPE tokens.
    word.back() += "</w>";

    // Greedily merge pairs until no more merges apply.
    while (word.size() > 1) {
        // Find the pair with the lowest merge rank (highest priority).
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

        if (best_idx < 0) break;  // No mergeable pairs remain.

        // Merge the best pair into a single token.
        std::string merged = word[best_idx] + word[best_idx + 1];
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); i++) {
            if ((int)i == best_idx) {
                new_word.push_back(merged);
                i++;  // skip the second token of the merged pair
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

    // Start with <|startoftext|> token (always the first token).
    tokens.push_back(sot_token_);

    // Lowercase the input (CLIP was trained on lowercased text).
    std::string clean;
    for (char c : text) {
        clean += std::tolower((unsigned char)c);
    }

    // Split on whitespace, then byte-encode and BPE each word.
    //
    // Note: the real CLIP tokenizer uses a regex to split on word boundaries
    // and punctuation. This simplified version splits on spaces only, which
    // works fine for typical prompts like "Taj Mahal in space".
    std::string word;
    for (size_t i = 0; i < clean.size(); i++) {
        char c = clean[i];
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!word.empty()) {
                // Convert word bytes to Unicode representations, then run BPE.
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

    // Process the last word (no trailing space).
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

    // End with <|endoftext|> token.
    tokens.push_back(eot_token_);

    // CLIP always expects exactly 77 tokens. Truncate if too long,
    // pad with <|endoftext|> if too short.
    if ((int)tokens.size() > max_length) {
        tokens.resize(max_length);
        tokens.back() = eot_token_;
    }
    while ((int)tokens.size() < max_length) {
        tokens.push_back(eot_token_);
    }

    return tokens;
}
