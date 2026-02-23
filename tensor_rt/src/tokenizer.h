#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// CLIP BPE (Byte-Pair Encoding) tokenizer.
//
// Before the text encoder neural network can process a prompt like
// "Taj Mahal in space", it needs to be converted to a sequence of
// integer token IDs. This tokenizer does that conversion.
//
// How BPE tokenization works:
//
//   1. BYTE ENCODING: Each character is mapped to a Unicode string
//      representation. This handles all possible bytes (0-255) by mapping
//      printable ASCII to themselves and other bytes to Unicode codepoints
//      256+. This lets the tokenizer handle any input without "unknown" tokens.
//
//   2. BPE MERGING: Starting from individual characters, repeatedly merge
//      the most common adjacent pair. For example:
//        "space" → ["s", "p", "a", "c", "e</w>"]
//                → ["sp", "a", "c", "e</w>"]      (merge "s"+"p")
//                → ["sp", "a", "ce</w>"]           (merge "c"+"e</w>")
//                → ["sp", "ace</w>"]               (merge "a"+"ce</w>")
//                → ["space</w>"]                   (merge "sp"+"ace</w>")
//      The </w> suffix marks end-of-word.
//
//   3. VOCABULARY LOOKUP: Each BPE token maps to an integer ID using a
//      vocabulary of ~49,000 tokens learned during CLIP training.
//
//   4. PADDING: CLIP always expects exactly 77 tokens, padded with the
//      end-of-text token and bookended by start/end-of-text markers.
//
// The vocabulary file (bpe_simple_vocab_16e6.txt) comes from OpenAI's CLIP
// and contains 48,894 merge rules. The full vocab has 49,408 tokens:
//   - 256 single-byte tokens
//   - 256 single-byte + </w> tokens
//   - 48,894 merged tokens
//   - 2 special tokens (<|startoftext|>, <|endoftext|>)
class ClipTokenizer {
  public:
    bool load(const std::string& vocab_path);

    // Tokenize a text prompt into a fixed-length sequence of token IDs.
    // Always returns exactly max_length (77) tokens:
    //   [<|startoftext|>, ...tokens..., <|endoftext|>, pad, pad, ...]
    std::vector<int32_t> encode(const std::string& text,
                                 int max_length = 77) const;

  private:
    using BpePair = std::pair<std::string, std::string>;

    // Convert each byte of the input to its Unicode string representation.
    std::vector<std::string> byte_encode(const std::string& text) const;

    // Apply BPE merges to a single word until no more merges are possible.
    // Returns the final list of BPE tokens (e.g. ["space</w>"]).
    std::vector<std::string> bpe(const std::string& token) const;

    // token string → integer ID (e.g. "space</w>" → 2330)
    std::unordered_map<std::string, int> encoder_;

    // "first second" → merge priority rank (lower = merge first).
    // Built from the vocabulary file's merge rules.
    std::unordered_map<std::string, int> bpe_ranks_;

    // byte value → Unicode string (e.g. 0x20 (space) → "Ġ", 0x41 ('A') → "A")
    std::unordered_map<uint8_t, std::string> byte_encoder_;

    int sot_token_ = 49406;  // <|startoftext|>
    int eot_token_ = 49407;  // <|endoftext|>
};
