// CLIP BPE (Byte-Pair Encoding) tokenizer — pure Zig.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  WHAT THIS DOES                                                         │
// │                                                                         │
// │  Neural networks only understand numbers. Before the text encoder can   │
// │  process a prompt like "Taj Mahal in space", it must be converted to    │
// │  a sequence of integer token IDs:                                       │
// │                                                                         │
// │    "Taj Mahal in space"                                                 │
// │        ↓ lowercase                                                      │
// │    "taj mahal in space"                                                 │
// │        ↓ split on whitespace                                            │
// │    ["taj", "mahal", "in", "space"]                                      │
// │        ↓ byte-encode each word → BPE merge → vocab lookup               │
// │    [24805, 22301, 530, 2138]                                            │
// │        ↓ wrap with start/end tokens, pad to 77                          │
// │    [49406, 24805, 22301, 530, 2138, 49407, 49407, ...]                  │
// │                                                                         │
// │  HOW BPE WORKS                                                          │
// │                                                                         │
// │  BPE is a compression algorithm. Starting from individual characters,   │
// │  it repeatedly merges the most common adjacent pair:                    │
// │                                                                         │
// │    "space" → ["s", "p", "a", "c", "e</w>"]                             │
// │            → ["sp", "a", "c", "e</w>"]       merge "s"+"p"             │
// │            → ["sp", "a", "ce</w>"]            merge "c"+"e</w>"         │
// │            → ["sp", "ace</w>"]                merge "a"+"ce</w>"        │
// │            → ["space</w>"]                    merge "sp"+"ace</w>"      │
// │                                                                         │
// │  The </w> suffix marks end-of-word, so "space" at the end of a word    │
// │  gets a different token than "space" as a prefix of "spaceship".        │
// │                                                                         │
// │  WHY BYTE-LEVEL ENCODING?                                               │
// │                                                                         │
// │  To handle ANY input (Unicode, control chars, etc.), each byte (0-255)  │
// │  is mapped to a unique visible Unicode character. Printable ASCII maps  │
// │  to itself (A→A), while non-printable bytes get mapped to codepoints    │
// │  starting at 256 (space→Ġ, null→Ā). This avoids "unknown token"       │
// │  issues entirely.                                                       │
// │                                                                         │
// │  VOCABULARY STRUCTURE (49,408 tokens total):                            │
// │    IDs 0-255:     single-byte tokens                                    │
// │    IDs 256-511:   single-byte + "</w>" tokens                           │
// │    IDs 512-49405: merged tokens (one per merge rule)                    │
// │    ID 49406:      <|startoftext|>                                       │
// │    ID 49407:      <|endoftext|>                                         │
// └─────────────────────────────────────────────────────────────────────────┘

const std = @import("std");

const max_merges = 48894;
const max_token_len = 77;
const sot_token: i32 = 49406;
const eot_token: i32 = 49407;

pub const ClipTokenizer = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    /// token string → integer ID (e.g. "space</w>" → 2330)
    encoder: std.StringHashMap(i32),
    /// "first second" → merge priority rank (lower = merge first)
    bpe_ranks: std.StringHashMap(i32),
    /// byte value (0-255) → Unicode string representation
    byte_encoder: [256][]const u8,

    pub fn init(allocator: std.mem.Allocator) ClipTokenizer {
        var self = ClipTokenizer{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .encoder = std.StringHashMap(i32).init(allocator),
            .bpe_ranks = std.StringHashMap(i32).init(allocator),
            .byte_encoder = undefined,
        };
        // Initialize byte_encoder to empty slices
        for (&self.byte_encoder) |*entry| {
            entry.* = "";
        }
        return self;
    }

    pub fn deinit(self: *ClipTokenizer) void {
        self.encoder.deinit();
        self.bpe_ranks.deinit();
        self.arena.deinit();
    }

    /// Load the BPE vocabulary file (bpe_simple_vocab_16e6.txt).
    pub fn load(self: *ClipTokenizer, vocab_path: []const u8) !void {
        self.buildByteEncoder();

        // Read the entire file into memory (~3.2MB)
        const file = try std.fs.cwd().openFile(vocab_path, .{});
        defer file.close();
        const file_size = (try file.stat()).size;
        const contents = try self.arena.allocator().alloc(u8, file_size);
        var total_read: usize = 0;
        while (total_read < file_size) {
            const n = try file.read(contents[total_read..]);
            if (n == 0) break;
            total_read += n;
        }
        const data = contents[0..total_read];

        // Split into lines
        var lines = std.mem.splitScalar(u8, data, '\n');

        // Skip header line
        _ = lines.next();

        // First pass: read BPE merge rules
        var rank: i32 = 0;
        var lines_copy = lines; // save position for second pass
        while (rank < max_merges) {
            const l = lines.next() orelse break;
            if (l.len == 0) continue;
            const space_idx = std.mem.indexOfScalar(u8, l, ' ') orelse continue;
            const key = try std.fmt.allocPrint(self.arena.allocator(), "{s} {s}", .{ l[0..space_idx], l[space_idx + 1 ..] });
            try self.bpe_ranks.put(key, rank);
            rank += 1;
        }

        // Build vocabulary in the specific order that determines token IDs.
        // Single-byte tokens (IDs 0-255)
        for (0..256) |b| {
            const s = self.byte_encoder[b];
            if (s.len > 0) {
                try self.encoder.put(s, @intCast(b));
            }
        }

        // Byte + </w> tokens (IDs 256-511)
        for (0..256) |b| {
            const s = self.byte_encoder[b];
            if (s.len > 0) {
                const sw = try std.fmt.allocPrint(self.arena.allocator(), "{s}</w>", .{s});
                try self.encoder.put(sw, @as(i32, @intCast(b)) + 256);
            }
        }

        // Second pass: merged tokens (IDs 512+)
        var merge_id: i32 = 512;
        var mc: i32 = 0;
        while (mc < max_merges) {
            const l = lines_copy.next() orelse break;
            if (l.len == 0) continue;
            const space_idx = std.mem.indexOfScalar(u8, l, ' ') orelse continue;
            mc += 1;
            const merged = try std.fmt.allocPrint(self.arena.allocator(), "{s}{s}", .{ l[0..space_idx], l[space_idx + 1 ..] });
            try self.encoder.put(merged, merge_id);
            merge_id += 1;
        }

        // Special tokens
        const sot_str = try self.arena.allocator().dupe(u8, "<|startoftext|>");
        const eot_str = try self.arena.allocator().dupe(u8, "<|endoftext|>");
        try self.encoder.put(sot_str, sot_token);
        try self.encoder.put(eot_str, eot_token);

        std.debug.print("Loaded CLIP tokenizer: {d} tokens, {d} merges\n", .{ self.encoder.count(), self.bpe_ranks.count() });
    }

    /// Tokenize text into a fixed-length 77-token sequence:
    /// [<|startoftext|>, ...tokens..., <|endoftext|>, pad, pad, ...]
    pub fn encode(self: *ClipTokenizer, text: []const u8, buf: []i32) void {
        @memset(buf, eot_token);
        buf[0] = sot_token;
        var count: usize = 1;

        // Lowercase the input (CLIP was trained on lowercased text)
        var lower: [512]u8 = undefined;
        const text_len = @min(text.len, lower.len);
        for (0..text_len) |i| {
            lower[i] = std.ascii.toLower(text[i]);
        }
        const clean = lower[0..text_len];

        // Split on whitespace and process each word
        var iter = std.mem.tokenizeAny(u8, clean, " \t\n");
        while (iter.next()) |word| {
            if (count >= max_token_len - 1) break; // leave room for EOT

            // Byte-encode the word: convert each byte to its Unicode representation
            var encoded_buf: [2048]u8 = undefined;
            var encoded_len: usize = 0;
            for (word) |ch| {
                const s = self.byte_encoder[ch];
                @memcpy(encoded_buf[encoded_len..][0..s.len], s);
                encoded_len += s.len;
            }
            const encoded = encoded_buf[0..encoded_len];

            // Apply BPE merges, then look up each resulting token in the vocabulary
            var bpe_result: [64][]const u8 = undefined;
            const bpe_count = self.bpe(encoded, &bpe_result);

            for (bpe_result[0..bpe_count]) |token| {
                if (count >= max_token_len - 1) break;
                if (self.encoder.get(token)) |id| {
                    buf[count] = id;
                    count += 1;
                }
            }
        }

        // Ensure EOT is placed after the last real token
        if (count < max_token_len) {
            buf[count] = eot_token;
        }
    }

    /// Build the byte-to-unicode mapping (same as OpenAI's bytes_to_unicode).
    fn buildByteEncoder(self: *ClipTokenizer) void {
        const alloc = self.arena.allocator();

        // Printable ASCII and Latin-1 supplement map to themselves
        var is_printable = [_]bool{false} ** 256;
        for ('!'..('~' + 1)) |b| is_printable[b] = true;
        for (0xA1..0xAD) |b| is_printable[b] = true;
        for (0xAE..0x100) |b| is_printable[b] = true;

        for (0..256) |b| {
            if (is_printable[b]) {
                // Map to the Unicode codepoint for this byte value
                if (b < 0x80) {
                    self.byte_encoder[b] = alloc.dupe(u8, &[_]u8{@intCast(b)}) catch "";
                } else {
                    // 2-byte UTF-8 encoding
                    const bytes = [_]u8{
                        @intCast(0xC0 | (b >> 6)),
                        @intCast(0x80 | (b & 0x3F)),
                    };
                    self.byte_encoder[b] = alloc.dupe(u8, &bytes) catch "";
                }
            }
        }

        // Non-printable bytes get mapped to codepoints starting at 256
        var n: usize = 0;
        for (0..256) |b| {
            if (!is_printable[b]) {
                const cp = 256 + n;
                if (cp < 0x80) {
                    self.byte_encoder[b] = alloc.dupe(u8, &[_]u8{@intCast(cp)}) catch "";
                } else if (cp < 0x800) {
                    const bytes = [_]u8{
                        @intCast(0xC0 | (cp >> 6)),
                        @intCast(0x80 | (cp & 0x3F)),
                    };
                    self.byte_encoder[b] = alloc.dupe(u8, &bytes) catch "";
                }
                n += 1;
            }
        }
    }

    /// Apply BPE merges to a single word.
    /// Starts from individual UTF-8 characters, adds </w> to last,
    /// then greedily merges the highest-priority pair until done.
    fn bpe(self: *ClipTokenizer, token: []const u8, result: *[64][]const u8) usize {
        // Split into individual UTF-8 characters
        var word: [128][]const u8 = undefined;
        var word_len: usize = 0;
        var i: usize = 0;
        while (i < token.len and word_len < 127) {
            // Determine UTF-8 character length from the first byte
            var char_len: usize = 1;
            const c = token[i];
            if (c & 0xE0 == 0xC0) char_len = 2 else if (c & 0xF0 == 0xE0) char_len = 3 else if (c & 0xF8 == 0xF0) char_len = 4;
            if (i + char_len > token.len) break;
            word[word_len] = token[i .. i + char_len];
            word_len += 1;
            i += char_len;
        }
        if (word_len == 0) return 0;

        // Append </w> to the last character (marks end of word)
        var last_buf: [32]u8 = undefined;
        const last = word[word_len - 1];
        @memcpy(last_buf[0..last.len], last);
        @memcpy(last_buf[last.len..][0..4], "</w>");
        // We need a stable pointer — use the arena
        const last_with_w = self.arena.allocator().dupe(u8, last_buf[0 .. last.len + 4]) catch {
            result[0] = token;
            return 1;
        };
        word[word_len - 1] = last_with_w;

        // Greedily merge pairs until no more merges apply
        var pair_buf: [256]u8 = undefined;
        while (word_len > 1) {
            // Find the pair with the lowest merge rank
            var best_rank: i32 = std.math.maxInt(i32);
            var best_idx: usize = word_len; // sentinel = no match

            for (0..word_len - 1) |j| {
                const a = word[j];
                const b = word[j + 1];
                if (a.len + 1 + b.len > pair_buf.len) continue;
                @memcpy(pair_buf[0..a.len], a);
                pair_buf[a.len] = ' ';
                @memcpy(pair_buf[a.len + 1 ..][0..b.len], b);
                const pair_key = pair_buf[0 .. a.len + 1 + b.len];

                if (self.bpe_ranks.get(pair_key)) |r| {
                    if (r < best_rank) {
                        best_rank = r;
                        best_idx = j;
                    }
                }
            }

            if (best_idx >= word_len) break; // No more merges

            // Merge the pair at best_idx
            const a = word[best_idx];
            const b_tok = word[best_idx + 1];
            const merged = self.arena.allocator().alloc(u8, a.len + b_tok.len) catch break;
            @memcpy(merged[0..a.len], a);
            @memcpy(merged[a.len..][0..b_tok.len], b_tok);

            // Rebuild word array with the merged token replacing the pair
            var new_word: [128][]const u8 = undefined;
            var nw: usize = 0;
            var k: usize = 0;
            while (k < word_len) {
                if (k == best_idx) {
                    new_word[nw] = merged;
                    nw += 1;
                    k += 2; // skip both parts of the merged pair
                } else {
                    new_word[nw] = word[k];
                    nw += 1;
                    k += 1;
                }
            }
            @memcpy(word[0..nw], new_word[0..nw]);
            word_len = nw;
        }

        @memcpy(result[0..word_len], word[0..word_len]);
        return word_len;
    }
};
