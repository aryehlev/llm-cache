//! Chunking and hashing helpers: turn prompts into the block-hash sequences the
//! trie consumes (DESIGN.md §3.1).
//!
//! Only *full* blocks are emitted — a trailing partial block can never be a
//! stable prefix boundary, so it is dropped from the key. Use the same block
//! size as [`crate::Config::block_tokens`].
//!
//! Hashes come from [`std::collections::hash_map::DefaultHasher`]: stable
//! within a process, **not** guaranteed stable across Rust releases — do not
//! persist tries across binaries built with different toolchains.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Pluggable prompt-to-blocks conversion, so any tokenizer can drive the trie.
///
/// Implement this once around a real tokenizer for token-exact block
/// boundaries:
///
/// ```ignore
/// struct TiktokenChunker { bpe: tiktoken_rs::CoreBPE, block_tokens: usize }
/// impl llm_cache::chunk::Chunker for TiktokenChunker {
///     fn blocks(&self, prompt: &str) -> Vec<u64> {
///         let tokens: Vec<u32> = self.bpe.encode_ordinary(prompt)
///             .into_iter().map(|t| t as u32).collect();
///         llm_cache::chunk::blocks_from_tokens(&tokens, self.block_tokens)
///     }
/// }
/// ```
pub trait Chunker {
    /// Convert a prompt into block hashes for [`crate::Engine::observe`].
    fn blocks(&self, prompt: &str) -> Vec<u64>;
}

/// Tokenizer-free [`Chunker`]: fixed-size byte blocks. With ~4 bytes/token on
/// typical English text, `bytes_per_block = 4 * Config::block_tokens` keeps
/// granularity roughly token-aligned. When using this, also scale token-based
/// thresholds (`min_cacheable_tokens`) by the same approximation.
pub struct ByteChunker {
    /// Bytes per block.
    pub bytes_per_block: usize,
}

impl Chunker for ByteChunker {
    fn blocks(&self, prompt: &str) -> Vec<u64> {
        blocks_from_bytes(prompt.as_bytes(), self.bytes_per_block)
    }
}

fn hash_of(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

/// Hash a token-ID sequence into block hashes (`block_tokens` tokens each,
/// trailing partial block dropped). Use when a real tokenizer is available —
/// block boundaries then match provider-side token positions exactly.
pub fn blocks_from_tokens(tokens: &[u32], block_tokens: usize) -> Vec<u64> {
    assert!(block_tokens > 0);
    tokens
        .chunks_exact(block_tokens)
        .map(|chunk| {
            let mut h = DefaultHasher::new();
            chunk.hash(&mut h);
            h.finish()
        })
        .collect()
}

/// Hash raw prompt bytes into block hashes of `bytes_per_block` each (trailing
/// partial block dropped). A tokenizer-free approximation: with ~4 bytes per
/// token on typical English text, `bytes_per_block = 4 * block_tokens` keeps
/// block granularity roughly aligned with token blocks. Byte-exact prefix
/// sharing is what providers key on anyway, so approximate boundaries only
/// cost granularity, never correctness.
pub fn blocks_from_bytes(bytes: &[u8], bytes_per_block: usize) -> Vec<u64> {
    assert!(bytes_per_block > 0);
    bytes.chunks_exact(bytes_per_block).map(hash_of).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefixes_share_block_hashes() {
        let a: Vec<u32> = (0..1000).collect();
        let mut b = a.clone();
        b.extend(5000..5100); // divergent suffix
        let ha = blocks_from_tokens(&a, 256);
        let hb = blocks_from_tokens(&b, 256);
        assert_eq!(ha.len(), 3); // 1000 / 256 -> 3 full blocks, partial dropped
        assert_eq!(hb.len(), 4);
        assert_eq!(&ha[..3], &hb[..3]);
    }

    #[test]
    fn different_content_differs() {
        let a: Vec<u32> = (0..512).collect();
        let mut b = a.clone();
        b[0] = 9999;
        assert_ne!(
            blocks_from_tokens(&a, 256)[0],
            blocks_from_tokens(&b, 256)[0]
        );
    }

    #[test]
    fn byte_blocks_drop_partial_tail() {
        let data = vec![7u8; 1030];
        assert_eq!(blocks_from_bytes(&data, 1024).len(), 1);
    }
}
