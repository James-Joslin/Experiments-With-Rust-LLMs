use std::sync::Arc;
use tokenizers::Tokenizer;
use anyhow::{Result, anyhow}; // Import `anyhow` macro

/// A wrapper struct for managing tokenization and token history.
///
/// This struct holds a tokenizer, a vector of tokens, and a history of token vectors.
///
/// # Fields
///
/// * `tokenizer` - An `Arc<Tokenizer>` that provides the tokenization functionality.
/// * `tokens` - A vector of `i64` tokens currently being processed.
/// * `history` - A vector of vectors, where each inner vector represents a history of tokens.
pub struct TokenizerWrapper {
    tokenizer: Arc<Tokenizer>,
    tokens: Vec<i64>,
    history: Vec<Vec<i64>>,  // Add a field for storing history
}

impl TokenizerWrapper {
    /// Creates a new instance of the struct with the tokenizer loaded from the specified path.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_path` - A string slice that holds the path to the tokenizer file.
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - A result containing the new instance or an error if the tokenizer fails to load.
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from file: {}. Error: {}", tokenizer_path, e))?;
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            tokens: Vec::new(),
            history: Vec::new(),  // Initialize history as an empty Vec
        })
    }
 
    /// Uses the tokenizer to encode the provided text and converts the resulting
    /// token IDs to `i64` before appending them to the `tokens` vector.
    /// 
    /// # Arguments
    /// * `text` - A string slice that holds the text to be tokenized.
    /// 
    /// # Errors
    /// If the tokenizer fails to encode the text, an error message is logged.
    pub fn store_initial_tokenize(&mut self, text: &str) {
        if let Ok(encoding) = self.tokenizer.encode(text, true) {
            self.tokens.extend(encoding.get_ids().iter().map(|&id| id as i64));
        } else {
            // Handle the error appropriately, e.g., log it or return a Result
            eprintln!("Failed to encode text");
        }
    }

    /// Returns a vector of tokens for the model.
    /// 
    /// This function combines the tokens from the history and the current tokens.
    /// If the history is not empty, it flattens the history (a vector of vectors)
    /// and appends the current tokens to it. If the history is empty, it simply
    /// returns a clone of the current tokens.
    /// 
    /// # Returns
    /// 
    /// A vector of `i64` tokens.
    pub fn get_tokens_for_model(&self) -> Vec<i64> {
        let mut returned_tokens: Vec<i64> = Vec::new();
        if !self.history.is_empty() {
            // Flatten the history vector of vectors
            for vec in &self.history {
                returned_tokens.extend(vec.iter().cloned());
            }
            // Add the tokens
            returned_tokens.extend(self.tokens.iter().cloned());
        } else {
            returned_tokens = self.tokens.clone();
        }
        returned_tokens
    }
    
    /// Appends a token to the internal tokens vector.
    ///
    /// This function adds the provided `token` to the `tokens` vector.
    ///
    /// # Arguments
    ///
    /// * `token` - An `i64` value representing the token to be appended.
    pub fn append_token(&mut self, token: i64) {
        self.tokens.push(token);
        println!("");
    }

    // /// Appends the current tokens to the history and clears the tokens vector.
    // ///
    // /// This function clones the current `tokens` vector and appends it to the `history` vector.
    // /// After appending, it clears the `tokens` vector to prepare for new tokens.
    // pub fn append_to_history(&mut self) {
    //     self.history.push(self.tokens.clone());
    //     self.tokens.clear();
    // }

    // pub fn decode(&self, tokens: &Array<u32, ndarray::IxDyn>) -> Result<String> {
    //     let token_vec: Vec<u32> = tokens.iter().cloned().collect();
    //     self.tokenizer.decode(&token_vec, true)
    //         .map_err(|e| anyhow!("Decoding failed for tokens: {:?}. Error: {}", token_vec, e))
    // }
}
