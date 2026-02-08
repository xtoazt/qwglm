/**
 * Tokenizer
 * Proper BPE tokenizer for Qwen3
 */

export interface Tokenizer {
  encode(text: string): number[];
  decode(tokens: number[]): string;
  vocabSize: number;
}

export interface TokenizerConfig {
  vocab: Record<string, number>;
  merges: string[];
  specialTokens?: Record<string, number>;
}

/**
 * BPE Tokenizer for Qwen3
 */
export class QwenTokenizer implements Tokenizer {
  vocabSize: number;
  private vocab: Map<string, number>;
  private reverseVocab: Map<number, string>;
  private merges: Map<string, number>;
  private specialTokens: Map<string, number>;
  private pattern: RegExp;

  constructor(config?: TokenizerConfig) {
    this.vocab = new Map();
    this.reverseVocab = new Map();
    this.merges = new Map();
    this.specialTokens = new Map();
    
    // Qwen3 pattern for tokenization
    this.pattern = /[\p{L}\p{N}]+|[\p{P}]+|\s+/gu;

    if (config) {
      // Load vocab
      for (const [token, id] of Object.entries(config.vocab)) {
        this.vocab.set(token, id);
        this.reverseVocab.set(id, token);
      }

      // Load merges
      config.merges.forEach((merge, idx) => {
        this.merges.set(merge, idx);
      });

      // Load special tokens
      if (config.specialTokens) {
        for (const [token, id] of Object.entries(config.specialTokens)) {
          this.specialTokens.set(token, id);
        }
      }
    } else {
      // Initialize with default vocab (for testing)
      this.initializeDefaultVocab();
    }

    this.vocabSize = this.vocab.size;
  }

  /**
   * Initialize default vocabulary
   */
  private initializeDefaultVocab(): void {
    // Add special tokens
    this.vocab.set('<|endoftext|>', 151643);
    this.vocab.set('<|im_start|>', 151644);
    this.vocab.set('<|im_end|>', 151645);
    this.reverseVocab.set(151643, '<|endoftext|>');
    this.reverseVocab.set(151644, '<|im_start|>');
    this.reverseVocab.set(151645, '<|im_end|>');

    // Add byte-level tokens (0-255)
    for (let i = 0; i < 256; i++) {
      const token = String.fromCharCode(i);
      this.vocab.set(token, i);
      this.reverseVocab.set(i, token);
    }

    // Add common subwords (simplified)
    const commonTokens = [
      'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'for',
      'with', 'as', 'on', 'was', 'at', 'by', 'be', 'this', 'an', 'or',
    ];

    let id = 256;
    for (const token of commonTokens) {
      if (!this.vocab.has(token)) {
        this.vocab.set(token, id);
        this.reverseVocab.set(id, token);
        id++;
      }
    }
  }

  /**
   * Apply BPE merges
   */
  private applyBPE(token: string): string[] {
    if (token.length === 1) {
      return [token];
    }

    let pairs = this.getPairs(token.split(''));
    if (pairs.length === 0) {
      return [token];
    }

    while (true) {
      const minPair = this.getMinPair(pairs);
      if (!minPair) break;

      const [first, second] = minPair.split(' ');
      const newToken: string[] = [];
      let i = 0;
      const chars = token.split('');

      while (i < chars.length) {
        const j = chars.indexOf(first, i);
        if (j === -1) {
          newToken.push(...chars.slice(i));
          break;
        }

        newToken.push(...chars.slice(i, j));
        i = j;

        if (i < chars.length - 1 && chars[i] === first && chars[i + 1] === second) {
          newToken.push(first + second);
          i += 2;
        } else {
          newToken.push(chars[i]);
          i += 1;
        }
      }

      token = newToken.join('');
      if (newToken.length === 1) break;
      pairs = this.getPairs(newToken);
    }

    return token.split('');
  }

  /**
   * Get character pairs
   */
  private getPairs(chars: string[]): string[] {
    const pairs: string[] = [];
    for (let i = 0; i < chars.length - 1; i++) {
      pairs.push(`${chars[i]} ${chars[i + 1]}`);
    }
    return pairs;
  }

  /**
   * Get minimum rank pair
   */
  private getMinPair(pairs: string[]): string | null {
    let minPair: string | null = null;
    let minRank = Infinity;

    for (const pair of pairs) {
      const rank = this.merges.get(pair);
      if (rank !== undefined && rank < minRank) {
        minRank = rank;
        minPair = pair;
      }
    }

    return minPair;
  }

  /**
   * Encode text to token IDs
   */
  encode(text: string): number[] {
    const tokens: number[] = [];

    // Check for special tokens
    for (const [specialToken, id] of this.specialTokens) {
      if (text.startsWith(specialToken)) {
        tokens.push(id);
        text = text.slice(specialToken.length);
      }
    }

    // Tokenize text
    const matches = text.match(this.pattern) || [];

    for (const match of matches) {
      // Apply BPE
      const bpeTokens = this.applyBPE(match);

      for (const token of bpeTokens) {
        const id = this.vocab.get(token);
        if (id !== undefined) {
          tokens.push(id);
        } else {
          // Fallback to byte encoding
          for (const char of token) {
            const charCode = char.charCodeAt(0);
            if (charCode < 256) {
              tokens.push(charCode);
            } else {
              tokens.push(0); // UNK token
            }
          }
        }
      }
    }

    return tokens;
  }

  /**
   * Decode token IDs to text
   */
  decode(tokens: number[]): string {
    const pieces: string[] = [];

    for (const token of tokens) {
      const piece = this.reverseVocab.get(token);
      if (piece) {
        // Skip special tokens in output
        if (!piece.startsWith('<|') || !piece.endsWith('|>')) {
          pieces.push(piece);
        }
      }
    }

    return pieces.join('');
  }

  /**
   * Load from tokenizer.json
   */
  static async fromJSON(json: any): Promise<QwenTokenizer> {
    const config: TokenizerConfig = {
      vocab: json.model?.vocab || {},
      merges: json.model?.merges || [],
      specialTokens: json.added_tokens?.reduce((acc: any, token: any) => {
        acc[token.content] = token.id;
        return acc;
      }, {}),
    };

    return new QwenTokenizer(config);
  }
}
