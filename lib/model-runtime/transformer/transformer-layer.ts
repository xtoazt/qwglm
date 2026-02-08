/**
 * Transformer Layer
 * Complete transformer decoder layer for Qwen3-VL
 */

import { layerNorm } from './layer-norm';
import { multiHeadAttention, type AttentionWeights, type AttentionConfig } from './attention';
import { feedForward, type FFNWeights } from './feed-forward';

export interface TransformerLayerWeights {
  attention: AttentionWeights;
  ffn: FFNWeights;
  ln1Gamma: Float32Array;
  ln1Beta: Float32Array;
  ln2Gamma: Float32Array;
  ln2Beta: Float32Array;
}

export interface TransformerLayerConfig {
  hiddenSize: number;
  numHeads: number;
  headDim: number;
  intermediateSize: number;
  attention: AttentionConfig;
}

/**
 * Transformer decoder layer forward pass
 */
export function transformerLayer(
  hiddenStates: Float32Array[],
  weights: TransformerLayerWeights,
  config: TransformerLayerConfig,
  kvCache?: { key: Float32Array[]; value: Float32Array[] }
): {
  output: Float32Array[];
  newKvCache?: { key: Float32Array[]; value: Float32Array[] };
} {
  // Pre-attention layer norm
  const norm1 = hiddenStates.map((state) =>
    layerNorm(state, weights.ln1Gamma, weights.ln1Beta)
  );

  // Self-attention
  const { output: attnOut, newKvCache } = multiHeadAttention(
    norm1,
    weights.attention,
    config.attention,
    kvCache
  );

  // Residual connection
  const residual1 = hiddenStates.map((state, i) => {
    const out = new Float32Array(state.length);
    for (let j = 0; j < state.length; j++) {
      out[j] = state[j] + attnOut[i][j];
    }
    return out;
  });

  // Pre-FFN layer norm
  const norm2 = residual1.map((state) =>
    layerNorm(state, weights.ln2Gamma, weights.ln2Beta)
  );

  // Feed-forward network
  const ffnOut = feedForward(
    norm2,
    weights.ffn,
    config.hiddenSize,
    config.intermediateSize
  );

  // Residual connection
  const output = residual1.map((state, i) => {
    const out = new Float32Array(state.length);
    for (let j = 0; j < state.length; j++) {
      out[j] = state[j] + ffnOut[i][j];
    }
    return out;
  });

  return {
    output,
    newKvCache,
  };
}
