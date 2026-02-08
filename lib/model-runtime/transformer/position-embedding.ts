/**
 * Position Embeddings
 * Implements rotary position embeddings (RoPE) for Qwen3
 */

export interface PositionEmbeddingConfig {
  hiddenSize: number;
  numHeads: number;
  maxSeqLen: number;
  ropeTheta: number;
}

/**
 * Generate rotary position embeddings
 */
export function generateRotaryEmbeddings(
  config: PositionEmbeddingConfig
): { cos: Float32Array[]; sin: Float32Array[] } {
  const { hiddenSize, numHeads, maxSeqLen, ropeTheta } = config;
  const headDim = hiddenSize / numHeads;
  const numFreqs = headDim / 2;

  const cos: Float32Array[] = [];
  const sin: Float32Array[] = [];

  for (let pos = 0; pos < maxSeqLen; pos++) {
    const cosRow = new Float32Array(headDim);
    const sinRow = new Float32Array(headDim);

    for (let i = 0; i < numFreqs; i++) {
      const freq = 1.0 / (ropeTheta ** (2 * i / headDim));
      const angle = pos * freq;
      cosRow[2 * i] = Math.cos(angle);
      cosRow[2 * i + 1] = Math.cos(angle);
      sinRow[2 * i] = Math.sin(angle);
      sinRow[2 * i + 1] = Math.sin(angle);
    }

    cos.push(cosRow);
    sin.push(sinRow);
  }

  return { cos, sin };
}

/**
 * Apply rotary position embeddings to query/key
 */
export function applyRotaryEmbedding(
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array
): Float32Array {
  const headDim = x.length;
  const result = new Float32Array(headDim);

  for (let i = 0; i < headDim; i += 2) {
    const x0 = x[i];
    const x1 = x[i + 1];
    const cosVal = cos[i];
    const sinVal = sin[i];

    result[i] = x0 * cosVal - x1 * sinVal;
    result[i + 1] = x0 * sinVal + x1 * cosVal;
  }

  return result;
}
