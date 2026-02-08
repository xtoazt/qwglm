/**
 * Feed-Forward Network
 * Implements the FFN component of transformer layers
 */

export interface FFNWeights {
  gateWeight: Float32Array;
  upWeight: Float32Array;
  downWeight: Float32Array;
  gateBias?: Float32Array;
  upBias?: Float32Array;
  downBias?: Float32Array;
}

/**
 * SwiGLU activation: Swish(x) * y
 */
function swiglu(x: Float32Array, y: Float32Array): Float32Array {
  const result = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    // Swish: x * sigmoid(x)
    const sigmoid = 1 / (1 + Math.exp(-x[i]));
    result[i] = (x[i] * sigmoid) * y[i];
  }
  return result;
}

/**
 * Feed-forward network forward pass
 */
export function feedForward(
  hiddenStates: Float32Array[],
  weights: FFNWeights,
  hiddenSize: number,
  intermediateSize: number
): Float32Array[] {
  const seqLen = hiddenStates.length;
  const output: Float32Array[] = [];

  for (let i = 0; i < seqLen; i++) {
    // Gate projection
    const gate = new Float32Array(intermediateSize);
    for (let j = 0; j < intermediateSize; j++) {
      let sum = weights.gateBias ? weights.gateBias[j] : 0;
      for (let k = 0; k < hiddenSize; k++) {
        sum += hiddenStates[i][k] * weights.gateWeight[j * hiddenSize + k];
      }
      gate[j] = sum;
    }

    // Up projection
    const up = new Float32Array(intermediateSize);
    for (let j = 0; j < intermediateSize; j++) {
      let sum = weights.upBias ? weights.upBias[j] : 0;
      for (let k = 0; k < hiddenSize; k++) {
        sum += hiddenStates[i][k] * weights.upWeight[j * hiddenSize + k];
      }
      up[j] = sum;
    }

    // SwiGLU activation
    const activated = swiglu(gate, up);

    // Down projection
    const down = new Float32Array(hiddenSize);
    for (let j = 0; j < hiddenSize; j++) {
      let sum = weights.downBias ? weights.downBias[j] : 0;
      for (let k = 0; k < intermediateSize; k++) {
        sum += activated[k] * weights.downWeight[j * intermediateSize + k];
      }
      down[j] = sum;
    }

    output.push(down);
  }

  return output;
}
