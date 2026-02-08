/**
 * Layer Normalization
 * Implements layer normalization for transformer layers
 */

export function layerNorm(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number = 1e-5
): Float32Array {
  const n = x.length;
  const mean = x.reduce((sum, val) => sum + val, 0) / n;
  const variance =
    x.reduce((sum, val) => sum + (val - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance + eps);

  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = (x[i] - mean) / std * gamma[i] + beta[i];
  }

  return result;
}

export function layerNorm2D(
  x: Float32Array[],
  gamma: Float32Array,
  beta: Float32Array,
  eps: number = 1e-5
): Float32Array[] {
  return x.map((row) => layerNorm(row, gamma, beta, eps));
}
