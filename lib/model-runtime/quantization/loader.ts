/**
 * Model Weight Loader
 * Loads quantized model weights from files
 */

import { loadQuantizedWeight, dequantize, type QuantizedWeight } from './quantization';

export type ModelWeights = Map<string, Float32Array>;

/**
 * Load model weights from a quantized format
 */
export async function loadModelWeights(
  weightFiles: Map<string, { url: string; shape: number[]; dtype: string }>
): Promise<ModelWeights> {
  const weights: ModelWeights = new Map();

  for (const [name, fileInfo] of weightFiles) {
    try {
      const response = await fetch(fileInfo.url);
      const buffer = await response.arrayBuffer();

      // Parse dtype
      const dtype = fileInfo.dtype as 'int4' | 'int8' | 'fp16' | 'fp32';

      // Load scale and zero point if needed
      let scale: Float32Array | undefined;
      let zeroPoint: Uint8Array | undefined;

      if (dtype === 'int4' || dtype === 'int8') {
        // Try to load scale and zero point files
        try {
          const scaleResponse = await fetch(fileInfo.url.replace('.bin', '.scale.bin'));
          const scaleBuffer = await scaleResponse.arrayBuffer();
          scale = new Float32Array(scaleBuffer);

          const zpResponse = await fetch(fileInfo.url.replace('.bin', '.zp.bin'));
          const zpBuffer = await zpResponse.arrayBuffer();
          zeroPoint = new Uint8Array(zpBuffer);
        } catch (e) {
          // Default scale and zero point
          scale = new Float32Array([1.0]);
          zeroPoint = new Uint8Array([0]);
        }
      }

      const quantized = loadQuantizedWeight(
        buffer,
        0,
        fileInfo.shape,
        dtype,
        scale,
        zeroPoint
      );

      const dequantized = dequantize(quantized);
      weights.set(name, dequantized);
    } catch (error) {
      console.error(`Failed to load weight ${name}:`, error);
      throw error;
    }
  }

  return weights;
}

/**
 * Load weights from Hugging Face format
 */
export async function loadHuggingFaceWeights(
  modelPath: string,
  weightManifest: any
): Promise<ModelWeights> {
  const weights: ModelWeights = new Map();

  for (const [name, info] of Object.entries(weightManifest)) {
    const fileInfo = info as { filename: string; shape: number[]; dtype: string };
    const url = `${modelPath}/${fileInfo.filename}`;

    try {
      const response = await fetch(url);
      const buffer = await response.arrayBuffer();

      const dtype = fileInfo.dtype as 'int4' | 'int8' | 'fp16' | 'fp32';
      let scale: Float32Array | undefined;
      let zeroPoint: Uint8Array | undefined;

      if (dtype === 'int4' || dtype === 'int8') {
        // Load quantization parameters
        try {
          const scaleUrl = url.replace('.safetensors', '.scale.bin');
          const scaleResponse = await fetch(scaleUrl);
          scale = new Float32Array(await scaleResponse.arrayBuffer());

          const zpUrl = url.replace('.safetensors', '.zp.bin');
          const zpResponse = await fetch(zpUrl);
          zeroPoint = new Uint8Array(await zpResponse.arrayBuffer());
        } catch (e) {
          scale = new Float32Array([1.0]);
          zeroPoint = new Uint8Array([0]);
        }
      }

      const quantized = loadQuantizedWeight(
        buffer,
        0,
        fileInfo.shape,
        dtype,
        scale,
        zeroPoint
      );

      const dequantized = dequantize(quantized);
      weights.set(name, dequantized);
    } catch (error) {
      console.error(`Failed to load weight ${name}:`, error);
      throw error;
    }
  }

  return weights;
}
