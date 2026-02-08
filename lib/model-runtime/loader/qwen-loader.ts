/**
 * Qwen3-VL Model Loader
 * Loads Qwen3-VL-8B model from Hugging Face
 */

import { loadModelConfig, loadSafetensors, loadFileFromHF, fetchModelFiles } from './huggingface-loader';
import { loadQuantizedWeight, dequantize, type QuantizedWeight } from '../quantization/quantization';
import type { ModelWeights } from '../quantization/loader';

export interface QwenModelConfig {
  modelId: string;
  revision?: string;
  useAuthToken?: string;
  quantized?: boolean;
  quantizationBits?: 4 | 8;
}

/**
 * Load Qwen3-VL model weights from Hugging Face
 */
export async function loadQwenModel(
  config: QwenModelConfig
): Promise<{ weights: ModelWeights; modelConfig: any }> {
  const modelId = config.modelId || 'DavidAU/Qwen3-VL-8B-GLM-4.7-Flash-Heretic-Uncensored-Thinking';
  const revision = config.revision || 'main';
  
  console.log(`Loading Qwen3-VL model from ${modelId}...`);

  // Load model config
  const modelConfig = await loadModelConfig(modelId, revision);
  console.log('Model config loaded:', modelConfig);

  // Fetch available files
  const files = await fetchModelFiles(modelId, revision);
  const safetensorsFiles = files.filter((f) => f.filename.endsWith('.safetensors'));
  
  console.log(`Found ${safetensorsFiles.length} safetensors files`);

  const weights: ModelWeights = new Map();

  // Load weights from safetensors files
  for (const file of safetensorsFiles) {
    console.log(`Loading ${file.filename}...`);
    
    try {
      const tensors = await loadSafetensors(modelId, file.filename, revision);
      
      for (const [key, buffer] of tensors) {
        // Determine shape and dtype from key or config
        // For now, use default shape inference
        const shape = inferShape(key, modelConfig);
        const dtype = config.quantized 
          ? (config.quantizationBits === 4 ? 'int4' : 'int8')
          : 'fp16';
        
        // Load quantization parameters if quantized
        let scale: Float32Array | undefined;
        let zeroPoint: Uint8Array | undefined;
        
        if (config.quantized) {
          try {
            const scaleFile = file.filename.replace('.safetensors', '.scale.bin');
            const scaleBuffer = await loadFileFromHF(modelId, scaleFile, revision);
            scale = new Float32Array(scaleBuffer);
            
            const zpFile = file.filename.replace('.safetensors', '.zp.bin');
            const zpBuffer = await loadFileFromHF(modelId, zpFile, revision);
            zeroPoint = new Uint8Array(zpBuffer);
          } catch (e) {
            // Default quantization params
            scale = new Float32Array([1.0]);
            zeroPoint = new Uint8Array([0]);
          }
        }

        const quantized: QuantizedWeight = {
          data: dtype === 'fp16' 
            ? new Uint16Array(buffer)
            : new Uint8Array(buffer),
          scale: scale || new Float32Array([1.0]),
          zeroPoint,
          shape,
          dtype,
        };

        const dequantized = dequantize(quantized);
        weights.set(key, dequantized);
      }
    } catch (error) {
      console.error(`Error loading ${file.filename}:`, error);
      // Continue with other files
    }
  }

  console.log(`Loaded ${weights.size} weight tensors`);
  return { weights, modelConfig };
}

/**
 * Infer tensor shape from key name and model config
 */
function inferShape(key: string, modelConfig: any): number[] {
  // Try to infer from key name patterns
  if (key.includes('embed')) {
    return [modelConfig.vocab_size || 151936, modelConfig.hidden_size || 4096];
  }
  
  if (key.includes('attention') && key.includes('weight')) {
    if (key.includes('q_proj') || key.includes('k_proj') || key.includes('v_proj')) {
      return [modelConfig.hidden_size || 4096, modelConfig.hidden_size || 4096];
    }
    if (key.includes('o_proj')) {
      return [modelConfig.hidden_size || 4096, modelConfig.hidden_size || 4096];
    }
  }
  
  if (key.includes('mlp') || key.includes('gate_proj') || key.includes('up_proj')) {
    return [modelConfig.intermediate_size || 11008, modelConfig.hidden_size || 4096];
  }
  
  if (key.includes('down_proj')) {
    return [modelConfig.hidden_size || 4096, modelConfig.intermediate_size || 11008];
  }
  
  if (key.includes('norm') || key.includes('layer_norm')) {
    return [modelConfig.hidden_size || 4096];
  }
  
  // Default shape
  return [modelConfig.hidden_size || 4096];
}

/**
 * Load model with progress callback
 */
export async function loadQwenModelWithProgress(
  config: QwenModelConfig,
  onProgress?: (progress: { loaded: number; total: number; currentFile: string }) => void
): Promise<{ weights: ModelWeights; modelConfig: any }> {
  const modelId = config.modelId || 'DavidAU/Qwen3-VL-8B-GLM-4.7-Flash-Heretic-Uncensored-Thinking';
  const revision = config.revision || 'main';
  
  const modelConfig = await loadModelConfig(modelId, revision);
  const files = await fetchModelFiles(modelId, revision);
  const safetensorsFiles = files.filter((f) => f.filename.endsWith('.safetensors'));
  
  const totalSize = safetensorsFiles.reduce((sum, f) => sum + f.size, 0);
  let loadedSize = 0;
  
  const weights: ModelWeights = new Map();

  for (const file of safetensorsFiles) {
    if (onProgress) {
      onProgress({
        loaded: loadedSize,
        total: totalSize,
        currentFile: file.filename,
      });
    }

    try {
      const tensors = await loadSafetensors(modelId, file.filename, revision);
      
      for (const [key, buffer] of tensors) {
        const shape = inferShape(key, modelConfig);
        const dtype = config.quantized 
          ? (config.quantizationBits === 4 ? 'int4' : 'int8')
          : 'fp16';
        
        let scale: Float32Array | undefined;
        let zeroPoint: Uint8Array | undefined;
        
        if (config.quantized) {
          try {
            const scaleFile = file.filename.replace('.safetensors', '.scale.bin');
            const scaleBuffer = await loadFileFromHF(modelId, scaleFile, revision);
            scale = new Float32Array(scaleBuffer);
            
            const zpFile = file.filename.replace('.safetensors', '.zp.bin');
            const zpBuffer = await loadFileFromHF(modelId, zpFile, revision);
            zeroPoint = new Uint8Array(zpBuffer);
          } catch (e) {
            scale = new Float32Array([1.0]);
            zeroPoint = new Uint8Array([0]);
          }
        }

        const quantized: QuantizedWeight = {
          data: dtype === 'fp16' 
            ? new Uint16Array(buffer)
            : new Uint8Array(buffer),
          scale: scale || new Float32Array([1.0]),
          zeroPoint,
          shape,
          dtype,
        };

        const dequantized = dequantize(quantized);
        weights.set(key, dequantized);
      }
      
      loadedSize += file.size;
    } catch (error) {
      console.error(`Error loading ${file.filename}:`, error);
    }
  }

  if (onProgress) {
    onProgress({
      loaded: totalSize,
      total: totalSize,
      currentFile: 'Complete',
    });
  }

  return { weights, modelConfig };
}
