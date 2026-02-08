/**
 * Hugging Face Model Loader
 * Loads models from Hugging Face Hub
 */

export interface HuggingFaceModelConfig {
  modelId: string;
  revision?: string;
  useAuthToken?: string;
}

export interface ModelFile {
  filename: string;
  size: number;
  sha256?: string;
}

/**
 * Fetch model files list from Hugging Face
 */
export async function fetchModelFiles(
  modelId: string,
  revision: string = 'main'
): Promise<ModelFile[]> {
  const url = `https://huggingface.co/api/models/${modelId}/tree/${revision}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch model files: ${response.statusText}`);
    }
    
    const files = await response.json();
    return files
      .filter((file: any) => file.type === 'file')
      .map((file: any) => ({
        filename: file.path,
        size: file.size,
        sha256: file.lfs?.sha256,
      }));
  } catch (error) {
    console.error('Error fetching model files:', error);
    throw error;
  }
}

/**
 * Load file from Hugging Face
 */
export async function loadFileFromHF(
  modelId: string,
  filename: string,
  revision: string = 'main',
  useAuthToken?: string
): Promise<ArrayBuffer> {
  const url = `https://huggingface.co/${modelId}/resolve/${revision}/${filename}`;
  
  const headers: HeadersInit = {};
  if (useAuthToken) {
    headers['Authorization'] = `Bearer ${useAuthToken}`;
  }

  try {
    const response = await fetch(url, { headers });
    if (!response.ok) {
      throw new Error(`Failed to load file ${filename}: ${response.statusText}`);
    }
    
    return await response.arrayBuffer();
  } catch (error) {
    console.error(`Error loading file ${filename}:`, error);
    throw error;
  }
}

/**
 * Load safetensors file
 * Safetensors format: [8 bytes header size][header JSON][tensor data...]
 */
export async function loadSafetensors(
  modelId: string,
  filename: string,
  revision: string = 'main'
): Promise<Map<string, ArrayBuffer>> {
  const buffer = await loadFileFromHF(modelId, filename, revision);
  
  // Parse safetensors format
  // Safetensors header: 8 bytes (header size) + JSON header + aligned data
  const view = new DataView(buffer);
  
  // Read header size (first 8 bytes)
  const headerSize = Number(view.getBigUint64(0, true));
  
  if (headerSize > buffer.byteLength || headerSize < 8) {
    throw new Error('Invalid safetensors header size');
  }
  
  // Read JSON header
  const headerBytes = buffer.slice(8, 8 + Number(headerSize));
  const headerJson = new TextDecoder().decode(headerBytes);
  let header: Record<string, any>;
  
  try {
    header = JSON.parse(headerJson);
  } catch (error) {
    throw new Error('Failed to parse safetensors header JSON');
  }
  
  // Calculate data offset (8 bytes + header size, aligned to 8 bytes)
  const dataOffset = 8 + Number(headerSize);
  const alignedOffset = Math.ceil(dataOffset / 8) * 8;
  
  const tensors = new Map<string, ArrayBuffer>();
  
  // Extract each tensor
  for (const [key, metadata] of Object.entries(header)) {
    if (typeof metadata !== 'object' || !metadata.data_offsets) {
      continue;
    }
    
    const meta = metadata as { 
      dtype: string; 
      shape: number[]; 
      data_offsets: [number, number] 
    };
    const [start, end] = meta.data_offsets;
    
    if (start < 0 || end > buffer.byteLength || start >= end) {
      console.warn(`Invalid tensor offsets for ${key}: [${start}, ${end}]`);
      continue;
    }
    
    const tensorData = buffer.slice(alignedOffset + start, alignedOffset + end);
    tensors.set(key, tensorData);
  }
  
  return tensors;
}

/**
 * Load model config from Hugging Face
 */
export async function loadModelConfig(
  modelId: string,
  revision: string = 'main'
): Promise<any> {
  const url = `https://huggingface.co/${modelId}/raw/${revision}/config.json`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load config: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error loading model config:', error);
    throw error;
  }
}

/**
 * Load tokenizer config
 */
export async function loadTokenizerConfig(
  modelId: string,
  revision: string = 'main'
): Promise<any> {
  const url = `https://huggingface.co/${modelId}/raw/${revision}/tokenizer_config.json`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      // Tokenizer config might not exist, return null
      return null;
    }
    
    return await response.json();
  } catch (error) {
    console.warn('Tokenizer config not found, using defaults');
    return null;
  }
}
