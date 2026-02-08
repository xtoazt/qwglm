/**
 * Parakeet TTS Model Loader
 * Loads NVIDIA Parakeet-tdt-0.6b-v2 from Hugging Face
 */

export interface ParakeetModelConfig {
  modelId?: string;
  revision?: string;
  useAuthToken?: string;
}

/**
 * Load Parakeet model from Hugging Face
 */
export async function loadParakeetModel(
  config: ParakeetModelConfig = {}
): Promise<{
  model: ArrayBuffer;
  config: any;
  tokenizer: any;
}> {
  const modelId = config.modelId || 'nvidia/parakeet-tdt-0.6b-v2';
  const revision = config.revision || 'main';

  console.log(`Loading Parakeet model from ${modelId}...`);

  const headers: HeadersInit = {};
  if (config.useAuthToken) {
    headers['Authorization'] = `Bearer ${config.useAuthToken}`;
  }

  // Load model files
  const modelUrl = `https://huggingface.co/${modelId}/resolve/${revision}/pytorch_model.bin`;
  const configUrl = `https://huggingface.co/${modelId}/raw/${revision}/config.json`;
  const tokenizerUrl = `https://huggingface.co/${modelId}/raw/${revision}/tokenizer.json`;

  const [modelResponse, configResponse, tokenizerResponse] = await Promise.all([
    fetch(modelUrl, { headers }),
    fetch(configUrl, { headers }),
    fetch(tokenizerUrl, { headers }).catch(() => null),
  ]);

  if (!modelResponse.ok) {
    throw new Error(`Failed to load Parakeet model: ${modelResponse.statusText}`);
  }

  const model = await modelResponse.arrayBuffer();
  const modelConfig = await configResponse.json();
  const tokenizer = tokenizerResponse?.ok ? await tokenizerResponse.json() : null;

  return { model, config: modelConfig, tokenizer };
}
