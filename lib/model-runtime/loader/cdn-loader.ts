/**
 * CDN Model Loader
 * Loads models from CDN with fallback support
 */

export interface CDNConfig {
  baseUrl: string;
  fallbackUrls?: string[];
  timeout?: number;
}

/**
 * Load file from CDN with fallback
 */
export async function loadFromCDN(
  path: string,
  config: CDNConfig
): Promise<ArrayBuffer> {
  const urls = [
    `${config.baseUrl}/${path}`,
    ...(config.fallbackUrls || []).map((url) => `${url}/${path}`),
  ];

  const timeout = config.timeout || 30000; // 30 seconds default

  for (const url of urls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.arrayBuffer();
    } catch (error) {
      console.warn(`Failed to load from ${url}:`, error);
      // Try next URL
      continue;
    }
  }

  throw new Error(`Failed to load ${path} from all CDN sources`);
}

/**
 * Load JSON from CDN
 */
export async function loadJSONFromCDN<T>(
  path: string,
  config: CDNConfig
): Promise<T> {
  const buffer = await loadFromCDN(path, config);
  const text = new TextDecoder().decode(buffer);
  return JSON.parse(text) as T;
}

/**
 * Predefined CDN configurations
 */
export const CDN_CONFIGS = {
  jsdelivr: {
    baseUrl: 'https://cdn.jsdelivr.net/gh',
    fallbackUrls: [
      'https://cdn.jsdelivr.net/npm',
      'https://unpkg.com',
    ],
  },
  unpkg: {
    baseUrl: 'https://unpkg.com',
    fallbackUrls: [
      'https://cdn.jsdelivr.net/npm',
    ],
  },
  cloudflare: {
    baseUrl: 'https://cdn.cloudflare.com',
  },
};
