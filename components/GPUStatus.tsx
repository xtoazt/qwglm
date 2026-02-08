'use client';

interface GPUStatusProps {
  cycle: number;
  activeThreads: number;
  totalThreads: number;
  memoryUsage: number;
  cacheHitRate?: number;
  isWebGPUAvailable?: boolean;
  backendUsage?: { webgpu: number; impossible: number };
}

export default function GPUStatus({
  cycle,
  activeThreads,
  totalThreads,
  memoryUsage,
  cacheHitRate,
  isWebGPUAvailable,
  backendUsage,
}: GPUStatusProps) {
  const threadPercentage = totalThreads > 0 ? (activeThreads / totalThreads) * 100 : 0;
  const totalOps = (backendUsage?.webgpu || 0) + (backendUsage?.impossible || 0);
  const impossiblePercentage = totalOps > 0 ? ((backendUsage?.impossible || 0) / totalOps) * 100 : 0;

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/50">
      <div className="border-b border-neutral-800 px-5 py-4">
        <h2 className="text-sm font-medium">System Status</h2>
      </div>
      
      <div className="p-5 space-y-4">
        {isWebGPUAvailable !== undefined && (
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-500">WebGPU</span>
            <div className="flex items-center gap-2">
              <div className={`h-1.5 w-1.5 rounded-full ${isWebGPUAvailable ? 'bg-neutral-100' : 'bg-neutral-700'}`} />
              <span className="text-xs text-neutral-300">
                {isWebGPUAvailable ? 'Active' : 'Unavailable'}
              </span>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className="text-xs text-neutral-500">Cycles</span>
          <span className="text-xs text-neutral-300 font-mono">{cycle.toLocaleString()}</span>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-500">Threads</span>
            <span className="text-xs text-neutral-300 font-mono">
              {activeThreads} / {totalThreads}
            </span>
          </div>
          <div className="w-full bg-neutral-900 rounded-full h-1">
            <div
              className="bg-neutral-400 h-1 rounded-full transition-all duration-300"
              style={{ width: `${threadPercentage}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-neutral-500">Memory</span>
            <span className="text-xs text-neutral-300 font-mono">{memoryUsage.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-neutral-900 rounded-full h-1">
            <div
              className="bg-neutral-100 h-1 rounded-full transition-all duration-300"
              style={{ width: `${memoryUsage}%` }}
            />
          </div>
        </div>

        {cacheHitRate !== undefined && (
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-500">Cache Hit Rate</span>
            <span className={`text-xs font-mono ${cacheHitRate === 1.0 ? 'text-neutral-100' : 'text-neutral-300'}`}>
              {(cacheHitRate * 100).toFixed(1)}%
            </span>
          </div>
        )}

        {backendUsage && totalOps > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-neutral-500">Backend Usage</span>
              <span className="text-xs text-neutral-300 font-mono">
                {impossiblePercentage.toFixed(0)}% optimized
              </span>
            </div>
            <div className="flex gap-0.5 h-1 rounded-full overflow-hidden">
              <div
                className="bg-neutral-600"
                style={{ width: `${100 - impossiblePercentage}%` }}
                title="Standard"
              />
              <div
                className="bg-neutral-100"
                style={{ width: `${impossiblePercentage}%` }}
                title="Optimized"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
