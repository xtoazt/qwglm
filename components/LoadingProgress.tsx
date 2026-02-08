'use client';

interface LoadingProgressProps {
  progress: number;
  currentFile?: string;
  message?: string;
}

export default function LoadingProgress({
  progress,
  currentFile,
  message,
}: LoadingProgressProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="rounded-lg border border-neutral-800 bg-neutral-950 max-w-md w-full mx-4">
        <div className="border-b border-neutral-800 px-5 py-4">
          <h2 className="text-sm font-medium">Loading Model</h2>
        </div>
        
        <div className="p-5 space-y-4">
          {message && (
            <p className="text-sm text-neutral-500">{message}</p>
          )}
          
          {currentFile && (
            <div className="space-y-1">
              <div className="text-xs text-neutral-500">Current File</div>
              <div className="text-sm text-neutral-300 font-mono truncate">{currentFile}</div>
            </div>
          )}

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-neutral-500">Progress</span>
              <span className="text-xs text-neutral-300 font-mono">{progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-neutral-900 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-neutral-100 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(progress, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
