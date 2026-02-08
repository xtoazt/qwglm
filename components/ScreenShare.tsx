'use client';

import { useState } from 'react';

interface ScreenShareProps {
  onStartCapture: () => Promise<boolean>;
  onStopCapture: () => void;
  isCapturing: boolean;
}

export default function ScreenShare({
  onStartCapture,
  onStopCapture,
  isCapturing,
}: ScreenShareProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleToggleCapture = async () => {
    if (isCapturing) {
      onStopCapture();
    } else {
      setIsLoading(true);
      try {
        await onStartCapture();
      } catch (error) {
        console.error('Screen capture error:', error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/50 p-4">
      <h3 className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-4">
        Screen Share
      </h3>
      
      <button
        onClick={handleToggleCapture}
        disabled={isLoading}
        className={`w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
          isCapturing
            ? 'bg-neutral-100 text-neutral-900 hover:bg-neutral-200'
            : 'bg-neutral-900 border border-neutral-800 text-neutral-300 hover:bg-neutral-800 hover:border-neutral-700'
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {isCapturing ? 'Stop Sharing' : 'Share Screen'}
      </button>
      
      {isCapturing && (
        <div className="mt-3 flex items-center gap-2 text-xs text-neutral-500">
          <div className="h-1.5 w-1.5 rounded-full bg-neutral-400 animate-pulse" />
          <span>Screen sharing active</span>
        </div>
      )}
    </div>
  );
}
