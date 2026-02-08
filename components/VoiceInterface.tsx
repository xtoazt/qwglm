'use client';

import { useState, useEffect } from 'react';

interface VoiceInterfaceProps {
  onStartRecording: () => Promise<void>;
  onStopRecording: () => Promise<void>;
  isRecording: boolean;
  isSpeaking: boolean;
}

export default function VoiceInterface({
  onStartRecording,
  onStopRecording,
  isRecording,
  isSpeaking,
}: VoiceInterfaceProps) {
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Initialize voice interface
    setIsInitialized(true);
  }, []);

  const handleToggleRecording = async () => {
    if (isRecording) {
      await onStopRecording();
    } else {
      await onStartRecording();
    }
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950/50 p-4">
      <h3 className="text-xs font-medium text-neutral-400 uppercase tracking-wider mb-4">
        Voice
      </h3>
      
      <button
        onClick={handleToggleRecording}
        disabled={!isInitialized || isSpeaking}
        className={`w-full px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
          isRecording
            ? 'bg-neutral-100 text-neutral-900 hover:bg-neutral-200'
            : 'bg-neutral-900 border border-neutral-800 text-neutral-300 hover:bg-neutral-800 hover:border-neutral-700'
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        {isRecording ? 'Stop Recording' : 'Start Recording'}
      </button>
      
      {(isRecording || isSpeaking) && (
        <div className="mt-3 flex items-center gap-2 text-xs text-neutral-500">
          <div className="h-1.5 w-1.5 rounded-full bg-neutral-400 animate-pulse" />
          <span>{isRecording ? 'Recording...' : 'Speaking...'}</span>
        </div>
      )}
    </div>
  );
}
