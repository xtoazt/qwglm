/**
 * Screen Capture
 * Chrome Screen Capture API wrapper with frame processing
 */

export interface ScreenCaptureConfig {
  video: boolean;
  audio: boolean;
  frameRate?: number;
  width?: number;
  height?: number;
}

export interface FrameProcessor {
  (frame: VideoFrame): Promise<ImageData | null>;
}

/**
 * Screen Capture Manager
 */
export class ScreenCaptureManager {
  private stream: MediaStream | null = null;
  private videoTrack: MediaStreamTrack | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private frameProcessor: FrameProcessor | null = null;
  private isCapturing = false;
  private animationFrameId: number | null = null;

  /**
   * Start screen capture
   */
  async startCapture(config: ScreenCaptureConfig = { video: true, audio: false }): Promise<boolean> {
    try {
      this.stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          frameRate: config.frameRate || 30,
          width: config.width,
          height: config.height,
        } as MediaTrackConstraints,
        audio: config.audio,
      });

      this.videoTrack = this.stream.getVideoTracks()[0];
      
      // Set up canvas for frame processing
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
      
      if (!this.ctx) {
        throw new Error('Failed to get canvas context');
      }

      // Handle track ended
      this.videoTrack.onended = () => {
        this.stopCapture();
      };

      this.isCapturing = true;
      this.startFrameProcessing();

      return true;
    } catch (error) {
      console.error('Failed to start screen capture:', error);
      return false;
    }
  }

  /**
   * Stop screen capture
   */
  stopCapture(): void {
    this.isCapturing = false;

    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.videoTrack) {
      this.videoTrack.stop();
      this.videoTrack = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.canvas = null;
    this.ctx = null;
  }

  /**
   * Start frame processing loop
   */
  private startFrameProcessing(): void {
    if (!this.videoTrack || !this.canvas || !this.ctx) {
      return;
    }

    const processFrame = async () => {
      if (!this.isCapturing || !this.videoTrack || !this.canvas || !this.ctx) {
        return;
      }

      try {
        // Create video element to capture frame
        const video = document.createElement('video');
        video.srcObject = this.stream;
        video.play();

        await new Promise((resolve) => {
          video.onloadedmetadata = () => {
            this.canvas!.width = video.videoWidth;
            this.canvas!.height = video.videoHeight;
            resolve(null);
          };
        });

        this.ctx.drawImage(video, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

        // Process frame if processor is set
        if (this.frameProcessor) {
          const videoFrame = new VideoFrame(this.canvas, { timestamp: performance.now() });
          await this.frameProcessor(videoFrame);
          videoFrame.close();
        }

        video.srcObject = null;
      } catch (error) {
        console.error('Error processing frame:', error);
      }

      this.animationFrameId = requestAnimationFrame(processFrame);
    };

    processFrame();
  }

  /**
   * Set frame processor
   */
  setFrameProcessor(processor: FrameProcessor): void {
    this.frameProcessor = processor;
  }

  /**
   * Get current frame as ImageData
   */
  getCurrentFrame(): ImageData | null {
    if (!this.canvas || !this.ctx) {
      return null;
    }

    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Get current frame as blob
   */
  async getCurrentFrameAsBlob(): Promise<Blob | null> {
    if (!this.canvas) {
      return null;
    }

    return new Promise((resolve) => {
      this.canvas!.toBlob((blob) => {
        resolve(blob);
      });
    });
  }

  /**
   * Check if capturing
   */
  isCapturingActive(): boolean {
    return this.isCapturing;
  }

  /**
   * Get stream
   */
  getStream(): MediaStream | null {
    return this.stream;
  }
}

/**
 * Process frame for vision model
 * Resizes and normalizes image for model input
 */
export async function processFrameForVision(
  frame: VideoFrame,
  targetSize: number = 224
): Promise<ImageData> {
  const canvas = document.createElement('canvas');
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Failed to get canvas context');
  }

  // Draw and resize frame
  ctx.drawImage(frame, 0, 0, targetSize, targetSize);

  // Get image data
  return ctx.getImageData(0, 0, targetSize, targetSize);
}
