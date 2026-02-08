/**
 * Thread Scheduler
 * Ported from tiny-gpu Verilog thread scheduler
 * Manages thread execution and warp scheduling
 */

import type { ThreadState, BlockState } from '../types';

export interface Warp {
  threads: ThreadState[];
  active: boolean;
  pc: number;
}

export class ThreadScheduler {
  private warps: Warp[] = [];
  private readonly warpSize = 32; // Threads per warp
  private currentWarpIndex = 0;
  private cycle = 0;

  /**
   * Initialize threads from blocks
   */
  initializeBlocks(blocks: BlockState[]): void {
    this.warps = [];
    this.currentWarpIndex = 0;

    for (const block of blocks) {
      // Split threads into warps
      for (let i = 0; i < block.threads.length; i += this.warpSize) {
        const warpThreads = block.threads.slice(i, i + this.warpSize);
        this.warps.push({
          threads: warpThreads,
          active: true,
          pc: warpThreads[0]?.pc || 0,
        });
      }
    }
  }

  /**
   * Get next warp to execute
   * Implements round-robin scheduling
   */
  getNextWarp(): Warp | null {
    if (this.warps.length === 0) {
      return null;
    }

    // Round-robin scheduling
    let attempts = 0;
    while (attempts < this.warps.length) {
      const warp = this.warps[this.currentWarpIndex];
      this.currentWarpIndex = (this.currentWarpIndex + 1) % this.warps.length;

      if (warp.active && warp.threads.some((t) => t.active)) {
        return warp;
      }

      attempts++;
    }

    return null;
  }

  /**
   * Check if all threads in a warp have the same PC
   * (no divergence)
   */
  isWarpDiverged(warp: Warp): boolean {
    if (warp.threads.length === 0) return false;

    const firstPC = warp.threads[0].pc;
    return warp.threads.some((t) => t.active && t.pc !== firstPC);
  }

  /**
   * Get active threads from warp
   */
  getActiveThreads(warp: Warp): ThreadState[] {
    return warp.threads.filter((t) => t.active);
  }

  /**
   * Update warp PC
   */
  updateWarpPC(warp: Warp, newPC: number): void {
    warp.pc = newPC;
    for (const thread of warp.threads) {
      if (thread.active) {
        thread.pc = newPC;
      }
    }
  }

  /**
   * Mark thread as inactive
   */
  deactivateThread(threadId: number, blockId: number): void {
    for (const warp of this.warps) {
      for (const thread of warp.threads) {
        if (thread.id === threadId && thread.blockId === blockId) {
          thread.active = false;
        }
      }
    }
  }

  /**
   * Check if all warps are complete
   */
  isComplete(): boolean {
    return this.warps.every(
      (warp) => !warp.active || warp.threads.every((t) => !t.active)
    );
  }

  /**
   * Get scheduler statistics
   */
  getStats(): {
    totalWarps: number;
    activeWarps: number;
    totalThreads: number;
    activeThreads: number;
  } {
    const activeWarps = this.warps.filter(
      (w) => w.active && w.threads.some((t) => t.active)
    ).length;
    const totalThreads = this.warps.reduce(
      (sum, w) => sum + w.threads.length,
      0
    );
    const activeThreads = this.warps.reduce(
      (sum, w) => sum + w.threads.filter((t) => t.active).length,
      0
    );

    return {
      totalWarps: this.warps.length,
      activeWarps,
      totalThreads,
      activeThreads,
    };
  }

  /**
   * Reset scheduler
   */
  reset(): void {
    this.warps = [];
    this.currentWarpIndex = 0;
    this.cycle = 0;
  }

  /**
   * Increment cycle counter
   */
  tick(): void {
    this.cycle++;
  }

  /**
   * Get current cycle
   */
  getCycle(): number {
    return this.cycle;
  }
}
