/**
 * Cache Implementation
 * Ported from tiny-gpu Verilog cache
 * Implements a simple direct-mapped cache
 */

import type { Word, Address, CacheLine, MemoryConfig } from '../types';

export class Cache {
  private lines: CacheLine[];
  private config: MemoryConfig;
  private hits: number = 0;
  private misses: number = 0;

  constructor(config: MemoryConfig) {
    this.config = config;
    this.lines = new Array(config.cacheSize);
    for (let i = 0; i < config.cacheSize; i++) {
      this.lines[i] = {
        tag: 0,
        data: new Array(config.lineSize).fill(0),
        valid: false,
        dirty: false,
      };
    }
  }

  /**
   * Read from cache
   * Returns { data, hit } where hit indicates cache hit
   */
  read(address: Address, memory: Word[]): { data: Word; hit: boolean } {
    const lineIndex = this.getLineIndex(address);
    const tag = this.getTag(address);
    const offset = this.getOffset(address);

    const line = this.lines[lineIndex];

    if (line.valid && line.tag === tag) {
      // Cache hit
      this.hits++;
      return { data: line.data[offset], hit: true };
    }

    // Cache miss - load from memory
    this.misses++;
    this.loadLine(address, memory);
    return { data: this.lines[lineIndex].data[offset], hit: false };
  }

  /**
   * Write to cache
   * Returns true if write-through, false if write-back
   */
  write(address: Address, data: Word, memory: Word[]): boolean {
    const lineIndex = this.getLineIndex(address);
    const tag = this.getTag(address);
    const offset = this.getOffset(address);

    const line = this.lines[lineIndex];

    if (line.valid && line.tag === tag) {
      // Cache hit - write to cache
      line.data[offset] = data;
      line.dirty = true;
      this.hits++;
      // Write-through for simplicity
      memory[address] = data;
      return true;
    }

    // Cache miss - load line first
    this.misses++;
    this.loadLine(address, memory);
    this.lines[lineIndex].data[offset] = data;
    this.lines[lineIndex].dirty = true;
    memory[address] = data;
    return true;
  }

  /**
   * Load a cache line from memory
   */
  private loadLine(address: Address, memory: Word[]): void {
    const lineIndex = this.getLineIndex(address);
    const tag = this.getTag(address);
    const baseAddress = (address >> 2) << 2; // Align to line boundary

    const line = this.lines[lineIndex];

    // Write back if dirty
    if (line.valid && line.dirty) {
      const oldBaseAddress = (line.tag << 2) | (lineIndex << 2);
      for (let i = 0; i < this.config.lineSize; i++) {
        if (oldBaseAddress + i < memory.length) {
          memory[oldBaseAddress + i] = line.data[i];
        }
      }
    }

    // Load new line
    for (let i = 0; i < this.config.lineSize; i++) {
      const addr = baseAddress + i;
      if (addr < memory.length) {
        line.data[i] = memory[addr];
      } else {
        line.data[i] = 0;
      }
    }

    line.tag = tag;
    line.valid = true;
    line.dirty = false;
  }

  /**
   * Get cache line index from address
   */
  private getLineIndex(address: Address): number {
    return (address >> 2) % this.config.cacheSize;
  }

  /**
   * Get tag from address
   */
  private getTag(address: Address): Address {
    return address >> (2 + Math.log2(this.config.cacheSize));
  }

  /**
   * Get offset within cache line
   */
  private getOffset(address: Address): number {
    return address % this.config.lineSize;
  }

  /**
   * Flush cache (write back all dirty lines)
   */
  flush(memory: Word[]): void {
    for (let i = 0; i < this.lines.length; i++) {
      const line = this.lines[i];
      if (line.valid && line.dirty) {
        const baseAddress = (line.tag << 2) | (i << 2);
        for (let j = 0; j < this.config.lineSize; j++) {
          const addr = baseAddress + j;
          if (addr < memory.length) {
            memory[addr] = line.data[j];
          }
        }
        line.dirty = false;
      }
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): { hits: number; misses: number; hitRate: number } {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  /**
   * Reset cache statistics
   */
  resetStats(): void {
    this.hits = 0;
    this.misses = 0;
  }
}
