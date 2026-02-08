/**
 * Memory Controller
 * Ported from tiny-gpu Verilog memory controller
 * Handles global memory access and memory coalescing
 */

import type { Word, Address, MemoryRequest, MemoryResponse, MemoryConfig } from '../types';

export class MemoryController {
  private memory: Word[];
  private config: MemoryConfig;
  private pendingRequests: MemoryRequest[] = [];
  private responseQueue: Map<number, MemoryResponse> = new Map();

  constructor(config: MemoryConfig) {
    this.config = config;
    this.memory = new Array(config.globalMemorySize).fill(0);
  }

  /**
   * Initialize memory with data
   */
  initialize(data: Word[], offset: Address = 0): void {
    for (let i = 0; i < data.length && offset + i < this.memory.length; i++) {
      this.memory[offset + i] = data[i];
    }
  }

  /**
   * Request memory access
   */
  request(req: MemoryRequest): void {
    this.pendingRequests.push(req);
  }

  /**
   * Process memory requests with coalescing
   * Coalesces adjacent memory requests from different threads
   */
  processRequests(): void {
    // Sort requests by address for coalescing
    this.pendingRequests.sort((a, b) => a.address - b.address);

    // Coalesce adjacent requests
    const coalesced: MemoryRequest[] = [];
    let current: MemoryRequest | null = null;

    for (const req of this.pendingRequests) {
      if (current && this.canCoalesce(current, req)) {
        // Coalesce with current request
        if (req.write && req.data !== undefined) {
          this.memory[req.address] = req.data;
        }
        this.responseQueue.set(this.getRequestId(req), {
          data: req.write ? req.data! : this.memory[req.address],
          valid: true,
        });
      } else {
        if (current) {
          coalesced.push(current);
        }
        current = req;
      }
    }

    if (current) {
      coalesced.push(current);
    }

    // Process coalesced requests
    for (const req of coalesced) {
      if (req.write && req.data !== undefined) {
        this.memory[req.address] = req.data;
      }
      this.responseQueue.set(this.getRequestId(req), {
        data: req.write ? req.data! : this.memory[req.address],
        valid: true,
      });
    }

    this.pendingRequests = [];
  }

  /**
   * Check if two requests can be coalesced
   */
  private canCoalesce(a: MemoryRequest, b: MemoryRequest): boolean {
    // Coalesce if addresses are adjacent and both are reads or both are writes
    return (
      Math.abs(a.address - b.address) <= 4 && // Adjacent words (cache line size)
      a.write === b.write
    );
  }

  /**
   * Get response for a request
   */
  getResponse(requestId: number): MemoryResponse | null {
    const response = this.responseQueue.get(requestId);
    if (response) {
      this.responseQueue.delete(requestId);
      return response;
    }
    return null;
  }

  /**
   * Direct memory access (bypass controller, for initialization)
   */
  read(address: Address): Word {
    if (address >= 0 && address < this.memory.length) {
      return this.memory[address];
    }
    return 0;
  }

  /**
   * Direct memory write (bypass controller, for initialization)
   */
  write(address: Address, data: Word): void {
    if (address >= 0 && address < this.memory.length) {
      this.memory[address] = data;
    }
  }

  /**
   * Get memory snapshot
   */
  getMemory(): Word[] {
    return [...this.memory];
  }

  /**
   * Generate unique request ID
   */
  private getRequestId(req: MemoryRequest): number {
    return req.threadId * 1000000 + req.blockId * 10000 + req.address;
  }
}
