/**
 * Type definitions for GPU simulator
 * Ported from tiny-gpu Verilog implementation
 */

export type Word = number; // 32-bit word
export type Address = number; // Memory address
export type RegisterIndex = number; // Register index (0-15)

export interface MemoryRequest {
  address: Address;
  data?: Word;
  write: boolean;
  threadId: number;
  blockId: number;
}

export interface MemoryResponse {
  data: Word;
  valid: boolean;
}

export interface CacheLine {
  tag: Address;
  data: Word[];
  valid: boolean;
  dirty: boolean;
}

export interface MemoryConfig {
  globalMemorySize: number; // Size in words
  cacheSize: number; // Cache size in lines
  lineSize: number; // Words per cache line
  latency: number; // Memory access latency in cycles
}

export interface ThreadState {
  id: number;
  blockId: number;
  pc: Address;
  registers: Word[];
  active: boolean;
}

export interface BlockState {
  id: number;
  threads: ThreadState[];
  sharedMemory?: Word[];
}

export interface Instruction {
  opcode: string;
  rd?: RegisterIndex;
  rs1?: RegisterIndex;
  rs2?: RegisterIndex;
  immediate?: number;
  address?: Address;
}

export interface GPUState {
  memory: Word[];
  cache: CacheLine[];
  threads: ThreadState[];
  blocks: BlockState[];
  cycle: number;
}
