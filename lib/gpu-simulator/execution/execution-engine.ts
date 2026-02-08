/**
 * Execution Engine
 * Ported from tiny-gpu Verilog execution engine
 * Main orchestrator for GPU execution
 */

import type { Word, Address, Instruction, ThreadState, BlockState, GPUState, MemoryConfig } from '../types';
import { RegisterFile } from '../core/register-file';
import { ThreadScheduler } from '../core/thread-scheduler';
import { ExecutionUnit } from '../core/execution-unit';
import { MemoryController } from '../memory/memory-controller';
import { Cache } from '../memory/cache';
import { InstructionDecoder } from '../instruction-set/decoder';
import { executeInstruction, type InstructionContext } from '../instruction-set/instructions';

export interface ExecutionConfig {
  memory: MemoryConfig;
  maxCycles?: number;
  enableCache?: boolean;
}

export class ExecutionEngine {
  private registerFile: RegisterFile;
  private threadScheduler: ThreadScheduler;
  private executionUnit: ExecutionUnit;
  private memoryController: MemoryController;
  private cache: Cache;
  private decoder: InstructionDecoder;
  private instructionMemory: Word[] = [];
  private memory: Word[];
  private config: ExecutionConfig;
  private cycle: number = 0;
  private maxCycles: number;

  constructor(config: ExecutionConfig) {
    this.config = config;
    this.maxCycles = config.maxCycles || 1000000;
    
    this.registerFile = new RegisterFile();
    this.threadScheduler = new ThreadScheduler();
    this.executionUnit = new ExecutionUnit(this.registerFile);
    this.memoryController = new MemoryController(config.memory);
    this.cache = config.enableCache !== false ? new Cache(config.memory) : null as any;
    this.decoder = new InstructionDecoder();
    this.memory = new Array(config.memory.globalMemorySize).fill(0);
  }

  /**
   * Load instruction memory
   */
  loadInstructions(instructions: Word[]): void {
    this.instructionMemory = [...instructions];
  }

  /**
   * Initialize blocks and threads
   */
  initializeBlocks(blocks: BlockState[]): void {
    // Initialize register files for all threads
    for (const block of blocks) {
      for (const thread of block.threads) {
        this.registerFile.initializeThread(thread.id);
        thread.registers = this.registerFile.getRegisters(thread.id);
      }
    }

    // Initialize thread scheduler
    this.threadScheduler.initializeBlocks(blocks);
  }

  /**
   * Execute one cycle
   */
  executeCycle(): boolean {
    if (this.cycle >= this.maxCycles) {
      return false; // Max cycles reached
    }

    const warp = this.threadScheduler.getNextWarp();
    if (!warp) {
      return false; // No more warps to execute
    }

    // Execute instructions for all active threads in warp
    const activeThreads = this.threadScheduler.getActiveThreads(warp);
    
    for (const thread of activeThreads) {
      if (thread.pc >= this.instructionMemory.length) {
        thread.active = false;
        continue;
      }

      // Fetch instruction
      const instructionWord = this.instructionMemory[thread.pc];
      const instruction = this.decoder.decode(instructionWord);

      // Create execution context
      const context: InstructionContext = {
        thread,
        registerFile: this.registerFile,
        executionUnit: this.executionUnit,
        memoryController: this.memoryController,
        cache: this.cache,
        memory: this.memory,
      };

      // Execute instruction
      const result = executeInstruction(instruction, context);

      // Update PC
      if (result.shouldBranch) {
        thread.pc = result.nextPC;
      } else {
        thread.pc = result.nextPC;
      }

      // Update thread state
      this.registerFile.updateThreadState(thread);

      // Handle memory access
      if (result.memoryAccess) {
        this.memoryController.request({
          address: result.memoryAccess.address,
          data: result.memoryAccess.data,
          write: result.memoryAccess.write,
          threadId: thread.id,
          blockId: thread.blockId,
        });
      }
    }

    // Process memory requests
    this.memoryController.processRequests();

    // Update cycle
    this.cycle++;
    this.threadScheduler.tick();

    return !this.threadScheduler.isComplete();
  }

  /**
   * Run until completion or max cycles
   */
  run(): GPUState {
    while (this.executeCycle()) {
      // Continue execution
    }

    return this.getState();
  }

  /**
   * Get current GPU state
   */
  getState(): GPUState {
    const blocks: BlockState[] = [];
    const threads: ThreadState[] = [];

    // Collect thread states
    const stats = this.threadScheduler.getStats();
    
    // Note: In a full implementation, we'd reconstruct blocks from scheduler state
    // For now, return the current execution state

    return {
      memory: this.memory,
      cache: this.cache ? [] : [], // Cache state would need to be serialized
      threads,
      blocks,
      cycle: this.cycle,
    };
  }

  /**
   * Get memory snapshot
   */
  getMemory(): Word[] {
    return [...this.memory];
  }

  /**
   * Get execution statistics
   */
  getStats(): {
    cycle: number;
    scheduler: ReturnType<ThreadScheduler['getStats']>;
    cache?: ReturnType<Cache['getStats']>;
  } {
    return {
      cycle: this.cycle,
      scheduler: this.threadScheduler.getStats(),
      cache: this.cache ? this.cache.getStats() : undefined,
    };
  }

  /**
   * Reset execution engine
   */
  reset(): void {
    this.cycle = 0;
    this.threadScheduler.reset();
    this.registerFile.clearAll();
    this.memory.fill(0);
    if (this.cache) {
      this.cache.resetStats();
    }
  }
}
