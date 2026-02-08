/**
 * Register File
 * Ported from tiny-gpu Verilog register file
 * Manages registers for each thread
 */

import type { Word, RegisterIndex, ThreadState } from '../types';

export class RegisterFile {
  private registers: Map<number, Word[]>; // threadId -> registers
  private readonly numRegisters = 16; // R0-R15

  constructor() {
    this.registers = new Map();
  }

  /**
   * Initialize registers for a thread
   */
  initializeThread(threadId: number): void {
    if (!this.registers.has(threadId)) {
      this.registers.set(threadId, new Array(this.numRegisters).fill(0));
    }
  }

  /**
   * Read register value
   */
  read(threadId: number, reg: RegisterIndex): Word {
    const threadRegs = this.registers.get(threadId);
    if (!threadRegs) {
      this.initializeThread(threadId);
      return 0;
    }
    if (reg >= 0 && reg < this.numRegisters) {
      return threadRegs[reg];
    }
    return 0;
  }

  /**
   * Write register value
   */
  write(threadId: number, reg: RegisterIndex, value: Word): void {
    let threadRegs = this.registers.get(threadId);
    if (!threadRegs) {
      this.initializeThread(threadId);
      threadRegs = this.registers.get(threadId)!;
    }
    if (reg >= 0 && reg < this.numRegisters) {
      threadRegs[reg] = value;
    }
  }

  /**
   * Get all registers for a thread
   */
  getRegisters(threadId: number): Word[] {
    const threadRegs = this.registers.get(threadId);
    if (!threadRegs) {
      this.initializeThread(threadId);
      return this.registers.get(threadId)!;
    }
    return [...threadRegs];
  }

  /**
   * Update thread state with register values
   */
  updateThreadState(thread: ThreadState): void {
    thread.registers = this.getRegisters(thread.id);
  }

  /**
   * Clear registers for a thread
   */
  clearThread(threadId: number): void {
    this.registers.delete(threadId);
  }

  /**
   * Clear all registers
   */
  clearAll(): void {
    this.registers.clear();
  }
}
