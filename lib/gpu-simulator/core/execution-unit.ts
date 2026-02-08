/**
 * Execution Unit
 * Ported from tiny-gpu Verilog execution unit
 * Handles ALU operations and instruction execution
 */

import type { Word, RegisterIndex, Instruction, ThreadState } from '../types';
import { RegisterFile } from './register-file';

export interface ALUResult {
  result: Word;
  zero: boolean;
  negative: boolean;
  overflow: boolean;
}

export class ExecutionUnit {
  private registerFile: RegisterFile;

  constructor(registerFile: RegisterFile) {
    this.registerFile = registerFile;
  }

  /**
   * Execute ALU operation
   */
  executeALU(
    threadId: number,
    op: string,
    rs1: RegisterIndex,
    rs2: RegisterIndex | number
  ): ALUResult {
    const val1 = this.registerFile.read(threadId, rs1);
    const val2 =
      typeof rs2 === 'number' && rs2 < 16
        ? this.registerFile.read(threadId, rs2)
        : (rs2 as number);

    let result: Word = 0;
    let overflow = false;

    switch (op) {
      case 'ADD':
        result = this.add(val1, val2);
        overflow = this.checkAddOverflow(val1, val2, result);
        break;
      case 'SUB':
        result = this.subtract(val1, val2);
        overflow = this.checkSubOverflow(val1, val2, result);
        break;
      case 'MUL':
        result = this.multiply(val1, val2);
        overflow = this.checkMulOverflow(val1, val2, result);
        break;
      case 'DIV':
        result = this.divide(val1, val2);
        break;
      case 'AND':
        result = val1 & val2;
        break;
      case 'OR':
        result = val1 | val2;
        break;
      case 'XOR':
        result = val1 ^ val2;
        break;
      case 'SHL': // Shift left
        result = val1 << (val2 & 0x1f);
        break;
      case 'SHR': // Shift right (logical)
        result = val1 >>> (val2 & 0x1f);
        break;
      case 'SRA': // Shift right (arithmetic)
        result = val1 >> (val2 & 0x1f);
        break;
      default:
        result = 0;
    }

    return {
      result,
      zero: result === 0,
      negative: (result & 0x80000000) !== 0,
      overflow,
    };
  }

  /**
   * Compare two values
   */
  compare(threadId: number, rs1: RegisterIndex, rs2: RegisterIndex | number): {
    equal: boolean;
    less: boolean;
    greater: boolean;
  } {
    const val1 = this.registerFile.read(threadId, rs1);
    const val2 =
      typeof rs2 === 'number' && rs2 < 16
        ? this.registerFile.read(threadId, rs2)
        : (rs2 as number);

    return {
      equal: val1 === val2,
      less: (val1 | 0) < (val2 | 0), // Signed comparison
      greater: (val1 | 0) > (val2 | 0),
    };
  }

  /**
   * Add with overflow detection
   */
  private add(a: Word, b: Word): Word {
    return (a + b) >>> 0; // Unsigned 32-bit
  }

  /**
   * Subtract
   */
  private subtract(a: Word, b: Word): Word {
    return (a - b) >>> 0;
  }

  /**
   * Multiply (32-bit result, may overflow)
   */
  private multiply(a: Word, b: Word): Word {
    const result = a * b;
    return result >>> 0; // Take lower 32 bits
  }

  /**
   * Divide
   */
  private divide(a: Word, b: Word): Word {
    if (b === 0) return 0;
    return Math.floor((a | 0) / (b | 0)) >>> 0; // Signed division
  }

  /**
   * Check addition overflow
   */
  private checkAddOverflow(a: Word, b: Word, result: Word): boolean {
    const aSigned = a | 0;
    const bSigned = b | 0;
    const resultSigned = result | 0;
    return (aSigned > 0 && bSigned > 0 && resultSigned < 0) ||
           (aSigned < 0 && bSigned < 0 && resultSigned > 0);
  }

  /**
   * Check subtraction overflow
   */
  private checkSubOverflow(a: Word, b: Word, result: Word): boolean {
    const aSigned = a | 0;
    const bSigned = b | 0;
    const resultSigned = result | 0;
    return (aSigned > 0 && bSigned < 0 && resultSigned < 0) ||
           (aSigned < 0 && bSigned > 0 && resultSigned > 0);
  }

  /**
   * Check multiplication overflow
   */
  private checkMulOverflow(a: Word, b: Word, result: Word): boolean {
    // Simple check: if both operands are large, result might overflow
    const aSigned = a | 0;
    const bSigned = b | 0;
    const resultSigned = result | 0;
    const expected = aSigned * bSigned;
    return resultSigned !== expected;
  }
}
