/**
 * Instruction Decoder
 * Ported from tiny-gpu Verilog instruction decoder
 * Decodes binary instructions into Instruction objects
 */

import type { Instruction, Word } from '../types';

export class InstructionDecoder {
  /**
   * Decode instruction from binary word
   * Instruction format based on tiny-gpu ISA
   */
  decode(instruction: Word): Instruction {
    const opcode = this.getOpcode(instruction);

    switch (opcode) {
      case 'CONST':
        return {
          opcode: 'CONST',
          rd: this.getRd(instruction),
          immediate: this.getImmediate(instruction),
        };
      case 'ADD':
      case 'SUB':
      case 'MUL':
      case 'DIV':
      case 'AND':
      case 'OR':
      case 'XOR':
        return {
          opcode,
          rd: this.getRd(instruction),
          rs1: this.getRs1(instruction),
          rs2: this.getRs2(instruction),
        };
      case 'LDR':
      case 'STR':
        return {
          opcode,
          rd: this.getRd(instruction),
          rs1: this.getRs1(instruction),
          immediate: this.getImmediate(instruction),
        };
      case 'BR':
      case 'BRz':
      case 'BRnz':
      case 'BRn':
      case 'BRp':
        return {
          opcode,
          rs1: this.getRs1(instruction),
          immediate: this.getImmediate(instruction),
        };
      case 'CMP':
        return {
          opcode: 'CMP',
          rs1: this.getRs1(instruction),
          rs2: this.getRs2(instruction),
        };
      case 'RET':
        return {
          opcode: 'RET',
        };
      default:
        return {
          opcode: 'NOP',
        };
    }
  }

  /**
   * Get opcode from instruction (bits 31-24)
   */
  private getOpcode(instruction: Word): string {
    const opcodeNum = (instruction >>> 24) & 0xff;
    return this.opcodeToString(opcodeNum);
  }

  /**
   * Convert opcode number to string
   */
  private opcodeToString(opcode: number): string {
    const opcodes: Record<number, string> = {
      0x00: 'NOP',
      0x01: 'CONST',
      0x02: 'ADD',
      0x03: 'SUB',
      0x04: 'MUL',
      0x05: 'DIV',
      0x06: 'AND',
      0x07: 'OR',
      0x08: 'XOR',
      0x09: 'LDR',
      0x0a: 'STR',
      0x0b: 'BR',
      0x0c: 'BRz',
      0x0d: 'BRnz',
      0x0e: 'BRn',
      0x0f: 'BRp',
      0x10: 'CMP',
      0x11: 'RET',
    };
    return opcodes[opcode] || 'NOP';
  }

  /**
   * Get destination register (bits 23-20)
   */
  private getRd(instruction: Word): number {
    return (instruction >>> 20) & 0xf;
  }

  /**
   * Get source register 1 (bits 19-16)
   */
  private getRs1(instruction: Word): number {
    return (instruction >>> 16) & 0xf;
  }

  /**
   * Get source register 2 (bits 15-12)
   */
  private getRs2(instruction: Word): number {
    return (instruction >>> 12) & 0xf;
  }

  /**
   * Get immediate value (bits 15-0, sign-extended)
   */
  private getImmediate(instruction: Word): number {
    const imm = instruction & 0xffff;
    // Sign extend from 16 bits
    return (imm & 0x8000) ? (imm | 0xffff0000) : imm;
  }

  /**
   * Get address from instruction (for LDR/STR with address)
   */
  getAddress(instruction: Word): number {
    return instruction & 0xffffffff;
  }
}
