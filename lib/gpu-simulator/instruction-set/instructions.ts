/**
 * Instruction Implementations
 * Ported from tiny-gpu Verilog instruction implementations
 * Each instruction type is implemented here
 */

import type { Instruction, Word, Address, RegisterIndex, ThreadState } from '../types';
import { RegisterFile } from '../core/register-file';
import { ExecutionUnit } from '../core/execution-unit';
import { MemoryController } from '../memory/memory-controller';
import { Cache } from '../memory/cache';

export interface InstructionContext {
  thread: ThreadState;
  registerFile: RegisterFile;
  executionUnit: ExecutionUnit;
  memoryController: MemoryController;
  cache: Cache;
  memory: Word[];
}

export interface InstructionResult {
  nextPC: Address;
  shouldBranch: boolean;
  memoryAccess?: {
    address: Address;
    data?: Word;
    write: boolean;
  };
}

/**
 * Execute an instruction
 */
export function executeInstruction(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile, executionUnit, memoryController, cache, memory } = context;

  switch (instruction.opcode) {
    case 'NOP':
      return { nextPC: thread.pc + 1, shouldBranch: false };

    case 'CONST':
      return executeCONST(instruction, context);

    case 'ADD':
    case 'SUB':
    case 'MUL':
    case 'DIV':
    case 'AND':
    case 'OR':
    case 'XOR':
      return executeALU(instruction, context);

    case 'LDR':
      return executeLDR(instruction, context);

    case 'STR':
      return executeSTR(instruction, context);

    case 'BR':
    case 'BRz':
    case 'BRnz':
    case 'BRn':
    case 'BRp':
      return executeBranch(instruction, context);

    case 'CMP':
      return executeCMP(instruction, context);

    case 'RET':
      return { nextPC: thread.pc, shouldBranch: true }; // Stop execution

    default:
      return { nextPC: thread.pc + 1, shouldBranch: false };
  }
}

/**
 * CONST rd, #imm - Load immediate value
 */
function executeCONST(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile } = context;
  if (instruction.rd !== undefined && instruction.immediate !== undefined) {
    registerFile.write(thread.id, instruction.rd, instruction.immediate);
  }
  return { nextPC: thread.pc + 1, shouldBranch: false };
}

/**
 * ALU operations: ADD, SUB, MUL, DIV, AND, OR, XOR
 */
function executeALU(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile, executionUnit } = context;
  
  if (instruction.rd !== undefined && instruction.rs1 !== undefined && instruction.rs2 !== undefined) {
    const result = executionUnit.executeALU(
      thread.id,
      instruction.opcode,
      instruction.rs1,
      instruction.rs2
    );
    registerFile.write(thread.id, instruction.rd, result.result);
  }
  
  return { nextPC: thread.pc + 1, shouldBranch: false };
}

/**
 * LDR rd, [rs1 + imm] - Load from memory
 */
function executeLDR(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile, cache, memory } = context;
  
  if (instruction.rd !== undefined && instruction.rs1 !== undefined) {
    const baseAddr = registerFile.read(thread.id, instruction.rs1);
    const offset = instruction.immediate || 0;
    const address = (baseAddr + offset) >>> 0;
    
    const { data } = cache.read(address, memory);
    registerFile.write(thread.id, instruction.rd, data);
    
    return {
      nextPC: thread.pc + 1,
      shouldBranch: false,
      memoryAccess: {
        address,
        data,
        write: false,
      },
    };
  }
  
  return { nextPC: thread.pc + 1, shouldBranch: false };
}

/**
 * STR [rs1 + imm], rs2 - Store to memory
 */
function executeSTR(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile, cache, memory } = context;
  
  if (instruction.rs1 !== undefined && instruction.rs2 !== undefined) {
    const baseAddr = registerFile.read(thread.id, instruction.rs1);
    const offset = instruction.immediate || 0;
    const address = (baseAddr + offset) >>> 0;
    const data = registerFile.read(thread.id, instruction.rs2);
    
    cache.write(address, data, memory);
    
    return {
      nextPC: thread.pc + 1,
      shouldBranch: false,
      memoryAccess: {
        address,
        data,
        write: true,
      },
    };
  }
  
  return { nextPC: thread.pc + 1, shouldBranch: false };
}

/**
 * Branch instructions: BR, BRz, BRnz, BRn, BRp
 */
function executeBranch(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, registerFile } = context;
  
  let shouldBranch = false;
  
  if (instruction.opcode === 'BR') {
    shouldBranch = true;
  } else if (instruction.rs1 !== undefined) {
    const value = registerFile.read(thread.id, instruction.rs1);
    const isZero = value === 0;
    const isNegative = (value & 0x80000000) !== 0;
    
    switch (instruction.opcode) {
      case 'BRz':
        shouldBranch = isZero;
        break;
      case 'BRnz':
        shouldBranch = !isZero;
        break;
      case 'BRn':
        shouldBranch = isNegative;
        break;
      case 'BRp':
        shouldBranch = !isNegative && !isZero;
        break;
    }
  }
  
  if (shouldBranch && instruction.immediate !== undefined) {
    return {
      nextPC: thread.pc + instruction.immediate,
      shouldBranch: true,
    };
  }
  
  return { nextPC: thread.pc + 1, shouldBranch: false };
}

/**
 * CMP rs1, rs2 - Compare registers (sets flags)
 */
function executeCMP(
  instruction: Instruction,
  context: InstructionContext
): InstructionResult {
  const { thread, executionUnit } = context;
  
  if (instruction.rs1 !== undefined && instruction.rs2 !== undefined) {
    // Comparison result can be stored in a status register
    // For now, we just perform the comparison
    executionUnit.compare(thread.id, instruction.rs1, instruction.rs2);
  }
  
  return { nextPC: thread.pc + 1, shouldBranch: false };
}
