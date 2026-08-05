#include "lib/LowerContractionToGpu.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include <array>

using namespace mlir;

namespace mlir::tutorial {
namespace {

struct KernelConfig {
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
  int64_t threads;
  int64_t vectorWidth;
  int64_t stages;
};

bool isContiguousF32Matrix(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 2 || !type.getElementType().isF32())
    return false;

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  return strides.back() == 1;
}

bool isContiguousF32Batch(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 3 || !type.getElementType().isF32())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.back() == 1;
}

bool isSupportedMatmul(linalg::MatmulOp matmul) {
  if (matmul->getNumResults() != 0 ||
      !linalg::MatmulOp::isDefaultIndexingMaps(matmul.getIndexingMapsAttr()))
    return false;
  auto inputs = matmul.getDpsInputs();
  auto outputs = matmul.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         isContiguousF32Matrix(inputs[0]) &&
         isContiguousF32Matrix(inputs[1]) &&
         isContiguousF32Matrix(outputs[0]);
}

bool isSupportedBatchMatmul(linalg::BatchMatmulOp batchMatmul) {
  if (batchMatmul->getNumResults() != 0 ||
      !linalg::BatchMatmulOp::isDefaultIndexingMaps(
          batchMatmul.getIndexingMapsAttr()))
    return false;
  auto inputs = batchMatmul.getDpsInputs();
  auto outputs = batchMatmul.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         isContiguousF32Batch(inputs[0]) &&
         isContiguousF32Batch(inputs[1]) &&
         isContiguousF32Batch(outputs[0]);
}

FailureOr<std::array<int64_t, 2>> getPositivePair(DenseIntElementsAttr attr) {
  if (!attr || attr.getNumElements() != 2)
    return failure();
  std::array<int64_t, 2> values;
  size_t index = 0;
  for (APInt value : attr.getValues<APInt>())
    values[index++] = value.getSExtValue();
  if (values[0] <= 0 || values[1] <= 0)
    return failure();
  return values;
}

bool isContiguousF32Rank4(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 4 || !type.getElementType().isF32())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  return succeeded(type.getStridesAndOffset(strides, offset)) &&
         strides.back() == 1;
}

bool isSupportedConvolution(linalg::Conv2DNchwFchwOp conv) {
  if (conv->getNumResults() != 0 ||
      failed(getPositivePair(conv.getStrides())) ||
      failed(getPositivePair(conv.getDilations())))
    return false;
  auto inputs = conv.getDpsInputs();
  auto outputs = conv.getDpsInits();
  return inputs.size() == 2 && outputs.size() == 1 &&
         isContiguousF32Rank4(inputs[0]) &&
         isContiguousF32Rank4(inputs[1]) &&
         isContiguousF32Rank4(outputs[0]);
}

SmallVector<Value> matrixIndices(Value batch, Value row, Value column) {
  SmallVector<Value> indices;
  if (batch)
    indices.push_back(batch);
  indices.append({row, column});
  return indices;
}

Value indexConstant(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

Value ceilDiv(OpBuilder &builder, Location loc, Value value, int64_t divisor) {
  Value adjustment = indexConstant(builder, loc, divisor - 1);
  Value divisorValue = indexConstant(builder, loc, divisor);
  Value adjusted = arith::AddIOp::create(builder, loc, value, adjustment);
  return arith::DivUIOp::create(builder, loc, adjusted, divisorValue);
}

Value andValues(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return arith::AndIOp::create(builder, loc, lhs, rhs);
}

Value inBounds2D(OpBuilder &builder, Location loc, Value row, Value rows,
                 Value column, Value columns) {
  Value rowOk = arith::CmpIOp::create(builder, loc,
                                      arith::CmpIPredicate::ult, row, rows);
  Value columnOk = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, column, columns);
  return andValues(builder, loc, rowOk, columnOk);
}

Value guardedLoad(OpBuilder &builder, Location loc, Value source,
                  ValueRange indices, Value condition, Value zero) {
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{zero.getType()},
                                condition, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value loaded = memref::LoadOp::create(builder, loc, source, indices);
  scf::YieldOp::create(builder, loc, loaded);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, zero);
  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

void guardedStore(OpBuilder &builder, Location loc, Value value, Value target,
                  ValueRange indices, Value condition) {
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange(), condition,
                                /*withElseRegion=*/false);
  ifOp.getThenRegion().front().getTerminator()->erase();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  memref::StoreOp::create(builder, loc, value, target, indices);
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(ifOp);
}

void emitVectorizedTileLoad(OpBuilder &builder, Location loc, Value source,
                            Value shared, Value batch, Value globalRow,
                            Value globalColumn, Value localRow,
                            Value localColumn, Value rowLimit,
                            Value columnLimit, Value zeroFloat,
                            int64_t vectorWidth) {
  Value lastLane = indexConstant(builder, loc, vectorWidth - 1);
  Value vectorEnd =
      arith::AddIOp::create(builder, loc, globalColumn, lastLane);
  Value fullVector = inBounds2D(builder, loc, globalRow, rowLimit, vectorEnd,
                                columnLimit);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange(), fullVector,
                                /*withElseRegion=*/true);
  ifOp.getThenRegion().front().getTerminator()->erase();
  ifOp.getElseRegion().front().getTerminator()->erase();

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto vectorType = VectorType::get({vectorWidth}, builder.getF32Type());
  Value loaded = vector::LoadOp::create(
      builder, loc, vectorType, source,
      matrixIndices(batch, globalRow, globalColumn));
  vector::StoreOp::create(builder, loc, loaded, shared,
                          ValueRange{localRow, localColumn});
  scf::YieldOp::create(builder, loc);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  for (int64_t lane = 0; lane < vectorWidth; ++lane) {
    Value laneValue = indexConstant(builder, loc, lane);
    Value sourceColumn =
        arith::AddIOp::create(builder, loc, globalColumn, laneValue);
    Value destinationColumn =
        arith::AddIOp::create(builder, loc, localColumn, laneValue);
    Value condition = inBounds2D(builder, loc, globalRow, rowLimit,
                                 sourceColumn, columnLimit);
    Value scalar = guardedLoad(
        builder, loc, source, matrixIndices(batch, globalRow, sourceColumn),
        condition, zeroFloat);
    memref::StoreOp::create(builder, loc, scalar, shared,
                            ValueRange{localRow, destinationColumn});
  }
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(ifOp);
}

void emitCooperativeTileLoads(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs, Value sharedA, Value sharedB,
                              Value batch, Value blockRow, Value blockColumn,
                              Value kBase, Value threadId, Value mSize,
                              Value nSize, Value kSize, Value zeroFloat,
                              const KernelConfig &config) {
  Value blockKValue = indexConstant(builder, loc, config.blockK);
  Value blockNValue = indexConstant(builder, loc, config.blockN);
  Value vectorWidthValue = indexConstant(builder, loc, config.vectorWidth);

  int64_t aVectors =
      config.blockM * config.blockK / config.vectorWidth;
  int64_t bVectors =
      config.blockK * config.blockN / config.vectorWidth;
  int64_t aCopiesPerThread = aVectors / config.threads;
  int64_t bCopiesPerThread = bVectors / config.threads;

  for (int64_t copy = 0; copy < aCopiesPerThread; ++copy) {
    Value vectorIndex = arith::AddIOp::create(
        builder, loc, threadId,
        indexConstant(builder, loc, copy * config.threads));
    Value linear =
        arith::MulIOp::create(builder, loc, vectorIndex, vectorWidthValue);
    Value localRow =
        arith::DivUIOp::create(builder, loc, linear, blockKValue);
    Value localK = arith::RemUIOp::create(builder, loc, linear, blockKValue);
    Value globalRow =
        arith::AddIOp::create(builder, loc, blockRow, localRow);
    Value globalK = arith::AddIOp::create(builder, loc, kBase, localK);
    emitVectorizedTileLoad(builder, loc, lhs, sharedA, batch, globalRow,
                           globalK, localRow, localK, mSize, kSize, zeroFloat,
                           config.vectorWidth);
  }

  for (int64_t copy = 0; copy < bCopiesPerThread; ++copy) {
    Value vectorIndex = arith::AddIOp::create(
        builder, loc, threadId,
        indexConstant(builder, loc, copy * config.threads));
    Value linear =
        arith::MulIOp::create(builder, loc, vectorIndex, vectorWidthValue);
    Value localK =
        arith::DivUIOp::create(builder, loc, linear, blockNValue);
    Value localColumn =
        arith::RemUIOp::create(builder, loc, linear, blockNValue);
    Value globalK = arith::AddIOp::create(builder, loc, kBase, localK);
    Value globalColumn =
        arith::AddIOp::create(builder, loc, blockColumn, localColumn);
    emitVectorizedTileLoad(builder, loc, rhs, sharedB, batch, globalK,
                           globalColumn, localK, localColumn, kSize, nSize,
                           zeroFloat,
                           config.vectorWidth);
  }
}

Value availableVectorElements(OpBuilder &builder, Location loc, Value row,
                              Value rowLimit, Value column,
                              Value columnLimit, int64_t vectorWidth) {
  Value rowOk = arith::CmpIOp::create(builder, loc,
                                      arith::CmpIPredicate::ult, row, rowLimit);
  Value columnOk = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, column, columnLimit);
  Value anyAvailable = andValues(builder, loc, rowOk, columnOk);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{builder.getIndexType()},
                                anyAvailable, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value remaining =
      arith::SubIOp::create(builder, loc, columnLimit, column);
  Value available = arith::MinUIOp::create(
      builder, loc, remaining, indexConstant(builder, loc, vectorWidth));
  scf::YieldOp::create(builder, loc, available);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, indexConstant(builder, loc, 0));
  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

Value emitAsyncVectorCopy(OpBuilder &builder, Location loc, Value source,
                          Value shared, Value batch, Value stage,
                          Value globalRow, Value globalColumn, Value localRow,
                          Value localColumn, Value rowLimit, Value columnLimit,
                          int64_t vectorWidth) {
  Value rowOk = arith::CmpIOp::create(builder, loc,
                                      arith::CmpIPredicate::ult, globalRow,
                                      rowLimit);
  Value columnOk = arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::ult,
                                         globalColumn, columnLimit);
  Value safeRow = arith::SelectOp::create(
      builder, loc, rowOk, globalRow, indexConstant(builder, loc, 0));
  Value safeColumn = arith::SelectOp::create(
      builder, loc, columnOk, globalColumn, indexConstant(builder, loc, 0));
  Value sourceElements = availableVectorElements(
      builder, loc, globalRow, rowLimit, globalColumn, columnLimit,
      vectorWidth);
  auto tokenType = nvgpu::DeviceAsyncTokenType::get(builder.getContext());
  return nvgpu::DeviceAsyncCopyOp::create(
      builder, loc, tokenType, shared,
      ValueRange{stage, localRow, localColumn}, source,
      matrixIndices(batch, safeRow, safeColumn),
      builder.getIndexAttr(vectorWidth),
      sourceElements,
      vectorWidth == 4 ? builder.getUnitAttr() : UnitAttr());
}

Value emitAsyncTileLoads(OpBuilder &builder, Location loc, Value lhs,
                         Value rhs, Value sharedA, Value sharedB,
                         Value batch, Value stage, Value blockRow,
                         Value blockColumn, Value kBase, Value threadId,
                         Value mSize, Value nSize, Value kSize,
                         const KernelConfig &config) {
  Value blockKValue = indexConstant(builder, loc, config.blockK);
  Value blockNValue = indexConstant(builder, loc, config.blockN);
  Value vectorWidthValue = indexConstant(builder, loc, config.vectorWidth);
  int64_t aVectors =
      config.blockM * config.blockK / config.vectorWidth;
  int64_t bVectors =
      config.blockK * config.blockN / config.vectorWidth;
  int64_t aCopiesPerThread = aVectors / config.threads;
  int64_t bCopiesPerThread = bVectors / config.threads;
  SmallVector<Value> tokens;
  tokens.reserve(aCopiesPerThread + bCopiesPerThread);

  for (int64_t copy = 0; copy < aCopiesPerThread; ++copy) {
    Value vectorIndex = arith::AddIOp::create(
        builder, loc, threadId,
        indexConstant(builder, loc, copy * config.threads));
    Value linear =
        arith::MulIOp::create(builder, loc, vectorIndex, vectorWidthValue);
    Value localRow =
        arith::DivUIOp::create(builder, loc, linear, blockKValue);
    Value localK = arith::RemUIOp::create(builder, loc, linear, blockKValue);
    Value globalRow =
        arith::AddIOp::create(builder, loc, blockRow, localRow);
    Value globalK = arith::AddIOp::create(builder, loc, kBase, localK);
    tokens.push_back(emitAsyncVectorCopy(
        builder, loc, lhs, sharedA, batch, stage, globalRow, globalK, localRow,
        localK, mSize, kSize, config.vectorWidth));
  }

  for (int64_t copy = 0; copy < bCopiesPerThread; ++copy) {
    Value vectorIndex = arith::AddIOp::create(
        builder, loc, threadId,
        indexConstant(builder, loc, copy * config.threads));
    Value linear =
        arith::MulIOp::create(builder, loc, vectorIndex, vectorWidthValue);
    Value localK =
        arith::DivUIOp::create(builder, loc, linear, blockNValue);
    Value localColumn =
        arith::RemUIOp::create(builder, loc, linear, blockNValue);
    Value globalK = arith::AddIOp::create(builder, loc, kBase, localK);
    Value globalColumn =
        arith::AddIOp::create(builder, loc, blockColumn, localColumn);
    tokens.push_back(emitAsyncVectorCopy(
        builder, loc, rhs, sharedB, batch, stage, globalK, globalColumn, localK,
        localColumn, kSize, nSize, config.vectorWidth));
  }

  return nvgpu::DeviceAsyncCreateGroupOp::create(
      builder, loc, nvgpu::DeviceAsyncTokenType::get(builder.getContext()),
      tokens);
}

void emitConvolutionTileLoads(
    OpBuilder &builder, Location loc, Value input, Value filter, Value sharedA,
    Value sharedB, Value batch, Value blockFilter, Value blockPosition,
    Value kBase, Value threadId, Value fSize, Value cSize, Value ihSize,
    Value iwSize, Value ohSize, Value owSize, Value khSize, Value kwSize,
    Value reductionSize, Value strideH, Value strideW, Value dilationH,
    Value dilationW, Value zeroFloat, const KernelConfig &config) {
  Value threadsValue = indexConstant(builder, loc, config.threads);
  Value totalA = indexConstant(builder, loc, config.blockM * config.blockK);
  Value totalB = indexConstant(builder, loc, config.blockK * config.blockN);
  Value blockKValue = indexConstant(builder, loc, config.blockK);
  Value blockNValue = indexConstant(builder, loc, config.blockN);

  scf::ForOp::create(
      builder, loc, threadId, totalA, threadsValue, ValueRange{},
      [&](OpBuilder &loadBuilder, Location loadLoc, Value linear, ValueRange) {
        Value localFilter = arith::DivUIOp::create(
            loadBuilder, loadLoc, linear, blockKValue);
        Value localK = arith::RemUIOp::create(loadBuilder, loadLoc, linear,
                                              blockKValue);
        Value globalFilter = arith::AddIOp::create(
            loadBuilder, loadLoc, blockFilter, localFilter);
        Value globalK =
            arith::AddIOp::create(loadBuilder, loadLoc, kBase, localK);
        Value kw = arith::RemUIOp::create(loadBuilder, loadLoc, globalK,
                                          kwSize);
        Value quotient = arith::DivUIOp::create(loadBuilder, loadLoc, globalK,
                                                kwSize);
        Value kh = arith::RemUIOp::create(loadBuilder, loadLoc, quotient,
                                          khSize);
        Value channel = arith::DivUIOp::create(loadBuilder, loadLoc, quotient,
                                               khSize);
        Value condition = inBounds2D(loadBuilder, loadLoc, globalFilter, fSize,
                                     globalK, reductionSize);
        Value loaded = guardedLoad(
            loadBuilder, loadLoc, filter,
            ValueRange{globalFilter, channel, kh, kw}, condition, zeroFloat);
        memref::StoreOp::create(loadBuilder, loadLoc, loaded, sharedA,
                                ValueRange{localFilter, localK});
        scf::YieldOp::create(loadBuilder, loadLoc);
      });

  Value positionSize =
      arith::MulIOp::create(builder, loc, ohSize, owSize);
  scf::ForOp::create(
      builder, loc, threadId, totalB, threadsValue, ValueRange{},
      [&](OpBuilder &loadBuilder, Location loadLoc, Value linear, ValueRange) {
        Value localK = arith::DivUIOp::create(loadBuilder, loadLoc, linear,
                                              blockNValue);
        Value localPosition = arith::RemUIOp::create(
            loadBuilder, loadLoc, linear, blockNValue);
        Value globalK =
            arith::AddIOp::create(loadBuilder, loadLoc, kBase, localK);
        Value globalPosition = arith::AddIOp::create(
            loadBuilder, loadLoc, blockPosition, localPosition);
        Value kw = arith::RemUIOp::create(loadBuilder, loadLoc, globalK,
                                          kwSize);
        Value quotient = arith::DivUIOp::create(loadBuilder, loadLoc, globalK,
                                                kwSize);
        Value kh = arith::RemUIOp::create(loadBuilder, loadLoc, quotient,
                                          khSize);
        Value channel = arith::DivUIOp::create(loadBuilder, loadLoc, quotient,
                                               khSize);
        Value ow = arith::RemUIOp::create(loadBuilder, loadLoc, globalPosition,
                                          owSize);
        Value oh = arith::DivUIOp::create(loadBuilder, loadLoc, globalPosition,
                                          owSize);
        Value inputH = arith::MulIOp::create(loadBuilder, loadLoc, oh, strideH);
        inputH = arith::AddIOp::create(
            loadBuilder, loadLoc, inputH,
            arith::MulIOp::create(loadBuilder, loadLoc, kh, dilationH));
        Value inputW = arith::MulIOp::create(loadBuilder, loadLoc, ow, strideW);
        inputW = arith::AddIOp::create(
            loadBuilder, loadLoc, inputW,
            arith::MulIOp::create(loadBuilder, loadLoc, kw, dilationW));
        Value kOk = arith::CmpIOp::create(loadBuilder, loadLoc,
                                          arith::CmpIPredicate::ult, globalK,
                                          reductionSize);
        Value positionOk = arith::CmpIOp::create(
            loadBuilder, loadLoc, arith::CmpIPredicate::ult, globalPosition,
            positionSize);
        Value hOk = arith::CmpIOp::create(loadBuilder, loadLoc,
                                          arith::CmpIPredicate::ult, inputH,
                                          ihSize);
        Value wOk = arith::CmpIOp::create(loadBuilder, loadLoc,
                                          arith::CmpIPredicate::ult, inputW,
                                          iwSize);
        Value condition = andValues(loadBuilder, loadLoc, kOk, positionOk);
        condition = andValues(loadBuilder, loadLoc, condition, hOk);
        condition = andValues(loadBuilder, loadLoc, condition, wOk);
        Value loaded = guardedLoad(
            loadBuilder, loadLoc, input,
            ValueRange{batch, channel, inputH, inputW}, condition, zeroFloat);
        memref::StoreOp::create(loadBuilder, loadLoc, loaded, sharedB,
                                ValueRange{localK, localPosition});
        scf::YieldOp::create(loadBuilder, loadLoc);
      });
}

Value zeroVector(OpBuilder &builder, Location loc, VectorType type) {
  auto zero = builder.getF32FloatAttr(0.0);
  return arith::ConstantOp::create(builder, loc, type,
                                   DenseElementsAttr::get(type, zero));
}

Value insertVectorElement(OpBuilder &builder, Location loc, Value scalar,
                          Value vector, ArrayRef<int64_t> position) {
  return vector::InsertOp::create(builder, loc, scalar, vector, position);
}

Value initializeTensorCoreAccumulator(OpBuilder &builder, Location loc,
                                      Value output, Value batch, Value row0,
                                      Value row1, Value column, Value mSize,
                                      Value nSize, Value zeroFloat) {
  auto accumulatorType = VectorType::get({2, 2}, builder.getF32Type());
  Value accumulator = zeroVector(builder, loc, accumulatorType);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t columnLane = 0; columnLane < 2; ++columnLane) {
      Value globalRow = row == 0 ? row0 : row1;
      Value globalColumn = arith::AddIOp::create(
          builder, loc, column, indexConstant(builder, loc, columnLane));
      Value condition = inBounds2D(builder, loc, globalRow, mSize,
                                   globalColumn, nSize);
      Value initial = guardedLoad(
          builder, loc, output,
          matrixIndices(batch, globalRow, globalColumn), condition, zeroFloat);
      accumulator = insertVectorElement(builder, loc, initial, accumulator,
                                        {row, columnLane});
    }
  }
  return accumulator;
}

Value emitTensorCoreMma(OpBuilder &builder, Location loc, Value sharedA,
                        Value sharedB, Value localM, Value localN,
                        Value localK, Value laneId, Value stage,
                        Value accumulator) {
  Value sixteen = indexConstant(builder, loc, 16);
  Value four = indexConstant(builder, loc, 4);
  Value laneMod16 = arith::RemUIOp::create(builder, loc, laneId, sixteen);
  Value laneHalf = arith::DivUIOp::create(builder, loc, laneId, sixteen);
  Value aRow = arith::AddIOp::create(builder, loc, localM, laneMod16);
  Value aColumnOffset =
      arith::MulIOp::create(builder, loc, laneHalf, four);
  Value aColumn =
      arith::AddIOp::create(builder, loc, localK, aColumnOffset);
  auto aFragmentType = VectorType::get({4, 1}, builder.getF32Type());
  SmallVector<Value> aIndices;
  if (stage)
    aIndices.push_back(stage);
  aIndices.append({aRow, aColumn});
  Value aFragment = nvgpu::LdMatrixOp::create(
      builder, loc, aFragmentType, sharedA, aIndices,
      /*transpose=*/false, /*numTiles=*/4);

  Value laneMod4 = arith::RemUIOp::create(builder, loc, laneId, four);
  Value laneDiv4 = arith::DivUIOp::create(builder, loc, laneId, four);
  Value bColumn = arith::AddIOp::create(builder, loc, localN, laneDiv4);
  auto bFragmentType = VectorType::get({2, 1}, builder.getF32Type());
  Value bFragment = zeroVector(builder, loc, bFragmentType);
  for (int64_t part = 0; part < 2; ++part) {
    Value bRowOffset = arith::AddIOp::create(
        builder, loc, laneMod4, indexConstant(builder, loc, part * 4));
    Value bRow = arith::AddIOp::create(builder, loc, localK, bRowOffset);
    SmallVector<Value> bIndices;
    if (stage)
      bIndices.push_back(stage);
    bIndices.append({bRow, bColumn});
    Value loaded =
        memref::LoadOp::create(builder, loc, sharedB, bIndices);
    bFragment =
        insertVectorElement(builder, loc, loaded, bFragment, {part, 0});
  }

  return nvgpu::MmaSyncOp::create(builder, loc, aFragment, bFragment,
                                  accumulator,
                                  ArrayRef<int64_t>{16, 8, 8},
                                  /*tf32Enabled=*/true);
}

void emitTensorCoreContractionKernel(IRRewriter &rewriter, Operation *sourceOp,
                                     Value lhs, Value rhs, Value output,
                                     bool batched,
                                     const KernelConfig &config) {
  Location loc = sourceOp->getLoc();
  Value zero = indexConstant(rewriter, loc, 0);
  Value one = indexConstant(rewriter, loc, 1);
  Value zeroFloat = arith::ConstantFloatOp::create(
      rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
  int64_t matrixOffset = batched ? 1 : 0;
  Value batchSize = batched
                        ? memref::DimOp::create(rewriter, loc, output, 0)
                        : one;
  Value mSize =
      memref::DimOp::create(rewriter, loc, output, matrixOffset);
  Value nSize =
      memref::DimOp::create(rewriter, loc, output, matrixOffset + 1);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, matrixOffset + 1);
  Value gridX = ceilDiv(rewriter, loc, nSize, config.blockN);
  Value gridY = ceilDiv(rewriter, loc, mSize, config.blockM);
  Value blockSize = indexConstant(rewriter, loc, config.threads);

  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::AddressSpace::Workgroup);
  SmallVector<int64_t> sharedAShape = {config.blockM, config.blockK};
  SmallVector<int64_t> sharedBShape = {config.blockK, config.blockN};
  if (config.stages == 2) {
    sharedAShape.insert(sharedAShape.begin(), 2);
    sharedBShape.insert(sharedBShape.begin(), 2);
  }
  auto sharedAType = MemRefType::get(
      sharedAShape, rewriter.getF32Type(),
      MemRefLayoutAttrInterface{}, workgroupSpace);
  auto sharedBType = MemRefType::get(
      sharedBShape, rewriter.getF32Type(),
      MemRefLayoutAttrInterface{}, workgroupSpace);
  auto launch = gpu::LaunchOp::create(
      rewriter, loc, gridX, gridY, batchSize, blockSize, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/ValueRange{},
      SmallVector<Type>{sharedAType, sharedBType});

  Block &body = launch.getBody().front();
  rewriter.setInsertionPointToStart(&body);
  Value blockX = launch.getBlockIds().x;
  Value blockY = launch.getBlockIds().y;
  Value batch = batched ? launch.getBlockIds().z : Value();
  Value threadId = launch.getThreadIds().x;
  Value sharedA = body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes);
  Value sharedB =
      body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes + 1);
  Value blockRow = arith::MulIOp::create(
      rewriter, loc, blockY, indexConstant(rewriter, loc, config.blockM));
  Value blockColumn = arith::MulIOp::create(
      rewriter, loc, blockX, indexConstant(rewriter, loc, config.blockN));
  Value laneId = arith::RemUIOp::create(
      rewriter, loc, threadId, indexConstant(rewriter, loc, 32));
  Value warpId = arith::DivUIOp::create(
      rewriter, loc, threadId, indexConstant(rewriter, loc, 32));
  int64_t warps = config.threads / 32;
  int64_t tilesM = config.blockM / 16;
  int64_t tilesN = config.blockN / 8;
  int64_t tilesPerWarp = tilesM * tilesN / warps;

  struct WarpTile {
    Value localM;
    Value localN;
    Value row0;
    Value row1;
    Value column;
  };
  SmallVector<WarpTile> warpTiles;
  SmallVector<Value> accumulators;
  for (int64_t assignment = 0; assignment < tilesPerWarp; ++assignment) {
    Value tile = arith::AddIOp::create(
        rewriter, loc, warpId,
        indexConstant(rewriter, loc, assignment * warps));
    Value tileM = arith::DivUIOp::create(
        rewriter, loc, tile, indexConstant(rewriter, loc, tilesN));
    Value tileN = arith::RemUIOp::create(
        rewriter, loc, tile, indexConstant(rewriter, loc, tilesN));
    Value localM = arith::MulIOp::create(
        rewriter, loc, tileM, indexConstant(rewriter, loc, 16));
    Value localN = arith::MulIOp::create(
        rewriter, loc, tileN, indexConstant(rewriter, loc, 8));
    Value laneRow = arith::DivUIOp::create(
        rewriter, loc, laneId, indexConstant(rewriter, loc, 4));
    Value laneColumn = arith::RemUIOp::create(
        rewriter, loc, laneId, indexConstant(rewriter, loc, 4));
    laneColumn = arith::MulIOp::create(
        rewriter, loc, laneColumn, indexConstant(rewriter, loc, 2));
    Value row0 = arith::AddIOp::create(rewriter, loc, blockRow, localM);
    row0 = arith::AddIOp::create(rewriter, loc, row0, laneRow);
    Value row1 = arith::AddIOp::create(
        rewriter, loc, row0, indexConstant(rewriter, loc, 8));
    Value column =
        arith::AddIOp::create(rewriter, loc, blockColumn, localN);
    column = arith::AddIOp::create(rewriter, loc, column, laneColumn);
    warpTiles.push_back({localM, localN, row0, row1, column});
    accumulators.push_back(initializeTensorCoreAccumulator(
        rewriter, loc, output, batch, row0, row1, column, mSize, nSize,
        zeroFloat));
  }

  Value blockKValue = indexConstant(rewriter, loc, config.blockK);
  if (config.stages == 2) {
    Value initialGroup = emitAsyncTileLoads(
        rewriter, loc, lhs, rhs, sharedA, sharedB, batch, zero, blockRow,
        blockColumn, zero, threadId, mSize, nSize, kSize, config);
    nvgpu::DeviceAsyncWaitOp::create(rewriter, loc, initialGroup, nullptr);
    gpu::BarrierOp::create(rewriter, loc, gpu::AddressSpace::Workgroup);
  }
  auto kTiles = scf::ForOp::create(
      rewriter, loc, zero, kSize, blockKValue, accumulators,
      [&](OpBuilder &tileBuilder, Location tileLoc, Value kBase,
          ValueRange tileAccumulators) {
        Value currentStage;
        Value nextGroup;
        if (config.stages == 2) {
          Value tileNumber = arith::DivUIOp::create(
              tileBuilder, tileLoc, kBase, blockKValue);
          currentStage = arith::RemUIOp::create(
              tileBuilder, tileLoc, tileNumber,
              indexConstant(tileBuilder, tileLoc, 2));
          Value nextStage = arith::SubIOp::create(
              tileBuilder, tileLoc, one, currentStage);
          Value nextK = arith::AddIOp::create(tileBuilder, tileLoc, kBase,
                                              blockKValue);
          nextGroup = emitAsyncTileLoads(
              tileBuilder, tileLoc, lhs, rhs, sharedA, sharedB, batch,
              nextStage, blockRow, blockColumn, nextK, threadId, mSize, nSize,
              kSize, config);
        } else {
          emitCooperativeTileLoads(
              tileBuilder, tileLoc, lhs, rhs, sharedA, sharedB, batch,
              blockRow, blockColumn, kBase, threadId, mSize, nSize, kSize,
              zeroFloat, config);
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
        }
        SmallVector<Value> next(tileAccumulators.begin(),
                                tileAccumulators.end());
        for (int64_t localK = 0; localK < config.blockK; localK += 8) {
          Value localKValue = indexConstant(tileBuilder, tileLoc, localK);
          for (size_t tile = 0; tile < warpTiles.size(); ++tile)
            next[tile] = emitTensorCoreMma(
                tileBuilder, tileLoc, sharedA, sharedB,
                warpTiles[tile].localM, warpTiles[tile].localN, localKValue,
                laneId, currentStage, next[tile]);
        }
        if (config.stages == 2)
          nvgpu::DeviceAsyncWaitOp::create(tileBuilder, tileLoc, nextGroup,
                                           nullptr);
        gpu::BarrierOp::create(tileBuilder, tileLoc,
                               gpu::AddressSpace::Workgroup);
        scf::YieldOp::create(tileBuilder, tileLoc, next);
      });

  for (size_t tile = 0; tile < warpTiles.size(); ++tile) {
    for (int64_t row = 0; row < 2; ++row) {
      for (int64_t columnLane = 0; columnLane < 2; ++columnLane) {
        Value scalar = vector::ExtractOp::create(
            rewriter, loc, kTiles.getResult(tile),
            ArrayRef<int64_t>{row, columnLane});
        Value globalRow = row == 0 ? warpTiles[tile].row0
                                   : warpTiles[tile].row1;
        Value globalColumn = arith::AddIOp::create(
            rewriter, loc, warpTiles[tile].column,
            indexConstant(rewriter, loc, columnLane));
        Value condition = inBounds2D(rewriter, loc, globalRow, mSize,
                                     globalColumn, nSize);
        guardedStore(rewriter, loc, scalar, output,
                     matrixIndices(batch, globalRow, globalColumn), condition);
      }
    }
  }
  gpu::TerminatorOp::create(rewriter, loc);
  rewriter.setInsertionPointAfter(launch);
}

void emitSharedContractionKernel(IRRewriter &rewriter, Operation *sourceOp,
                                 Value lhs, Value rhs, Value output,
                                 bool batched, const KernelConfig &config) {
  Location loc = sourceOp->getLoc();

  Value one = indexConstant(rewriter, loc, 1);
  Value zero = indexConstant(rewriter, loc, 0);
  Value zeroFloat = arith::ConstantFloatOp::create(
      rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
  int64_t matrixOffset = batched ? 1 : 0;
  Value batchSize = batched
                        ? memref::DimOp::create(rewriter, loc, output, 0)
                        : one;
  Value mSize =
      memref::DimOp::create(rewriter, loc, output, matrixOffset);
  Value nSize =
      memref::DimOp::create(rewriter, loc, output, matrixOffset + 1);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, matrixOffset + 1);
  Value gridX = ceilDiv(rewriter, loc, nSize, config.blockN);
  Value gridY = ceilDiv(rewriter, loc, mSize, config.blockM);
  Value blockSize = indexConstant(rewriter, loc, config.threads);

  auto workgroupSpace =
      gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                 gpu::AddressSpace::Workgroup);
  SmallVector<int64_t> sharedAShape;
  SmallVector<int64_t> sharedBShape;
  if (config.stages == 2) {
    sharedAShape = {2, config.blockM, config.blockK};
    sharedBShape = {2, config.blockK, config.blockN};
  } else {
    sharedAShape = {config.blockM, config.blockK};
    sharedBShape = {config.blockK, config.blockN};
  }
  auto sharedAType = MemRefType::get(sharedAShape, rewriter.getF32Type(),
                                     MemRefLayoutAttrInterface{},
                                     workgroupSpace);
  auto sharedBType = MemRefType::get(sharedBShape, rewriter.getF32Type(),
                                     MemRefLayoutAttrInterface{},
                                     workgroupSpace);
  SmallVector<Type> workgroupTypes{sharedAType, sharedBType};
  auto launch = gpu::LaunchOp::create(
      rewriter, loc, gridX, gridY, batchSize, blockSize, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/ValueRange{}, workgroupTypes);

  Block &body = launch.getBody().front();
  rewriter.setInsertionPointToStart(&body);
  Value blockX = launch.getBlockIds().x;
  Value blockY = launch.getBlockIds().y;
  Value batch = batched ? launch.getBlockIds().z : Value();
  Value threadId = launch.getThreadIds().x;
  Value sharedA = body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes);
  Value sharedB =
      body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes + 1);
  Value blockMValue = indexConstant(rewriter, loc, config.blockM);
  Value blockNValue = indexConstant(rewriter, loc, config.blockN);
  Value blockKValue = indexConstant(rewriter, loc, config.blockK);
  Value warpWidth = indexConstant(rewriter, loc, 32);
  int64_t threadRows = config.threads / 32;
  int64_t microRows = config.blockM / threadRows;
  int64_t microColumns = config.blockN / 32;
  Value threadRowsValue = indexConstant(rewriter, loc, threadRows);
  Value blockRow = arith::MulIOp::create(rewriter, loc, blockY, blockMValue);
  Value blockColumn =
      arith::MulIOp::create(rewriter, loc, blockX, blockNValue);
  Value threadRow =
      arith::DivUIOp::create(rewriter, loc, threadId, warpWidth);
  Value threadColumn =
      arith::RemUIOp::create(rewriter, loc, threadId, warpWidth);

  SmallVector<Value> rows;
  SmallVector<Value> columns;
  for (int64_t row = 0; row < microRows; ++row) {
    Value rowOffset = arith::AddIOp::create(
        rewriter, loc, threadRow,
        indexConstant(rewriter, loc, row * threadRows));
    rows.push_back(
        arith::AddIOp::create(rewriter, loc, blockRow, rowOffset));
  }
  for (int64_t column = 0; column < microColumns; ++column) {
    Value columnOffset = arith::AddIOp::create(
        rewriter, loc, threadColumn,
        indexConstant(rewriter, loc, column * 32));
    columns.push_back(
        arith::AddIOp::create(rewriter, loc, blockColumn, columnOffset));
  }

  SmallVector<Value> accumulators;
  for (Value row : rows) {
    for (Value column : columns) {
      Value condition =
          inBounds2D(rewriter, loc, row, mSize, column, nSize);
      accumulators.push_back(guardedLoad(
          rewriter, loc, output, matrixIndices(batch, row, column), condition,
          zeroFloat));
    }
  }

  if (config.stages == 2) {
    Value initialGroup = emitAsyncTileLoads(
        rewriter, loc, lhs, rhs, sharedA, sharedB, batch, zero, blockRow,
        blockColumn, zero, threadId, mSize, nSize, kSize, config);
    nvgpu::DeviceAsyncWaitOp::create(rewriter, loc, initialGroup, nullptr);
    gpu::BarrierOp::create(rewriter, loc, gpu::AddressSpace::Workgroup);
  }

  auto kTiles = scf::ForOp::create(
      rewriter, loc, zero, kSize, blockKValue, accumulators,
      [&](OpBuilder &tileBuilder, Location tileLoc, Value kBase,
          ValueRange tileAccumulators) {
        Value currentStage = zero;
        Value nextGroup;
        if (config.stages == 2) {
          Value tileNumber = arith::DivUIOp::create(
              tileBuilder, tileLoc, kBase, blockKValue);
          currentStage = arith::RemUIOp::create(
              tileBuilder, tileLoc, tileNumber,
              indexConstant(tileBuilder, tileLoc, 2));
          Value nextStage = arith::SubIOp::create(
              tileBuilder, tileLoc, one, currentStage);
          Value nextK = arith::AddIOp::create(tileBuilder, tileLoc, kBase,
                                              blockKValue);
          nextGroup = emitAsyncTileLoads(
              tileBuilder, tileLoc, lhs, rhs, sharedA, sharedB, batch,
              nextStage, blockRow, blockColumn, nextK, threadId, mSize, nSize,
              kSize, config);
        } else {
          emitCooperativeTileLoads(
              tileBuilder, tileLoc, lhs, rhs, sharedA, sharedB, batch,
              blockRow, blockColumn, kBase, threadId, mSize, nSize, kSize,
              zeroFloat, config);
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
        }

        auto reduction = scf::ForOp::create(
            tileBuilder, tileLoc, zero, blockKValue, one, tileAccumulators,
            [&](OpBuilder &reductionBuilder, Location reductionLoc,
                Value localK, ValueRange reductionAccumulators) {
              SmallVector<Value> lhsValues;
              SmallVector<Value> rhsValues;
              for (int64_t row = 0; row < microRows; ++row) {
                Value localRow = arith::SubIOp::create(
                    reductionBuilder, reductionLoc, rows[row], blockRow);
                SmallVector<Value> indices;
                if (config.stages == 2)
                  indices.push_back(currentStage);
                indices.append({localRow, localK});
                lhsValues.push_back(memref::LoadOp::create(
                    reductionBuilder, reductionLoc, sharedA, indices));
              }
              for (int64_t column = 0; column < microColumns; ++column) {
                Value localColumn = arith::SubIOp::create(
                    reductionBuilder, reductionLoc, columns[column],
                    blockColumn);
                SmallVector<Value> indices;
                if (config.stages == 2)
                  indices.push_back(currentStage);
                indices.append({localK, localColumn});
                rhsValues.push_back(memref::LoadOp::create(
                    reductionBuilder, reductionLoc, sharedB, indices));
              }

              SmallVector<Value> next;
              int64_t index = 0;
              for (Value lhsValue : lhsValues) {
                for (Value rhsValue : rhsValues) {
                  Value product = arith::MulFOp::create(
                      reductionBuilder, reductionLoc, lhsValue, rhsValue);
                  next.push_back(arith::AddFOp::create(
                      reductionBuilder, reductionLoc,
                      reductionAccumulators[index++], product));
                }
              }
              scf::YieldOp::create(reductionBuilder, reductionLoc, next);
            });
        if (config.stages == 2)
          nvgpu::DeviceAsyncWaitOp::create(tileBuilder, tileLoc, nextGroup,
                                           nullptr);
        gpu::BarrierOp::create(tileBuilder, tileLoc,
                               gpu::AddressSpace::Workgroup);
        scf::YieldOp::create(tileBuilder, tileLoc, reduction.getResults());
      });

  int64_t index = 0;
  for (Value row : rows) {
    for (Value column : columns) {
      Value condition =
          inBounds2D(rewriter, loc, row, mSize, column, nSize);
      guardedStore(rewriter, loc, kTiles.getResult(index++), output,
                   matrixIndices(batch, row, column), condition);
    }
  }
  gpu::TerminatorOp::create(rewriter, loc);
  rewriter.setInsertionPointAfter(launch);
}

void emitSharedMatmulKernel(IRRewriter &rewriter, linalg::MatmulOp matmul,
                            const KernelConfig &config) {
  emitSharedContractionKernel(rewriter, matmul, matmul.getDpsInputs()[0],
                              matmul.getDpsInputs()[1],
                              matmul.getDpsInits()[0], /*batched=*/false,
                              config);
}

void emitSharedBatchMatmulKernel(IRRewriter &rewriter,
                                 linalg::BatchMatmulOp batchMatmul,
                                 const KernelConfig &config) {
  emitSharedContractionKernel(
      rewriter, batchMatmul, batchMatmul.getDpsInputs()[0],
      batchMatmul.getDpsInputs()[1], batchMatmul.getDpsInits()[0],
      /*batched=*/true, config);
}

void emitTensorCoreMatmulKernel(IRRewriter &rewriter,
                                linalg::MatmulOp matmul,
                                const KernelConfig &config) {
  emitTensorCoreContractionKernel(rewriter, matmul, matmul.getDpsInputs()[0],
                                  matmul.getDpsInputs()[1],
                                  matmul.getDpsInits()[0], /*batched=*/false,
                                  config);
}

void emitTensorCoreBatchMatmulKernel(IRRewriter &rewriter,
                                     linalg::BatchMatmulOp batchMatmul,
                                     const KernelConfig &config) {
  emitTensorCoreContractionKernel(
      rewriter, batchMatmul, batchMatmul.getDpsInputs()[0],
      batchMatmul.getDpsInputs()[1], batchMatmul.getDpsInits()[0],
      /*batched=*/true, config);
}

SmallVector<Value> convolutionOutputIndices(OpBuilder &builder, Location loc,
                                            Value batch, Value filter,
                                            Value position, Value owSize) {
  Value ow = arith::RemUIOp::create(builder, loc, position, owSize);
  Value oh = arith::DivUIOp::create(builder, loc, position, owSize);
  return {batch, filter, oh, ow};
}

Value initializeConvolutionTensorCoreAccumulator(
    OpBuilder &builder, Location loc, Value output, Value batch, Value filter0,
    Value filter1, Value position, Value fSize, Value positionSize,
    Value owSize, Value zeroFloat) {
  auto accumulatorType = VectorType::get({2, 2}, builder.getF32Type());
  Value accumulator = zeroVector(builder, loc, accumulatorType);
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t column = 0; column < 2; ++column) {
      Value globalFilter = row == 0 ? filter0 : filter1;
      Value globalPosition = arith::AddIOp::create(
          builder, loc, position, indexConstant(builder, loc, column));
      Value condition = inBounds2D(builder, loc, globalFilter, fSize,
                                   globalPosition, positionSize);
      Value initial = guardedLoad(
          builder, loc, output,
          convolutionOutputIndices(builder, loc, batch, globalFilter,
                                   globalPosition, owSize),
          condition, zeroFloat);
      accumulator = insertVectorElement(builder, loc, initial, accumulator,
                                        {row, column});
    }
  }
  return accumulator;
}

void emitConvolutionKernel(IRRewriter &rewriter,
                           linalg::Conv2DNchwFchwOp conv,
                           const KernelConfig &config, bool tensorCore) {
  Location loc = conv.getLoc();
  Value input = conv.getDpsInputs()[0];
  Value filter = conv.getDpsInputs()[1];
  Value output = conv.getDpsInits()[0];
  auto strides = *getPositivePair(conv.getStrides());
  auto dilations = *getPositivePair(conv.getDilations());
  Value zero = indexConstant(rewriter, loc, 0);
  Value one = indexConstant(rewriter, loc, 1);
  Value zeroFloat = arith::ConstantFloatOp::create(
      rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
  Value strideH = indexConstant(rewriter, loc, strides[0]);
  Value strideW = indexConstant(rewriter, loc, strides[1]);
  Value dilationH = indexConstant(rewriter, loc, dilations[0]);
  Value dilationW = indexConstant(rewriter, loc, dilations[1]);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 0);
  Value fSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value ohSize = memref::DimOp::create(rewriter, loc, output, 2);
  Value owSize = memref::DimOp::create(rewriter, loc, output, 3);
  Value cSize = memref::DimOp::create(rewriter, loc, input, 1);
  Value ihSize = memref::DimOp::create(rewriter, loc, input, 2);
  Value iwSize = memref::DimOp::create(rewriter, loc, input, 3);
  Value khSize = memref::DimOp::create(rewriter, loc, filter, 2);
  Value kwSize = memref::DimOp::create(rewriter, loc, filter, 3);
  Value positionSize = arith::MulIOp::create(rewriter, loc, ohSize, owSize);
  Value reductionSize = arith::MulIOp::create(rewriter, loc, cSize, khSize);
  reductionSize =
      arith::MulIOp::create(rewriter, loc, reductionSize, kwSize);
  Value gridX = ceilDiv(rewriter, loc, positionSize, config.blockN);
  Value gridY = ceilDiv(rewriter, loc, fSize, config.blockM);
  Value blockSize = indexConstant(rewriter, loc, config.threads);
  Value blockKValue = indexConstant(rewriter, loc, config.blockK);

  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::AddressSpace::Workgroup);
  auto sharedAType = MemRefType::get(
      {config.blockM, config.blockK}, rewriter.getF32Type(),
      MemRefLayoutAttrInterface{}, workgroupSpace);
  auto sharedBType = MemRefType::get(
      {config.blockK, config.blockN}, rewriter.getF32Type(),
      MemRefLayoutAttrInterface{}, workgroupSpace);
  auto launch = gpu::LaunchOp::create(
      rewriter, loc, gridX, gridY, nSize, blockSize, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/ValueRange{},
      SmallVector<Type>{sharedAType, sharedBType});
  Block &body = launch.getBody().front();
  rewriter.setInsertionPointToStart(&body);
  Value blockX = launch.getBlockIds().x;
  Value blockY = launch.getBlockIds().y;
  Value batch = launch.getBlockIds().z;
  Value threadId = launch.getThreadIds().x;
  Value sharedA = body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes);
  Value sharedB =
      body.getArgument(gpu::LaunchOp::kNumConfigRegionAttributes + 1);
  Value blockFilter = arith::MulIOp::create(
      rewriter, loc, blockY, indexConstant(rewriter, loc, config.blockM));
  Value blockPosition = arith::MulIOp::create(
      rewriter, loc, blockX, indexConstant(rewriter, loc, config.blockN));

  if (!tensorCore) {
    Value warpWidth = indexConstant(rewriter, loc, 32);
    int64_t threadRows = config.threads / 32;
    int64_t microRows = config.blockM / threadRows;
    int64_t microColumns = config.blockN / 32;
    Value threadFilter =
        arith::DivUIOp::create(rewriter, loc, threadId, warpWidth);
    Value threadPosition =
        arith::RemUIOp::create(rewriter, loc, threadId, warpWidth);
    SmallVector<Value> filters;
    SmallVector<Value> positions;
    for (int64_t row = 0; row < microRows; ++row) {
      Value offset = arith::AddIOp::create(
          rewriter, loc, threadFilter,
          indexConstant(rewriter, loc, row * threadRows));
      filters.push_back(
          arith::AddIOp::create(rewriter, loc, blockFilter, offset));
    }
    for (int64_t column = 0; column < microColumns; ++column) {
      Value offset = arith::AddIOp::create(
          rewriter, loc, threadPosition,
          indexConstant(rewriter, loc, column * 32));
      positions.push_back(
          arith::AddIOp::create(rewriter, loc, blockPosition, offset));
    }
    SmallVector<Value> accumulators;
    for (Value globalFilter : filters) {
      for (Value globalPosition : positions) {
        Value condition = inBounds2D(rewriter, loc, globalFilter, fSize,
                                     globalPosition, positionSize);
        accumulators.push_back(guardedLoad(
            rewriter, loc, output,
            convolutionOutputIndices(rewriter, loc, batch, globalFilter,
                                     globalPosition, owSize),
            condition, zeroFloat));
      }
    }
    auto kTiles = scf::ForOp::create(
        rewriter, loc, zero, reductionSize, blockKValue, accumulators,
        [&](OpBuilder &tileBuilder, Location tileLoc, Value kBase,
            ValueRange tileAccumulators) {
          emitConvolutionTileLoads(
              tileBuilder, tileLoc, input, filter, sharedA, sharedB, batch,
              blockFilter, blockPosition, kBase, threadId, fSize, cSize,
              ihSize, iwSize, ohSize, owSize, khSize, kwSize, reductionSize,
              strideH, strideW, dilationH, dilationW, zeroFloat, config);
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
          auto reduction = scf::ForOp::create(
              tileBuilder, tileLoc, zero, blockKValue, one, tileAccumulators,
              [&](OpBuilder &kBuilder, Location kLoc, Value localK,
                  ValueRange kAccumulators) {
                SmallVector<Value> lhsValues;
                SmallVector<Value> rhsValues;
                for (Value globalFilter : filters) {
                  Value localFilter = arith::SubIOp::create(
                      kBuilder, kLoc, globalFilter, blockFilter);
                  lhsValues.push_back(memref::LoadOp::create(
                      kBuilder, kLoc, sharedA,
                      ValueRange{localFilter, localK}));
                }
                for (Value globalPosition : positions) {
                  Value localPosition = arith::SubIOp::create(
                      kBuilder, kLoc, globalPosition, blockPosition);
                  rhsValues.push_back(memref::LoadOp::create(
                      kBuilder, kLoc, sharedB,
                      ValueRange{localK, localPosition}));
                }
                SmallVector<Value> next;
                size_t index = 0;
                for (Value lhsValue : lhsValues) {
                  for (Value rhsValue : rhsValues) {
                    Value product = arith::MulFOp::create(
                        kBuilder, kLoc, lhsValue, rhsValue);
                    next.push_back(arith::AddFOp::create(
                        kBuilder, kLoc, kAccumulators[index++], product));
                  }
                }
                scf::YieldOp::create(kBuilder, kLoc, next);
              });
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
          scf::YieldOp::create(tileBuilder, tileLoc, reduction.getResults());
        });
    size_t index = 0;
    for (Value globalFilter : filters) {
      for (Value globalPosition : positions) {
        Value condition = inBounds2D(rewriter, loc, globalFilter, fSize,
                                     globalPosition, positionSize);
        guardedStore(
            rewriter, loc, kTiles.getResult(index++), output,
            convolutionOutputIndices(rewriter, loc, batch, globalFilter,
                                     globalPosition, owSize),
            condition);
      }
    }
  } else {
    Value laneId = arith::RemUIOp::create(
        rewriter, loc, threadId, indexConstant(rewriter, loc, 32));
    Value warpId = arith::DivUIOp::create(
        rewriter, loc, threadId, indexConstant(rewriter, loc, 32));
    int64_t warps = config.threads / 32;
    int64_t tilesN = config.blockN / 8;
    int64_t tilesPerWarp = (config.blockM / 16) * tilesN / warps;
    struct ConvWarpTile {
      Value localM;
      Value localN;
      Value filter0;
      Value filter1;
      Value position;
    };
    SmallVector<ConvWarpTile> warpTiles;
    SmallVector<Value> accumulators;
    for (int64_t assignment = 0; assignment < tilesPerWarp; ++assignment) {
      Value tile = arith::AddIOp::create(
          rewriter, loc, warpId,
          indexConstant(rewriter, loc, assignment * warps));
      Value tileM = arith::DivUIOp::create(
          rewriter, loc, tile, indexConstant(rewriter, loc, tilesN));
      Value tileN = arith::RemUIOp::create(
          rewriter, loc, tile, indexConstant(rewriter, loc, tilesN));
      Value localM = arith::MulIOp::create(
          rewriter, loc, tileM, indexConstant(rewriter, loc, 16));
      Value localN = arith::MulIOp::create(
          rewriter, loc, tileN, indexConstant(rewriter, loc, 8));
      Value laneRow = arith::DivUIOp::create(
          rewriter, loc, laneId, indexConstant(rewriter, loc, 4));
      Value laneColumn = arith::RemUIOp::create(
          rewriter, loc, laneId, indexConstant(rewriter, loc, 4));
      laneColumn = arith::MulIOp::create(
          rewriter, loc, laneColumn, indexConstant(rewriter, loc, 2));
      Value filter0 = arith::AddIOp::create(rewriter, loc, blockFilter, localM);
      filter0 = arith::AddIOp::create(rewriter, loc, filter0, laneRow);
      Value filter1 = arith::AddIOp::create(
          rewriter, loc, filter0, indexConstant(rewriter, loc, 8));
      Value position =
          arith::AddIOp::create(rewriter, loc, blockPosition, localN);
      position = arith::AddIOp::create(rewriter, loc, position, laneColumn);
      warpTiles.push_back({localM, localN, filter0, filter1, position});
      accumulators.push_back(initializeConvolutionTensorCoreAccumulator(
          rewriter, loc, output, batch, filter0, filter1, position, fSize,
          positionSize, owSize, zeroFloat));
    }
    auto kTiles = scf::ForOp::create(
        rewriter, loc, zero, reductionSize, blockKValue, accumulators,
        [&](OpBuilder &tileBuilder, Location tileLoc, Value kBase,
            ValueRange tileAccumulators) {
          emitConvolutionTileLoads(
              tileBuilder, tileLoc, input, filter, sharedA, sharedB, batch,
              blockFilter, blockPosition, kBase, threadId, fSize, cSize,
              ihSize, iwSize, ohSize, owSize, khSize, kwSize, reductionSize,
              strideH, strideW, dilationH, dilationW, zeroFloat, config);
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
          SmallVector<Value> next(tileAccumulators.begin(),
                                  tileAccumulators.end());
          for (int64_t localK = 0; localK < config.blockK; localK += 8) {
            Value localKValue = indexConstant(tileBuilder, tileLoc, localK);
            for (size_t tile = 0; tile < warpTiles.size(); ++tile)
              next[tile] = emitTensorCoreMma(
                  tileBuilder, tileLoc, sharedA, sharedB,
                  warpTiles[tile].localM, warpTiles[tile].localN, localKValue,
                  laneId, Value(), next[tile]);
          }
          gpu::BarrierOp::create(tileBuilder, tileLoc,
                                 gpu::AddressSpace::Workgroup);
          scf::YieldOp::create(tileBuilder, tileLoc, next);
        });
    for (size_t tile = 0; tile < warpTiles.size(); ++tile) {
      for (int64_t row = 0; row < 2; ++row) {
        for (int64_t column = 0; column < 2; ++column) {
          Value scalar = vector::ExtractOp::create(
              rewriter, loc, kTiles.getResult(tile),
              ArrayRef<int64_t>{row, column});
          Value globalFilter = row == 0 ? warpTiles[tile].filter0
                                        : warpTiles[tile].filter1;
          Value globalPosition = arith::AddIOp::create(
              rewriter, loc, warpTiles[tile].position,
              indexConstant(rewriter, loc, column));
          Value condition = inBounds2D(rewriter, loc, globalFilter, fSize,
                                       globalPosition, positionSize);
          guardedStore(
              rewriter, loc, scalar, output,
              convolutionOutputIndices(rewriter, loc, batch, globalFilter,
                                       globalPosition, owSize),
              condition);
        }
      }
    }
  }
  gpu::TerminatorOp::create(rewriter, loc);
  rewriter.setInsertionPointAfter(launch);
}

void emitBlockThreadMatmul(IRRewriter &rewriter, linalg::MatmulOp matmul) {
  Location loc = matmul.getLoc();
  Value lhs = matmul.getDpsInputs()[0];
  Value rhs = matmul.getDpsInputs()[1];
  Value output = matmul.getDpsInits()[0];
  Value zero = indexConstant(rewriter, loc, 0);
  Value one = indexConstant(rewriter, loc, 1);
  Value blockM = indexConstant(rewriter, loc, 8);
  Value blockN = indexConstant(rewriter, loc, 32);
  Value mSize = memref::DimOp::create(rewriter, loc, output, 0);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, 1);

  auto blocks = scf::ParallelOp::create(
      rewriter, loc, ValueRange{zero, zero}, ValueRange{mSize, nSize},
      ValueRange{blockM, blockN},
      [&](OpBuilder &blockBuilder, Location blockLoc, ValueRange blockIndices) {
        scf::ParallelOp::create(
            blockBuilder, blockLoc, ValueRange{zero, zero},
            ValueRange{blockM, blockN}, ValueRange{one, one},
            [&](OpBuilder &threadBuilder, Location threadLoc,
                ValueRange threadIndices) {
              Value row = arith::AddIOp::create(
                  threadBuilder, threadLoc, blockIndices[0], threadIndices[0]);
              Value column = arith::AddIOp::create(
                  threadBuilder, threadLoc, blockIndices[1], threadIndices[1]);
              Value condition = inBounds2D(threadBuilder, threadLoc, row, mSize,
                                           column, nSize);
              scf::IfOp::create(
                  threadBuilder, threadLoc, condition,
                  [&](OpBuilder &ifBuilder, Location ifLoc) {
                    Value initial = memref::LoadOp::create(
                        ifBuilder, ifLoc, output, ValueRange{row, column});
                    auto reduction = scf::ForOp::create(
                        ifBuilder, ifLoc, zero, kSize, one,
                        ValueRange{initial},
                        [&](OpBuilder &reductionBuilder, Location reductionLoc,
                            Value k, ValueRange iterArgs) {
                          Value lhsValue = memref::LoadOp::create(
                              reductionBuilder, reductionLoc, lhs,
                              ValueRange{row, k});
                          Value rhsValue = memref::LoadOp::create(
                              reductionBuilder, reductionLoc, rhs,
                              ValueRange{k, column});
                          Value product = arith::MulFOp::create(
                              reductionBuilder, reductionLoc, lhsValue,
                              rhsValue);
                          Value sum = arith::AddFOp::create(
                              reductionBuilder, reductionLoc, iterArgs[0],
                              product);
                          scf::YieldOp::create(reductionBuilder, reductionLoc,
                                               sum);
                        });
                    memref::StoreOp::create(ifBuilder, ifLoc,
                                            reduction.getResult(0), output,
                                            ValueRange{row, column});
                    scf::YieldOp::create(ifBuilder, ifLoc);
                  });
            });
      });
  rewriter.setInsertionPointAfter(blocks);
}

void emitBlockThreadBatchMatmul(IRRewriter &rewriter,
                                linalg::BatchMatmulOp batchMatmul) {
  Location loc = batchMatmul.getLoc();
  Value lhs = batchMatmul.getDpsInputs()[0];
  Value rhs = batchMatmul.getDpsInputs()[1];
  Value output = batchMatmul.getDpsInits()[0];
  Value zero = indexConstant(rewriter, loc, 0);
  Value one = indexConstant(rewriter, loc, 1);
  Value blockM = indexConstant(rewriter, loc, 8);
  Value blockN = indexConstant(rewriter, loc, 32);
  Value batchSize = memref::DimOp::create(rewriter, loc, output, 0);
  Value mSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 2);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, 2);
  auto blocks = scf::ParallelOp::create(
      rewriter, loc, ValueRange{zero, zero, zero},
      ValueRange{batchSize, mSize, nSize}, ValueRange{one, blockM, blockN},
      [&](OpBuilder &blockBuilder, Location blockLoc, ValueRange blockIndices) {
        scf::ParallelOp::create(
            blockBuilder, blockLoc, ValueRange{zero, zero},
            ValueRange{blockM, blockN}, ValueRange{one, one},
            [&](OpBuilder &threadBuilder, Location threadLoc,
                ValueRange threadIndices) {
              Value row = arith::AddIOp::create(
                  threadBuilder, threadLoc, blockIndices[1], threadIndices[0]);
              Value column = arith::AddIOp::create(
                  threadBuilder, threadLoc, blockIndices[2], threadIndices[1]);
              Value condition = inBounds2D(threadBuilder, threadLoc, row, mSize,
                                           column, nSize);
              scf::IfOp::create(
                  threadBuilder, threadLoc, condition,
                  [&](OpBuilder &ifBuilder, Location ifLoc) {
                    Value initial = memref::LoadOp::create(
                        ifBuilder, ifLoc, output,
                        ValueRange{blockIndices[0], row, column});
                    auto reduction = scf::ForOp::create(
                        ifBuilder, ifLoc, zero, kSize, one,
                        ValueRange{initial},
                        [&](OpBuilder &kBuilder, Location kLoc, Value k,
                            ValueRange iterArgs) {
                          Value lhsValue = memref::LoadOp::create(
                              kBuilder, kLoc, lhs,
                              ValueRange{blockIndices[0], row, k});
                          Value rhsValue = memref::LoadOp::create(
                              kBuilder, kLoc, rhs,
                              ValueRange{blockIndices[0], k, column});
                          Value product = arith::MulFOp::create(
                              kBuilder, kLoc, lhsValue, rhsValue);
                          Value sum = arith::AddFOp::create(
                              kBuilder, kLoc, iterArgs[0], product);
                          scf::YieldOp::create(kBuilder, kLoc, sum);
                        });
                    memref::StoreOp::create(
                        ifBuilder, ifLoc, reduction.getResult(0), output,
                        ValueRange{blockIndices[0], row, column});
                    scf::YieldOp::create(ifBuilder, ifLoc);
                  });
            });
      });
  rewriter.setInsertionPointAfter(blocks);
}

func::FuncOp getOrCreateRuntimeFunction(ModuleOp module, OpBuilder &builder,
                                        StringRef name, FunctionType type) {
  if (auto function = module.lookupSymbol<func::FuncOp>(name))
    return function;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto function = func::FuncOp::create(builder, module.getLoc(), name, type);
  function.setPrivate();
  return function;
}

Value buildAutotuneKey(OpBuilder &builder, Location loc, ValueRange dimensions,
                       int64_t seed) {
  Type i64Type = builder.getI64Type();
  Value hash =
      arith::ConstantIntOp::create(builder, loc, i64Type, seed);
  Value multiplier =
      arith::ConstantIntOp::create(builder, loc, i64Type, 1099511628211LL);
  for (Value dimension : dimensions) {
    Value cast = arith::IndexCastOp::create(builder, loc, i64Type, dimension);
    hash = arith::MulIOp::create(builder, loc, hash, multiplier);
    hash = arith::AddIOp::create(builder, loc, hash, cast);
  }
  return hash;
}

void lowerAutotunedMatmul(IRRewriter &rewriter, ModuleOp module,
                          linalg::MatmulOp matmul) {
  Location loc = matmul.getLoc();
  Value lhs = matmul.getDpsInputs()[0];
  Value output = matmul.getDpsInits()[0];
  rewriter.setInsertionPoint(matmul);
  Value mSize = memref::DimOp::create(rewriter, loc, output, 0);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, 1);
  Value key = buildAutotuneKey(
      rewriter, loc, ValueRange{mSize, nSize, kSize},
      0x434f4e5452414354LL);
  Value candidateCount = arith::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64Type(), 8);
  auto beginType = rewriter.getFunctionType(
      {rewriter.getI64Type(), rewriter.getI64Type()},
      {rewriter.getIndexType()});
  auto endType = rewriter.getFunctionType(
      {rewriter.getI64Type(), rewriter.getIndexType()}, {});
  func::FuncOp begin = getOrCreateRuntimeFunction(
      module, rewriter, "tutorial_autotune_begin", beginType);
  func::FuncOp end = getOrCreateRuntimeFunction(
      module, rewriter, "tutorial_autotune_end", endType);
  Value candidate =
      func::CallOp::create(rewriter, loc, begin, ValueRange{key, candidateCount})
          .getResult(0);

  SmallVector<KernelConfig> profiles{
      {64, 64, 16, 128, 4, 1},   {64, 128, 16, 256, 4, 1},
      {128, 64, 16, 256, 4, 1},  {128, 128, 16, 256, 4, 1},
      {128, 128, 16, 256, 4, 2}, {64, 128, 32, 256, 4, 2},
      {128, 64, 32, 256, 4, 2}};
  SmallVector<int64_t> cases{0, 1, 2, 3, 4, 5, 6};
  auto dispatch = scf::IndexSwitchOp::create(
      rewriter, loc, TypeRange{}, candidate, cases, cases.size());
  for (Region &region : dispatch->getRegions()) {
    Block *block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToEnd(block);
    scf::YieldOp::create(rewriter, loc);
  }

  rewriter.setInsertionPoint(dispatch.getCaseBlock(0).getTerminator());
  emitBlockThreadMatmul(rewriter, matmul);
  for (size_t index = 0; index < profiles.size() - 1; ++index) {
    rewriter.setInsertionPoint(
        dispatch.getCaseBlock(static_cast<unsigned>(index + 1)).getTerminator());
    emitSharedMatmulKernel(rewriter, matmul, profiles[index]);
  }
  rewriter.setInsertionPoint(dispatch.getDefaultBlock().getTerminator());
  emitSharedMatmulKernel(rewriter, matmul, profiles.back());

  rewriter.setInsertionPointAfter(dispatch);
  func::CallOp::create(rewriter, loc, end, ValueRange{key, candidate});
  rewriter.eraseOp(matmul);
}

std::pair<func::FuncOp, func::FuncOp>
getAutotuneRuntimeFunctions(ModuleOp module, OpBuilder &builder) {
  auto beginType = builder.getFunctionType(
      {builder.getI64Type(), builder.getI64Type()}, {builder.getIndexType()});
  auto endType = builder.getFunctionType(
      {builder.getI64Type(), builder.getIndexType()}, {});
  return {getOrCreateRuntimeFunction(module, builder,
                                     "tutorial_autotune_begin", beginType),
          getOrCreateRuntimeFunction(module, builder, "tutorial_autotune_end",
                                     endType)};
}

scf::IndexSwitchOp createAutotuneDispatch(OpBuilder &builder, Location loc,
                                          Value candidate) {
  SmallVector<int64_t> cases{0, 1, 2, 3, 4, 5, 6};
  auto dispatch = scf::IndexSwitchOp::create(
      builder, loc, TypeRange{}, candidate, cases, cases.size());
  for (Region &region : dispatch->getRegions()) {
    Block *block = builder.createBlock(&region);
    builder.setInsertionPointToEnd(block);
    scf::YieldOp::create(builder, loc);
  }
  return dispatch;
}

void lowerAutotunedBatchMatmul(IRRewriter &rewriter, ModuleOp module,
                               linalg::BatchMatmulOp batchMatmul) {
  Location loc = batchMatmul.getLoc();
  Value lhs = batchMatmul.getDpsInputs()[0];
  Value output = batchMatmul.getDpsInits()[0];
  rewriter.setInsertionPoint(batchMatmul);
  Value batch = memref::DimOp::create(rewriter, loc, output, 0);
  Value mSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 2);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, 2);
  Value key = buildAutotuneKey(
      rewriter, loc, ValueRange{batch, mSize, nSize, kSize},
      0x42415443484d4dLL);
  Value count = arith::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64Type(), 8);
  auto [begin, end] = getAutotuneRuntimeFunctions(module, rewriter);
  Value candidate =
      func::CallOp::create(rewriter, loc, begin, ValueRange{key, count})
          .getResult(0);
  auto dispatch = createAutotuneDispatch(rewriter, loc, candidate);
  SmallVector<KernelConfig> profiles{
      {64, 64, 16, 128, 4, 1},   {64, 128, 16, 256, 4, 1},
      {128, 64, 16, 256, 4, 1},  {128, 128, 16, 256, 4, 1},
      {128, 128, 16, 256, 4, 2}, {64, 128, 32, 256, 4, 2},
      {128, 64, 32, 256, 4, 2}};
  rewriter.setInsertionPoint(dispatch.getCaseBlock(0).getTerminator());
  emitBlockThreadBatchMatmul(rewriter, batchMatmul);
  for (size_t index = 0; index < profiles.size() - 1; ++index) {
    rewriter.setInsertionPoint(
        dispatch.getCaseBlock(static_cast<unsigned>(index + 1)).getTerminator());
    emitSharedBatchMatmulKernel(rewriter, batchMatmul, profiles[index]);
  }
  rewriter.setInsertionPoint(dispatch.getDefaultBlock().getTerminator());
  emitSharedBatchMatmulKernel(rewriter, batchMatmul, profiles.back());
  rewriter.setInsertionPointAfter(dispatch);
  func::CallOp::create(rewriter, loc, end, ValueRange{key, candidate});
  rewriter.eraseOp(batchMatmul);
}

void lowerAutotunedConvolution(IRRewriter &rewriter, ModuleOp module,
                               linalg::Conv2DNchwFchwOp conv) {
  Location loc = conv.getLoc();
  Value input = conv.getDpsInputs()[0];
  Value filter = conv.getDpsInputs()[1];
  Value output = conv.getDpsInits()[0];
  rewriter.setInsertionPoint(conv);
  SmallVector<Value> dimensions;
  for (int64_t dimension = 0; dimension < 4; ++dimension)
    dimensions.push_back(
        memref::DimOp::create(rewriter, loc, output, dimension));
  for (int64_t dimension = 1; dimension < 4; ++dimension)
    dimensions.push_back(
        memref::DimOp::create(rewriter, loc, input, dimension));
  dimensions.push_back(memref::DimOp::create(rewriter, loc, filter, 2));
  dimensions.push_back(memref::DimOp::create(rewriter, loc, filter, 3));
  auto strides = *getPositivePair(conv.getStrides());
  auto dilations = *getPositivePair(conv.getDilations());
  int64_t seed = 0x434f4e56474d4dLL ^ (strides[0] << 12) ^
                 (strides[1] << 8) ^ (dilations[0] << 4) ^ dilations[1];
  Value key = buildAutotuneKey(rewriter, loc, dimensions, seed);
  Value count = arith::ConstantIntOp::create(
      rewriter, loc, rewriter.getI64Type(), 8);
  auto [begin, end] = getAutotuneRuntimeFunctions(module, rewriter);
  Value candidate =
      func::CallOp::create(rewriter, loc, begin, ValueRange{key, count})
          .getResult(0);
  auto dispatch = createAutotuneDispatch(rewriter, loc, candidate);
  SmallVector<KernelConfig> profiles{
      {64, 64, 16, 128, 4, 1},   {64, 128, 16, 256, 4, 1},
      {128, 64, 16, 256, 4, 1},  {128, 128, 16, 256, 4, 1},
      {64, 64, 32, 128, 4, 1},   {64, 128, 32, 256, 4, 1},
      {128, 64, 32, 256, 4, 1},  {128, 128, 32, 256, 4, 1}};
  for (size_t index = 0; index < profiles.size() - 1; ++index) {
    rewriter.setInsertionPoint(
        dispatch.getCaseBlock(static_cast<unsigned>(index)).getTerminator());
    emitConvolutionKernel(rewriter, conv, profiles[index],
                          /*tensorCore=*/false);
  }
  rewriter.setInsertionPoint(dispatch.getDefaultBlock().getTerminator());
  emitConvolutionKernel(rewriter, conv, profiles.back(),
                        /*tensorCore=*/false);
  rewriter.setInsertionPointAfter(dispatch);
  func::CallOp::create(rewriter, loc, end, ValueRange{key, candidate});
  rewriter.eraseOp(conv);
}

} // namespace

void LowerContractionToGpuPass::runOnOperation() {
  KernelConfig config{blockM, blockN, blockK, threads, vectorWidth, stages};
  if (strategy != "shared-fp32" && strategy != "tensorcore-tf32" &&
      strategy != "autotuned") {
    getOperation().emitError() << "unknown GPU contraction strategy: "
                               << strategy;
    return signalPassFailure();
  }
  if (target != "sm_89")
    return;
  if (config.blockM <= 0 || config.blockN <= 0 || config.blockK <= 0 ||
      config.threads < 128 || config.threads > 512 ||
      config.threads % 32 != 0 || config.blockN % 32 != 0 ||
      config.blockM % (config.threads / 32) != 0 || config.vectorWidth <= 0 ||
      config.blockK % config.vectorWidth != 0 ||
      config.blockN % config.vectorWidth != 0 ||
      (config.blockM * config.blockK) %
              (config.vectorWidth * config.threads) !=
          0 ||
      (config.blockK * config.blockN) %
              (config.vectorWidth * config.threads) !=
          0 ||
      (config.stages == 2 && config.vectorWidth != 4) ||
      config.stages <= 0 || config.stages > 2) {
    getOperation().emitError()
        << "invalid GPU contraction configuration: block=" << config.blockM
        << "x" << config.blockN << "x" << config.blockK
        << " threads=" << config.threads
        << " vector-width=" << config.vectorWidth
        << " stages=" << config.stages;
    return signalPassFailure();
  }
  int64_t sharedBytes = config.stages *
                        (config.blockM * config.blockK +
                         config.blockK * config.blockN) *
                        static_cast<int64_t>(sizeof(float));
  if (sharedBytes > 48 * 1024) {
    getOperation().emitError()
        << "GPU contraction configuration uses " << sharedBytes
        << " bytes of workgroup memory; limit is 49152";
    return signalPassFailure();
  }
  SmallVector<linalg::MatmulOp> worklist;
  getOperation().walk([&](linalg::MatmulOp matmul) {
    if (isSupportedMatmul(matmul))
      worklist.push_back(matmul);
  });
  SmallVector<linalg::BatchMatmulOp> batchWorklist;
  getOperation().walk([&](linalg::BatchMatmulOp batchMatmul) {
    if (isSupportedBatchMatmul(batchMatmul))
      batchWorklist.push_back(batchMatmul);
  });
  SmallVector<linalg::Conv2DNchwFchwOp> convolutionWorklist;
  getOperation().walk([&](linalg::Conv2DNchwFchwOp conv) {
    if (isSupportedConvolution(conv))
      convolutionWorklist.push_back(conv);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::MatmulOp matmul : worklist) {
    if (strategy == "autotuned") {
      lowerAutotunedMatmul(rewriter, getOperation(), matmul);
      continue;
    }
    rewriter.setInsertionPoint(matmul);
    if (strategy == "tensorcore-tf32")
      emitTensorCoreMatmulKernel(rewriter, matmul, config);
    else
      emitSharedMatmulKernel(rewriter, matmul, config);
    rewriter.eraseOp(matmul);
  }
  for (linalg::BatchMatmulOp batchMatmul : batchWorklist) {
    if (strategy == "autotuned") {
      lowerAutotunedBatchMatmul(rewriter, getOperation(), batchMatmul);
      continue;
    }
    rewriter.setInsertionPoint(batchMatmul);
    if (strategy == "tensorcore-tf32")
      emitTensorCoreBatchMatmulKernel(rewriter, batchMatmul, config);
    else
      emitSharedBatchMatmulKernel(rewriter, batchMatmul, config);
    rewriter.eraseOp(batchMatmul);
  }
  for (linalg::Conv2DNchwFchwOp conv : convolutionWorklist) {
    if (strategy == "autotuned") {
      lowerAutotunedConvolution(rewriter, getOperation(), conv);
      continue;
    }
    rewriter.setInsertionPoint(conv);
    emitConvolutionKernel(rewriter, conv, config,
                          strategy == "tensorcore-tf32");
    rewriter.eraseOp(conv);
  }
}

} // namespace mlir::tutorial
