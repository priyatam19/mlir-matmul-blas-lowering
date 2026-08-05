#include "lib/LowerContractionToGpu.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

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
                            Value shared, Value globalRow, Value globalColumn,
                            Value localRow, Value localColumn, Value rowLimit,
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
      builder, loc, vectorType, source, ValueRange{globalRow, globalColumn});
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
    Value scalar = guardedLoad(builder, loc, source,
                               ValueRange{globalRow, sourceColumn}, condition,
                               zeroFloat);
    memref::StoreOp::create(builder, loc, scalar, shared,
                            ValueRange{localRow, destinationColumn});
  }
  scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(ifOp);
}

void emitCooperativeTileLoads(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs, Value sharedA, Value sharedB,
                              Value blockRow, Value blockColumn, Value kBase,
                              Value threadId, Value mSize, Value nSize,
                              Value kSize, Value zeroFloat,
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
    emitVectorizedTileLoad(builder, loc, lhs, sharedA, globalRow, globalK,
                           localRow, localK, mSize, kSize, zeroFloat,
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
    emitVectorizedTileLoad(builder, loc, rhs, sharedB, globalK, globalColumn,
                           localK, localColumn, kSize, nSize, zeroFloat,
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
                          Value shared, Value stage, Value globalRow,
                          Value globalColumn, Value localRow, Value localColumn,
                          Value rowLimit, Value columnLimit,
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
      ValueRange{safeRow, safeColumn}, builder.getIndexAttr(vectorWidth),
      sourceElements,
      vectorWidth == 4 ? builder.getUnitAttr() : UnitAttr());
}

Value emitAsyncTileLoads(OpBuilder &builder, Location loc, Value lhs,
                         Value rhs, Value sharedA, Value sharedB,
                         Value stage, Value blockRow, Value blockColumn,
                         Value kBase, Value threadId, Value mSize, Value nSize,
                         Value kSize, const KernelConfig &config) {
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
        builder, loc, lhs, sharedA, stage, globalRow, globalK, localRow,
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
        builder, loc, rhs, sharedB, stage, globalK, globalColumn, localK,
        localColumn, kSize, nSize, config.vectorWidth));
  }

  return nvgpu::DeviceAsyncCreateGroupOp::create(
      builder, loc, nvgpu::DeviceAsyncTokenType::get(builder.getContext()),
      tokens);
}

void lowerMatmul(IRRewriter &rewriter, linalg::MatmulOp matmul,
                 const KernelConfig &config) {
  Location loc = matmul.getLoc();
  Value lhs = matmul.getDpsInputs()[0];
  Value rhs = matmul.getDpsInputs()[1];
  Value output = matmul.getDpsInits()[0];

  rewriter.setInsertionPoint(matmul);
  Value one = indexConstant(rewriter, loc, 1);
  Value zero = indexConstant(rewriter, loc, 0);
  Value zeroFloat = arith::ConstantFloatOp::create(
      rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
  Value mSize = memref::DimOp::create(rewriter, loc, output, 0);
  Value nSize = memref::DimOp::create(rewriter, loc, output, 1);
  Value kSize = memref::DimOp::create(rewriter, loc, lhs, 1);
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
      rewriter, loc, gridX, gridY, one, blockSize, one, one,
      /*dynamicSharedMemorySize=*/nullptr, /*asyncTokenType=*/nullptr,
      /*asyncDependencies=*/ValueRange{}, workgroupTypes);

  Block &body = launch.getBody().front();
  rewriter.setInsertionPointToStart(&body);
  Value blockX = launch.getBlockIds().x;
  Value blockY = launch.getBlockIds().y;
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
          rewriter, loc, output, ValueRange{row, column}, condition, zeroFloat));
    }
  }

  if (config.stages == 2) {
    Value initialGroup = emitAsyncTileLoads(
        rewriter, loc, lhs, rhs, sharedA, sharedB, zero, blockRow, blockColumn,
        zero, threadId, mSize, nSize, kSize, config);
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
              tileBuilder, tileLoc, lhs, rhs, sharedA, sharedB, nextStage,
              blockRow, blockColumn, nextK, threadId, mSize, nSize, kSize,
              config);
        } else {
          emitCooperativeTileLoads(tileBuilder, tileLoc, lhs, rhs, sharedA,
                                   sharedB, blockRow, blockColumn, kBase,
                                   threadId, mSize, nSize, kSize, zeroFloat,
                                   config);
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
                   ValueRange{row, column}, condition);
    }
  }
  gpu::TerminatorOp::create(rewriter, loc);
  rewriter.eraseOp(matmul);
}

} // namespace

void LowerContractionToGpuPass::runOnOperation() {
  KernelConfig config{blockM, blockN, blockK, threads, vectorWidth, stages};
  if (strategy != "shared-fp32" && strategy != "tensorcore-tf32") {
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
  if (strategy == "tensorcore-tf32")
    return;

  SmallVector<linalg::MatmulOp> worklist;
  getOperation().walk([&](linalg::MatmulOp matmul) {
    if (isSupportedMatmul(matmul))
      worklist.push_back(matmul);
  });

  IRRewriter rewriter(&getContext());
  for (linalg::MatmulOp matmul : worklist)
    lowerMatmul(rewriter, matmul, config);
}

} // namespace mlir::tutorial
