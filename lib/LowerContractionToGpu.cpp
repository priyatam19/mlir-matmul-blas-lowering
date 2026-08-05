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

void emitCooperativeTileLoads(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs, Value sharedA, Value sharedB,
                              Value blockRow, Value blockColumn, Value kBase,
                              Value threadId, Value mSize, Value nSize,
                              Value kSize, Value zeroFloat,
                              const KernelConfig &config) {
  Value threadsValue = indexConstant(builder, loc, config.threads);
  Value totalA = indexConstant(builder, loc, config.blockM * config.blockK);
  Value totalB = indexConstant(builder, loc, config.blockK * config.blockN);
  Value blockKValue = indexConstant(builder, loc, config.blockK);
  Value blockNValue = indexConstant(builder, loc, config.blockN);

  scf::ForOp::create(
      builder, loc, threadId, totalA, threadsValue, ValueRange{},
      [&](OpBuilder &loadBuilder, Location loadLoc, Value linear,
          ValueRange) {
        Value localRow = arith::DivUIOp::create(loadBuilder, loadLoc, linear,
                                                blockKValue);
        Value localK = arith::RemUIOp::create(loadBuilder, loadLoc, linear,
                                              blockKValue);
        Value globalRow = arith::AddIOp::create(loadBuilder, loadLoc, blockRow,
                                                localRow);
        Value globalK = arith::AddIOp::create(loadBuilder, loadLoc, kBase,
                                              localK);
        Value condition = inBounds2D(loadBuilder, loadLoc, globalRow, mSize,
                                     globalK, kSize);
        Value loaded = guardedLoad(loadBuilder, loadLoc, lhs,
                                   ValueRange{globalRow, globalK}, condition,
                                   zeroFloat);
        memref::StoreOp::create(loadBuilder, loadLoc, loaded, sharedA,
                                ValueRange{localRow, localK});
        scf::YieldOp::create(loadBuilder, loadLoc);
      });

  scf::ForOp::create(
      builder, loc, threadId, totalB, threadsValue, ValueRange{},
      [&](OpBuilder &loadBuilder, Location loadLoc, Value linear,
          ValueRange) {
        Value localK = arith::DivUIOp::create(loadBuilder, loadLoc, linear,
                                              blockNValue);
        Value localColumn = arith::RemUIOp::create(
            loadBuilder, loadLoc, linear, blockNValue);
        Value globalK = arith::AddIOp::create(loadBuilder, loadLoc, kBase,
                                              localK);
        Value globalColumn = arith::AddIOp::create(
            loadBuilder, loadLoc, blockColumn, localColumn);
        Value condition = inBounds2D(loadBuilder, loadLoc, globalK, kSize,
                                     globalColumn, nSize);
        Value loaded = guardedLoad(loadBuilder, loadLoc, rhs,
                                   ValueRange{globalK, globalColumn}, condition,
                                   zeroFloat);
        memref::StoreOp::create(loadBuilder, loadLoc, loaded, sharedB,
                                ValueRange{localK, localColumn});
        scf::YieldOp::create(loadBuilder, loadLoc);
      });
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
  auto sharedAType = MemRefType::get({config.blockM, config.blockK},
                                     rewriter.getF32Type(),
                                     MemRefLayoutAttrInterface{},
                                     workgroupSpace);
  auto sharedBType = MemRefType::get({config.blockK, config.blockN},
                                     rewriter.getF32Type(),
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

  auto kTiles = scf::ForOp::create(
      rewriter, loc, zero, kSize, blockKValue, accumulators,
      [&](OpBuilder &tileBuilder, Location tileLoc, Value kBase,
          ValueRange tileAccumulators) {
        emitCooperativeTileLoads(tileBuilder, tileLoc, lhs, rhs, sharedA,
                                 sharedB, blockRow, blockColumn, kBase, threadId,
                                 mSize, nSize, kSize, zeroFloat, config);
        gpu::BarrierOp::create(tileBuilder, tileLoc,
                               gpu::AddressSpace::Workgroup);

        auto reduction = scf::ForOp::create(
            tileBuilder, tileLoc, zero, blockKValue, one, tileAccumulators,
            [&](OpBuilder &reductionBuilder, Location reductionLoc,
                Value localK, ValueRange reductionAccumulators) {
              SmallVector<Value> lhsValues;
              SmallVector<Value> rhsValues;
              for (int64_t row = 0; row < microRows; ++row) {
                Value localRow = arith::SubIOp::create(
                    reductionBuilder, reductionLoc, rows[row], blockRow);
                lhsValues.push_back(memref::LoadOp::create(
                    reductionBuilder, reductionLoc, sharedA,
                    ValueRange{localRow, localK}));
              }
              for (int64_t column = 0; column < microColumns; ++column) {
                Value localColumn = arith::SubIOp::create(
                    reductionBuilder, reductionLoc, columns[column],
                    blockColumn);
                rhsValues.push_back(memref::LoadOp::create(
                    reductionBuilder, reductionLoc, sharedB,
                    ValueRange{localK, localColumn}));
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
