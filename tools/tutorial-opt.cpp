#include "lib/ConvertBatchMatMulToCublas.h"
#include "lib/ConvertConv2DNchwToCudnn.h"
#include "lib/ConvertMatMulToBlas.h"
#include "lib/EnableFastMathForLoops.h"
#include "lib/TileBatchMatMulForGpu.h"
#include "lib/TileConv2DNchwForGpu.h"
#include "lib/TileMatMulForCache.h"
#include "lib/TileMatMulForGpu.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

std::unique_ptr<mlir::Pass> createConvertMatmulToBlasLibraryCallPass() {
  return std::make_unique<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
}

std::unique_ptr<mlir::Pass> createConvertBatchMatMulToCublasPass() {
  return std::make_unique<mlir::tutorial::ConvertBatchMatMulToCublasPass>();
}

std::unique_ptr<mlir::Pass> createConvertConv2DNchwToCudnnPass() {
  return std::make_unique<mlir::tutorial::ConvertConv2DNchwToCudnnPass>();
}

std::unique_ptr<mlir::Pass> createTileMatMulForCachePass() {
  return std::make_unique<mlir::tutorial::TileMatMulForCachePass>();
}

std::unique_ptr<mlir::Pass> createTileBatchMatMulForGpuPass() {
  return std::make_unique<mlir::tutorial::TileBatchMatMulForGpuPass>();
}

std::unique_ptr<mlir::Pass> createTileConv2DNchwForGpuPass() {
  return std::make_unique<mlir::tutorial::TileConv2DNchwForGpuPass>();
}

std::unique_ptr<mlir::Pass> createTileMatMulForGpuPass() {
  return std::make_unique<mlir::tutorial::TileMatMulForGpuPass>();
}

std::unique_ptr<mlir::Pass> createEnableFastMathForLoopsPass() {
  return std::make_unique<mlir::tutorial::EnableFastMathForLoopsPass>();
}

void linalgToBufferizationPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);
}

// Shared by both BufferizationToLLVMPipelineBuilder and
// BufferizationToLLVMGenericPipelineBuilder: everything after linalg has
// been reduced to loops (either by the BLAS pass leaving nothing behind, or
// by createConvertLinalgToLoopsPass in the generic-fallback case).
void addStandardLoweringTail(mlir::OpPassManager &manager) {
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createLowerAffinePass());
  manager.addPass(mlir::affine::createLoopFusionPass());
  manager.addPass(mlir::affine::createAffineVectorize());
  manager.addPass(mlir::createSCFToControlFlowPass());

  // Convert to LLVM - order matters here
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertMathToLLVMPass());
  manager.addPass(
      mlir::createConvertMathToLibmPass()); // For bert model to lower math.erf
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

void BufferizationToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // CRITICAL: Replace matmuls with BLAS calls AFTER bufferization but BEFORE
  // other LLVM conversions
  manager.addPass(createConvertMatmulToBlasLibraryCallPass());

  // Convert remaining linalg ops to loops
  manager.addPass(mlir::createConvertLinalgToLoopsPass());

  addStandardLoweringTail(manager);
}

// Same as BufferizationToLLVMPipelineBuilder but skips the BLAS conversion
// entirely, so every linalg op (including matmul) goes through the generic
// loops fallback. Exists to make that fallback path measurable on its own
// rather than only reachable for shapes/dtypes the BLAS pass rejects.
//
// createConvertLinalgToLoopsPass emits plain (non-fastmath) arith ops, which
// LLVM's loop vectorizer will not reorder regardless of -mcpu or a
// downstream -ffast-math flag -- see EnableFastMathForLoopsPass for why.
void BufferizationToLLVMGenericPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  manager.addPass(createEnableFastMathForLoopsPass());

  addStandardLoweringTail(manager);
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::PassRegistration<mlir::tutorial::ConvertMatmulToBlasLibraryCallPass>();
  mlir::PassRegistration<mlir::tutorial::ConvertBatchMatMulToCublasPass>();
  mlir::PassRegistration<mlir::tutorial::ConvertConv2DNchwToCudnnPass>();
  mlir::PassRegistration<mlir::tutorial::TileMatMulForCachePass>();
  mlir::PassRegistration<mlir::tutorial::TileBatchMatMulForGpuPass>();
  mlir::PassRegistration<mlir::tutorial::TileConv2DNchwForGpuPass>();
  mlir::PassRegistration<mlir::tutorial::TileMatMulForGpuPass>();
  mlir::PassRegistration<mlir::tutorial::EnableFastMathForLoopsPass>();

  mlir::PassPipelineRegistration<>(
      "linalg-to-bufferization",
      "Run passes to lower the linalg dialect to bufferization",
      linalgToBufferizationPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "bufferization-to-llvm", "Run passes to lower bufferized code to LLVM",
      BufferizationToLLVMPipelineBuilder);

  mlir::PassPipelineRegistration<>(
      "bufferization-to-llvm-generic",
      "Like bufferization-to-llvm, but skips the BLAS conversion so matmul "
      "goes through the generic loops fallback (with fastmath enabled so it "
      "actually vectorizes)",
      BufferizationToLLVMGenericPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
