add_mlir_library(ConvertMatMulToBlas
    ConvertBatchMatMulToCublas.cpp
    ConvertConv2DNchwToCudnn.cpp
    ConvertMatMulToBlas.cpp
    EnableFastMathForLoops.cpp
    LowerContractionToGpu.cpp
    TileBatchMatMulForGpu.cpp
    TileConv2DNchwForGpu.cpp
    TileMatMulForCache.cpp
    TileMatMulForGpu.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/ConvertMatMulToBlas

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRNVGPUDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
)
