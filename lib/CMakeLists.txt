add_mlir_library(ConvertMatMulToBlas
    ConvertBatchMatMulToCublas.cpp
    ConvertMatMulToBlas.cpp
    TileBatchMatMulForGpu.cpp
    TileMatMulForCache.cpp
    TileMatMulForGpu.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/ConvertMatMulToBlas

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
)
