add_mlir_library(ConvertMatMulToBlas
    ConvertMatMulToBlas.cpp
    TileMatMulForCache.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/ConvertMatMulToBlas

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
)
