add_mlir_dialect_library(MLIROpenMPTransforms
  MarkDeclareTarget.cpp
  OpenMPOffloadPrivatizationPrepare.cpp
  OpenMPSIMDInlineBoost.cpp

  DEPENDS
  MLIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRPass
  MLIRTransforms
  )
