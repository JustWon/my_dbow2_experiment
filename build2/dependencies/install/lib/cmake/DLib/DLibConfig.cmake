FIND_LIBRARY(DLib_LIBRARY DLib
    PATHS /home/dongwonshin/Desktop/DBoW2/build2/dependencies/install/lib
)
FIND_PATH(DLib_INCLUDE_DIR DLibConfig.cmake
    PATHS /home/dongwonshin/Desktop/DBoW2/build2/dependencies/install/include/DLib 
)
LIST(APPEND DLib_INCLUDE_DIR
  ${DLib_INCLUDE_DIR}/../ ${DLib_INCLUDE_DIR}/../DUtils
  ${DLib_INCLUDE_DIR}/../DUtilsCV ${DLib_INCLUDE_DIR}/../DVision)
SET(DLib_LIBRARIES ${DLib_LIBRARY})
SET(DLib_LIBS ${DLib_LIBRARY})
SET(DLib_INCLUDE_DIRS ${DLib_INCLUDE_DIR})
