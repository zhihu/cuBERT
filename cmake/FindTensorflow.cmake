include(FindPackageHandleStandardArgs)

find_path(tensorflow_INCLUDE_DIR NAMES tensorflow/c/c_api.h)
find_library(tensorflow_LIBRARIES NAMES tensorflow)

find_package_handle_standard_args(tensorflow DEFAULT_MSG tensorflow_LIBRARIES tensorflow_INCLUDE_DIR)

mark_as_advanced(
        tensorflow_LIBRARIES
        tensorflow_INCLUDE_DIR)
