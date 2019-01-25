project(external)

include_directories(./)

find_package(Threads REQUIRED)
add_definitions(-DHAVE_PTHREAD)

add_library(external
        # protobuf
        google/protobuf/stubs/common.cc
        google/protobuf/stubs/status.cc
        google/protobuf/stubs/stringpiece.cc
        google/protobuf/stubs/strutil.cc
        google/protobuf/stubs/int128.cc
        google/protobuf/stubs/mathlimits.cc
        google/protobuf/stubs/stringprintf.cc
        google/protobuf/stubs/io_win32.cc
        google/protobuf/stubs/substitute.cc
        google/protobuf/stubs/structurally_valid.cc
        google/protobuf/io/coded_stream.cc
        google/protobuf/io/zero_copy_stream.cc
        google/protobuf/io/zero_copy_stream_impl_lite.cc
        google/protobuf/io/strtod.cc
        google/protobuf/io/tokenizer.cc
        google/protobuf/io/zero_copy_stream_impl.cc
        google/protobuf/message_lite.cc
        google/protobuf/arena.cc
        google/protobuf/generated_message_util.cc
        google/protobuf/implicit_weak_message.cc
        google/protobuf/wire_format_lite.cc
        google/protobuf/repeated_field.cc
        google/protobuf/arenastring.cc
        google/protobuf/extension_set.cc
        google/protobuf/generated_message_table_driven.cc
        google/protobuf/generated_message_table_driven_lite.cc
        google/protobuf/wire_format.cc
        google/protobuf/descriptor.cc
        google/protobuf/descriptor.pb.cc
        google/protobuf/descriptor_database.cc
        google/protobuf/dynamic_message.cc
        google/protobuf/text_format.cc
        google/protobuf/unknown_field_set.cc
        google/protobuf/generated_message_reflection.cc
        google/protobuf/reflection_ops.cc
        google/protobuf/message.cc
        google/protobuf/map_field.cc
        google/protobuf/any.cc
        google/protobuf/extension_set_heavy.cc
        # tensorflow
        tensorflow/core/framework/attr_value.pb.cc
        tensorflow/core/framework/function.pb.cc
        tensorflow/core/framework/graph.pb.cc
        tensorflow/core/framework/node_def.pb.cc
        tensorflow/core/framework/op_def.pb.cc
        tensorflow/core/framework/resource_handle.pb.cc
        tensorflow/core/framework/tensor.pb.cc
        tensorflow/core/framework/tensor_shape.pb.cc
        tensorflow/core/framework/types.pb.cc
        tensorflow/core/framework/versions.pb.cc)
