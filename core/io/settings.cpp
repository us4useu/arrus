#include "arrus/common/compiler.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
// TODO(146) should point to arrus/core/io/...
#include "io/proto/session/SessionSettings.pb.h"
COMPILER_POP_DIAGNOSTIC_STATE

#include "arrus/core/api/io/settings.h"


namespace arrus::io {

SessionSettings readSessionSettings(const std::string &) {
//     get proto objects: SessionSettings
//     validate
//     convert to SessionSettings
}



}