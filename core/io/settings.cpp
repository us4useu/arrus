#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "arrus/core/api/io/settings.h"

// TODO(146) should point to arrus/core/io/...
#include "io/proto/session/SessionSettings.pb.h"

namespace arrus::io {

SessionSettings readSessionSettings(const std::string &file) {
//     get proto objects: SessionSettings
//     validate
//     convert to SessionSettings
}



}