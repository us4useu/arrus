#include <gtest/gtest.h>

// TODO remove
#include <build/core/proto/SessionSettings.pb.h>
#include <google/protobuf/text_format.h>

#include "core/proto/SessionSettings.pb.h"

TEST(test, test) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    arrus::proto::Us4OEMSettings settings;
    settings.set_pgagain(10.2);
    settings.set_lnagain(10.1);
    std::string input= "pgaGain: 8\nlnaGain: 10.1";
    google::protobuf::TextFormat::ParseFromString(input, &settings);
    std::cerr << "diagnostic message: " << settings.pgagain() << std::endl;

    // Using Any
    arrus::proto::SessionSettings sessionSettings;
    sessionSettings.mutable_systemsettings()->PackFrom(settings);
    std::string output;
    google::protobuf::TextFormat::PrintToString(sessionSettings, &output);
    std::cerr << "Session settings: " << output << std::endl;
}