#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/framework.h"


int main() {
    using arrus::ops::us4r::TxRxSequence;
    using arrus::ops::us4r::Pulse;
    using arrus::ops::us4r::Rx;
    using arrus::ops::us4r::Tx;

    using arrus::DeviceId;
    using arrus::DeviceType;

    using arrus::Variable;

    arrus::SessionSettings settings = arrus::io::readSessionSettings(
        "test.prototxt");
    arrus::Session::Handle session = arrus::createSession(settings);

    auto us4rId = DeviceId(DeviceType::Us4R, 0);
    auto gpuId = DeviceId(DeviceType::GPU, 0);

    auto seq = TxRxSequence(
        us4rId,
        {{Tx({true, false}, {0.0, 0.0},
             Pulse(10e6, 1.5, false)),
             Rx({true, false}, {0, 2048}, 1)}
        },
        160e-6,
        {1.0f, 2.0f});
    arrus::CircularQueue::Element rf = seq.getData();
//    arrus::CircularQueue::Element planeWaveAngles = seq.getMetadata("focus");

    // Imaging pipeline.
    arrus::Tensor iq = arrus::ops::downConversion(gpuId, rf);
    arrus::Tensor rfImg = arrus::ops::reconstructRFImage(gpuId, iq);
    arrus::Tensor envelope = arrus::ops::hilbert(gpuId, rfImg);
    arrus::Tensor bmode = arrus::ops::obtainBModeImage(gpuId, envelope);
    // TODO scan conversion

    // Dumping pipeline.
    arrus::Op save = arrus::ops::io::SaveOp(rf, "/home/pjarosik/tmp/directory/");

    // Start acquisition on the us4r.
    session.run(seq.start());
    while(true) {
        if(mode == "bmode") {
            auto const &image = session.run(bmode);
            // actually should return a list of output values of given operation/tensor evaluation
        }
        else {
            session.run(save); // Ignoring empty results.
        }
    }
}
