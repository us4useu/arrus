# Ultrasound Device Interface Sketch

Based on `arrus/core/api/devices/Ultrasound.h`.

```cpp
namespace arrus::devices {

class Ultrasound : public Device {
public:
    // Execution
    UploadResult upload(const Scheme& scheme);
    void start();
    void stop();
    void trigger(bool sync = false, std::optional<long long> timeout = {});
    void sync(std::optional<long long> timeout);

    // Introspection
    float getSamplingFrequency() const;
    float getCurrentSamplingFrequency() const;
    Probe* getProbe(Ordinal ordinal);
    int getNumberOfProbes() const;

    // Sub-sequences
    UploadResult setSubsequences(const std::vector<Slice>& slices,
                                 const std::vector<std::optional<float>>& sris);

    // Device-specific capabilities (e.g. "us4useu.us4r.v1")
    void* getCapability(const std::string& id);
    std::vector<std::string> getCapabilities() const;
};

}
```

Notes:
- `Scheme` needs to become device-agnostic (currently `arrus::ops::us4r::Scheme` — see TODO on `Ultrasound.h:13`).
- `UploadResult` bundles the buffer and per-sequence metadata (today: `std::pair<Buffer::SharedHandle, std::vector<Metadata::SharedHandle>>`).
