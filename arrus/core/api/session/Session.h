#ifndef ARRUS_CORE_API_SESSION_SESSION_H
#define ARRUS_CORE_API_SESSION_SESSION_H

#include "arrus/core/api/common/Parameters.h"
#include "arrus/core/api/common/Slice.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/common/macros.h"
#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/session/SessionSettings.h"
#include "arrus/core/api/session/UploadResult.h"

namespace arrus::session {

/**
 * A communication session with the device.
 */
class Session {
public:
    using Handle = std::unique_ptr<Session>;

    /**
     * Session state.
     *
     * - STOPPED: the session is stopped (no device is running).
     * - STARTED: the session is started (at least one of the session devices is running).
     * - CLOSED: the session was closed (the connection to all the session devices was closed).
     */
    enum class State {
        STOPPED, STARTED, CLOSED
    };

    /**
     * Returns a Session State name.
     *
     * @param state Session State
     * @return a Session State name (string)
     */
    static std::string getSessionStateAsString(const State state) {
        switch(state) {
            case State::STOPPED: return "STOPPED";
            case State::STARTED: return "STARTED";
            case State::CLOSED: return "CLOSED";
            default:
                throw IllegalArgumentException("Unrecognized Session State.");
        }
    }

    /**
     * Returns a handle to device with given Id. The string format is:
     * /DeviceType:Ordinal, e.g. "/Us4R:0".
     *
     * @param deviceId device identifier
     * @return a handle to the device
     */
    virtual arrus::devices::Device * getDevice(const std::string &deviceId) = 0;

    /**
     * Returns a handle to device with given Id.
     *
     * @param deviceId device identifier
     * @return a handle to the device
     */
    virtual arrus::devices::Device * getDevice(const arrus::devices::DeviceId &deviceId) = 0;

    /**
     * Uploads a given scheme on the available devices.
     *
     * Currently, the scheme upload is performed on the Us4R:0 device only.
     *
     * After uploading a new sequence the previously returned output buffers will be in invalid state.
     *
     * @param scheme scheme to upload
     * @return upload result information
     */
    virtual UploadResult upload(const ::arrus::ops::us4r::Scheme &scheme) = 0;

    /**
     * Starts currently uploaded scheme.
     */
    virtual void startScheme() = 0;

    /**
     * Stops currently uploaded scheme.
     */
    virtual void stopScheme() = 0;

    /**
     * Runs the uploaded scheme.
     *
     * The behaviour of this method depends on the work mode:
     * - MANUAL: triggers execution of batch of sequences only ONCE,
     * - MANUAL_OP: triggers execution of a single TX/RX only ONCE,
     * - HOST, ASYNC: triggers execution of batch of sequences IN A LOOP (Host: trigger is on buffer element release).
     *   The run function can be called only once (before the scheme is stopped).
     *
     * @param sync whether this method should work in a synchronous or asynchronous; true means synchronous, i.e.
     *        the caller will wait until the triggered TX/RX or sequence of TX/RXs has been done. The sync = true is only
     *        allowed when the work mode is set to MANUAL or MANUAL_OP. NOTE: For the US4R device, this method ONLY waits
     *        for the completion of the TX/RX sequence. Currently, it DOES NOT WAIT for the data transfer to the host PC
     *        or for the processing to finish — to wait for these two events, either wait for the final data using
     *        buffer.get(), or register your own callback function.
     * @param timeout timeout [ms]; std::nullopt means to wait infinitely. This parameter is only relevant when
     *        sync = true; the value of this parameter only matters when work mode is set to MANUAL or MANUAL_OP
     */
    virtual void run(bool sync = false, std::optional<long long> timeout = std::nullopt) = 0;

    /**
     * Closes session.
     *
     * This method disconnects with all the devices available during this session.
     * Sets the state of the session to closed, any subsequent call to the object methods (e.g. upload, startScheme..)
     * will result in InvalidStateException.
     */
    virtual void close() = 0;

    virtual void setParameters(const arrus::Parameters& params) = 0;

    /**
     * Returns the current state of the session. See also Session::State.
     */
    virtual State getCurrentState() = 0;

    /**
     * Turns on the sequence with the arrayId and sets the TX/RXs to the [start, end) range. This method turns off all
     * the uploaded TX/RX sequences except sequence pointed by `arrayId`.
     *
     * This method requires that:
     *
     * - start < end (start == end would mean that the given sequence should bet turned off, and that would mean that all TX/RXs sequences should be turned off, which current does not make sense),
     * - the scheme was uploaded,
     * - the TX/RX sequence length is greater than the `end` value,
     * - the scheme is stopped.
     *
     * @param start the TX/RX number which should now be the first TX/RX
     * @param end the TX/RX number which should now be the last TX/RX
     * @param sri the new SRI to apply
     * @param arrayId id array to select, default: array with id 0
     * @return the new data buffer and metadata
     */
    virtual UploadResult setSubsequence(uint16 start, uint16 end, std::optional<float> sri, uint16 arrayId) = 0;

    /**
     * Selects [start, end) slices for each sub-sequence.
     *
     * The `slices` array should have exactly n elements, where n is the number of currently uploaded sequences.
     * The element slice[i] sets the [start, end) range for the i-th sequence.
     *
     * The `sris` should have eactly n elements, or should be empty (which means that no additional sri should be
     * applied).
     *
     * To turn off the given sequence, just set start equal to end (e.g. Slice(0, 0)). For such sequences, the metadata
     * will
     *
     * @param slices slices to set to each Scheme sub-sequence
     * @param sris sris to apply to each Scheme sub-sequence
     * @return returns the buffer and metadata for the modified Scheme. The metadata array size is always equal to
     *   the number of seqeuences in the original Scheme
     */
    virtual UploadResult setSubsequences(const std::vector<Slice> &slices, const std::vector<std::optional<float>> &sris) = 0;

    /**
     * Returns true if this session has been configured to work with the given device, otherwise false.
     *
     * @param deviceId device identifier
     */
    virtual bool hasDevice(const std::string &deviceId) const = 0;

    /**
     * Returns true if this session has been configured to work with the given device, otherwise false.
     *
     * @param deviceId device identifier
     */
    virtual bool hasDevice(const arrus::devices::DeviceId &deviceId) const = 0;

    virtual ~Session() = default;

};

/**
* Creates a new session with the provided configuration.
*
* @param sessionSettings session settings to set.
* @return a unique handle to session
*/
ARRUS_CPP_EXPORT
Session::Handle createSession(const SessionSettings &sessionSettings);

/**
* Reads given configuration file and returns a handle to new session.
*
* @param filepath a path to session settings
* @return a unique handle to session
*/
ARRUS_CPP_EXPORT
Session::Handle createSession(const std::string& filepath);
}


#endif //ARRUS_CORE_API_SESSION_SESSION_H
