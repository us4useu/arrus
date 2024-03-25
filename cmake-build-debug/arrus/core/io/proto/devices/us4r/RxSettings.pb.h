// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: io/proto/devices/us4r/RxSettings.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3011000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3011004 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "io/proto/common/LinearFunction.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto;
namespace arrus {
namespace proto {
class RxSettings;
class RxSettingsDefaultTypeInternal;
extern RxSettingsDefaultTypeInternal _RxSettings_default_instance_;
}  // namespace proto
}  // namespace arrus
PROTOBUF_NAMESPACE_OPEN
template<> ::arrus::proto::RxSettings* Arena::CreateMaybeMessage<::arrus::proto::RxSettings>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace arrus {
namespace proto {

// ===================================================================

class RxSettings :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:arrus.proto.RxSettings) */ {
 public:
  RxSettings();
  virtual ~RxSettings();

  RxSettings(const RxSettings& from);
  RxSettings(RxSettings&& from) noexcept
    : RxSettings() {
    *this = ::std::move(from);
  }

  inline RxSettings& operator=(const RxSettings& from) {
    CopyFrom(from);
    return *this;
  }
  inline RxSettings& operator=(RxSettings&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const RxSettings& default_instance();

  enum DtgcAttenuationCase {
    kDtgcAttenuation = 2,
    DTGCATTENUATION__NOT_SET = 0,
  };

  enum ActiveTerminationCase {
    kActiveTermination = 9,
    ACTIVETERMINATION__NOT_SET = 0,
  };

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const RxSettings* internal_default_instance() {
    return reinterpret_cast<const RxSettings*>(
               &_RxSettings_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(RxSettings& a, RxSettings& b) {
    a.Swap(&b);
  }
  inline void Swap(RxSettings* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline RxSettings* New() const final {
    return CreateMaybeMessage<RxSettings>(nullptr);
  }

  RxSettings* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<RxSettings>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const RxSettings& from);
  void MergeFrom(const RxSettings& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(RxSettings* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "arrus.proto.RxSettings";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto);
    return ::descriptor_table_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kTgcSamplesFieldNumber = 6,
    kTgcCurveLinearFieldNumber = 5,
    kPgaGainFieldNumber = 3,
    kLnaGainFieldNumber = 4,
    kLpfCutoffFieldNumber = 7,
    kDtgcAttenuationFieldNumber = 2,
    kActiveTerminationFieldNumber = 9,
  };
  // repeated double tgc_samples = 6;
  int tgc_samples_size() const;
  private:
  int _internal_tgc_samples_size() const;
  public:
  void clear_tgc_samples();
  private:
  double _internal_tgc_samples(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      _internal_tgc_samples() const;
  void _internal_add_tgc_samples(double value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      _internal_mutable_tgc_samples();
  public:
  double tgc_samples(int index) const;
  void set_tgc_samples(int index, double value);
  void add_tgc_samples(double value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      tgc_samples() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      mutable_tgc_samples();

  // .arrus.proto.LinearFunction tgc_curve_linear = 5;
  bool has_tgc_curve_linear() const;
  private:
  bool _internal_has_tgc_curve_linear() const;
  public:
  void clear_tgc_curve_linear();
  const ::arrus::proto::LinearFunction& tgc_curve_linear() const;
  ::arrus::proto::LinearFunction* release_tgc_curve_linear();
  ::arrus::proto::LinearFunction* mutable_tgc_curve_linear();
  void set_allocated_tgc_curve_linear(::arrus::proto::LinearFunction* tgc_curve_linear);
  private:
  const ::arrus::proto::LinearFunction& _internal_tgc_curve_linear() const;
  ::arrus::proto::LinearFunction* _internal_mutable_tgc_curve_linear();
  public:

  // uint32 pga_gain = 3;
  void clear_pga_gain();
  ::PROTOBUF_NAMESPACE_ID::uint32 pga_gain() const;
  void set_pga_gain(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_pga_gain() const;
  void _internal_set_pga_gain(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // uint32 lna_gain = 4;
  void clear_lna_gain();
  ::PROTOBUF_NAMESPACE_ID::uint32 lna_gain() const;
  void set_lna_gain(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_lna_gain() const;
  void _internal_set_lna_gain(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // uint32 lpf_cutoff = 7;
  void clear_lpf_cutoff();
  ::PROTOBUF_NAMESPACE_ID::uint32 lpf_cutoff() const;
  void set_lpf_cutoff(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_lpf_cutoff() const;
  void _internal_set_lpf_cutoff(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // uint32 dtgc_attenuation = 2;
  private:
  bool _internal_has_dtgc_attenuation() const;
  public:
  void clear_dtgc_attenuation();
  ::PROTOBUF_NAMESPACE_ID::uint32 dtgc_attenuation() const;
  void set_dtgc_attenuation(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_dtgc_attenuation() const;
  void _internal_set_dtgc_attenuation(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // uint32 active_termination = 9;
  private:
  bool _internal_has_active_termination() const;
  public:
  void clear_active_termination();
  ::PROTOBUF_NAMESPACE_ID::uint32 active_termination() const;
  void set_active_termination(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_active_termination() const;
  void _internal_set_active_termination(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  void clear_dtgcAttenuation_();
  DtgcAttenuationCase dtgcAttenuation__case() const;
  void clear_activeTermination_();
  ActiveTerminationCase activeTermination__case() const;
  // @@protoc_insertion_point(class_scope:arrus.proto.RxSettings)
 private:
  class _Internal;
  void set_has_dtgc_attenuation();
  void set_has_active_termination();

  inline bool has_dtgcAttenuation_() const;
  inline void clear_has_dtgcAttenuation_();

  inline bool has_activeTermination_() const;
  inline void clear_has_activeTermination_();

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double > tgc_samples_;
  mutable std::atomic<int> _tgc_samples_cached_byte_size_;
  ::arrus::proto::LinearFunction* tgc_curve_linear_;
  ::PROTOBUF_NAMESPACE_ID::uint32 pga_gain_;
  ::PROTOBUF_NAMESPACE_ID::uint32 lna_gain_;
  ::PROTOBUF_NAMESPACE_ID::uint32 lpf_cutoff_;
  union DtgcAttenuationUnion {
    DtgcAttenuationUnion() {}
    ::PROTOBUF_NAMESPACE_ID::uint32 dtgc_attenuation_;
  } dtgcAttenuation__;
  union ActiveTerminationUnion {
    ActiveTerminationUnion() {}
    ::PROTOBUF_NAMESPACE_ID::uint32 active_termination_;
  } activeTermination__;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::uint32 _oneof_case_[2];

  friend struct ::TableStruct_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// RxSettings

// uint32 dtgc_attenuation = 2;
inline bool RxSettings::_internal_has_dtgc_attenuation() const {
  return dtgcAttenuation__case() == kDtgcAttenuation;
}
inline void RxSettings::set_has_dtgc_attenuation() {
  _oneof_case_[0] = kDtgcAttenuation;
}
inline void RxSettings::clear_dtgc_attenuation() {
  if (_internal_has_dtgc_attenuation()) {
    dtgcAttenuation__.dtgc_attenuation_ = 0u;
    clear_has_dtgcAttenuation_();
  }
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::_internal_dtgc_attenuation() const {
  if (_internal_has_dtgc_attenuation()) {
    return dtgcAttenuation__.dtgc_attenuation_;
  }
  return 0u;
}
inline void RxSettings::_internal_set_dtgc_attenuation(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  if (!_internal_has_dtgc_attenuation()) {
    clear_dtgcAttenuation_();
    set_has_dtgc_attenuation();
  }
  dtgcAttenuation__.dtgc_attenuation_ = value;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::dtgc_attenuation() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.dtgc_attenuation)
  return _internal_dtgc_attenuation();
}
inline void RxSettings::set_dtgc_attenuation(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_dtgc_attenuation(value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.dtgc_attenuation)
}

// uint32 pga_gain = 3;
inline void RxSettings::clear_pga_gain() {
  pga_gain_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::_internal_pga_gain() const {
  return pga_gain_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::pga_gain() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.pga_gain)
  return _internal_pga_gain();
}
inline void RxSettings::_internal_set_pga_gain(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  pga_gain_ = value;
}
inline void RxSettings::set_pga_gain(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_pga_gain(value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.pga_gain)
}

// uint32 lna_gain = 4;
inline void RxSettings::clear_lna_gain() {
  lna_gain_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::_internal_lna_gain() const {
  return lna_gain_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::lna_gain() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.lna_gain)
  return _internal_lna_gain();
}
inline void RxSettings::_internal_set_lna_gain(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  lna_gain_ = value;
}
inline void RxSettings::set_lna_gain(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_lna_gain(value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.lna_gain)
}

// .arrus.proto.LinearFunction tgc_curve_linear = 5;
inline bool RxSettings::_internal_has_tgc_curve_linear() const {
  return this != internal_default_instance() && tgc_curve_linear_ != nullptr;
}
inline bool RxSettings::has_tgc_curve_linear() const {
  return _internal_has_tgc_curve_linear();
}
inline const ::arrus::proto::LinearFunction& RxSettings::_internal_tgc_curve_linear() const {
  const ::arrus::proto::LinearFunction* p = tgc_curve_linear_;
  return p != nullptr ? *p : *reinterpret_cast<const ::arrus::proto::LinearFunction*>(
      &::arrus::proto::_LinearFunction_default_instance_);
}
inline const ::arrus::proto::LinearFunction& RxSettings::tgc_curve_linear() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.tgc_curve_linear)
  return _internal_tgc_curve_linear();
}
inline ::arrus::proto::LinearFunction* RxSettings::release_tgc_curve_linear() {
  // @@protoc_insertion_point(field_release:arrus.proto.RxSettings.tgc_curve_linear)
  
  ::arrus::proto::LinearFunction* temp = tgc_curve_linear_;
  tgc_curve_linear_ = nullptr;
  return temp;
}
inline ::arrus::proto::LinearFunction* RxSettings::_internal_mutable_tgc_curve_linear() {
  
  if (tgc_curve_linear_ == nullptr) {
    auto* p = CreateMaybeMessage<::arrus::proto::LinearFunction>(GetArenaNoVirtual());
    tgc_curve_linear_ = p;
  }
  return tgc_curve_linear_;
}
inline ::arrus::proto::LinearFunction* RxSettings::mutable_tgc_curve_linear() {
  // @@protoc_insertion_point(field_mutable:arrus.proto.RxSettings.tgc_curve_linear)
  return _internal_mutable_tgc_curve_linear();
}
inline void RxSettings::set_allocated_tgc_curve_linear(::arrus::proto::LinearFunction* tgc_curve_linear) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(tgc_curve_linear_);
  }
  if (tgc_curve_linear) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      tgc_curve_linear = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, tgc_curve_linear, submessage_arena);
    }
    
  } else {
    
  }
  tgc_curve_linear_ = tgc_curve_linear;
  // @@protoc_insertion_point(field_set_allocated:arrus.proto.RxSettings.tgc_curve_linear)
}

// repeated double tgc_samples = 6;
inline int RxSettings::_internal_tgc_samples_size() const {
  return tgc_samples_.size();
}
inline int RxSettings::tgc_samples_size() const {
  return _internal_tgc_samples_size();
}
inline void RxSettings::clear_tgc_samples() {
  tgc_samples_.Clear();
}
inline double RxSettings::_internal_tgc_samples(int index) const {
  return tgc_samples_.Get(index);
}
inline double RxSettings::tgc_samples(int index) const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.tgc_samples)
  return _internal_tgc_samples(index);
}
inline void RxSettings::set_tgc_samples(int index, double value) {
  tgc_samples_.Set(index, value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.tgc_samples)
}
inline void RxSettings::_internal_add_tgc_samples(double value) {
  tgc_samples_.Add(value);
}
inline void RxSettings::add_tgc_samples(double value) {
  _internal_add_tgc_samples(value);
  // @@protoc_insertion_point(field_add:arrus.proto.RxSettings.tgc_samples)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
RxSettings::_internal_tgc_samples() const {
  return tgc_samples_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
RxSettings::tgc_samples() const {
  // @@protoc_insertion_point(field_list:arrus.proto.RxSettings.tgc_samples)
  return _internal_tgc_samples();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
RxSettings::_internal_mutable_tgc_samples() {
  return &tgc_samples_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
RxSettings::mutable_tgc_samples() {
  // @@protoc_insertion_point(field_mutable_list:arrus.proto.RxSettings.tgc_samples)
  return _internal_mutable_tgc_samples();
}

// uint32 lpf_cutoff = 7;
inline void RxSettings::clear_lpf_cutoff() {
  lpf_cutoff_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::_internal_lpf_cutoff() const {
  return lpf_cutoff_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::lpf_cutoff() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.lpf_cutoff)
  return _internal_lpf_cutoff();
}
inline void RxSettings::_internal_set_lpf_cutoff(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  lpf_cutoff_ = value;
}
inline void RxSettings::set_lpf_cutoff(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_lpf_cutoff(value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.lpf_cutoff)
}

// uint32 active_termination = 9;
inline bool RxSettings::_internal_has_active_termination() const {
  return activeTermination__case() == kActiveTermination;
}
inline void RxSettings::set_has_active_termination() {
  _oneof_case_[1] = kActiveTermination;
}
inline void RxSettings::clear_active_termination() {
  if (_internal_has_active_termination()) {
    activeTermination__.active_termination_ = 0u;
    clear_has_activeTermination_();
  }
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::_internal_active_termination() const {
  if (_internal_has_active_termination()) {
    return activeTermination__.active_termination_;
  }
  return 0u;
}
inline void RxSettings::_internal_set_active_termination(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  if (!_internal_has_active_termination()) {
    clear_activeTermination_();
    set_has_active_termination();
  }
  activeTermination__.active_termination_ = value;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 RxSettings::active_termination() const {
  // @@protoc_insertion_point(field_get:arrus.proto.RxSettings.active_termination)
  return _internal_active_termination();
}
inline void RxSettings::set_active_termination(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_active_termination(value);
  // @@protoc_insertion_point(field_set:arrus.proto.RxSettings.active_termination)
}

inline bool RxSettings::has_dtgcAttenuation_() const {
  return dtgcAttenuation__case() != DTGCATTENUATION__NOT_SET;
}
inline void RxSettings::clear_has_dtgcAttenuation_() {
  _oneof_case_[0] = DTGCATTENUATION__NOT_SET;
}
inline bool RxSettings::has_activeTermination_() const {
  return activeTermination__case() != ACTIVETERMINATION__NOT_SET;
}
inline void RxSettings::clear_has_activeTermination_() {
  _oneof_case_[1] = ACTIVETERMINATION__NOT_SET;
}
inline RxSettings::DtgcAttenuationCase RxSettings::dtgcAttenuation__case() const {
  return RxSettings::DtgcAttenuationCase(_oneof_case_[0]);
}
inline RxSettings::ActiveTerminationCase RxSettings::activeTermination__case() const {
  return RxSettings::ActiveTerminationCase(_oneof_case_[1]);
}
#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace proto
}  // namespace arrus

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2fus4r_2fRxSettings_2eproto
