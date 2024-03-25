// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: io/proto/devices/file/FileSettings.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto

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
#include "io/proto/devices/probe/ProbeModel.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto;
namespace arrus {
namespace proto {
class FileSettings;
class FileSettingsDefaultTypeInternal;
extern FileSettingsDefaultTypeInternal _FileSettings_default_instance_;
}  // namespace proto
}  // namespace arrus
PROTOBUF_NAMESPACE_OPEN
template<> ::arrus::proto::FileSettings* Arena::CreateMaybeMessage<::arrus::proto::FileSettings>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace arrus {
namespace proto {

// ===================================================================

class FileSettings :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:arrus.proto.FileSettings) */ {
 public:
  FileSettings();
  virtual ~FileSettings();

  FileSettings(const FileSettings& from);
  FileSettings(FileSettings&& from) noexcept
    : FileSettings() {
    *this = ::std::move(from);
  }

  inline FileSettings& operator=(const FileSettings& from) {
    CopyFrom(from);
    return *this;
  }
  inline FileSettings& operator=(FileSettings&& from) noexcept {
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
  static const FileSettings& default_instance();

  enum OneOfProbeRepresentationCase {
    kProbeId = 3,
    kProbe = 4,
    ONE_OF_PROBE_REPRESENTATION_NOT_SET = 0,
  };

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const FileSettings* internal_default_instance() {
    return reinterpret_cast<const FileSettings*>(
               &_FileSettings_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FileSettings& a, FileSettings& b) {
    a.Swap(&b);
  }
  inline void Swap(FileSettings* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline FileSettings* New() const final {
    return CreateMaybeMessage<FileSettings>(nullptr);
  }

  FileSettings* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<FileSettings>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const FileSettings& from);
  void MergeFrom(const FileSettings& from);
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
  void InternalSwap(FileSettings* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "arrus.proto.FileSettings";
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
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto);
    return ::descriptor_table_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kFilepathFieldNumber = 1,
    kNFramesFieldNumber = 2,
    kProbeIdFieldNumber = 3,
    kProbeFieldNumber = 4,
  };
  // string filepath = 1;
  void clear_filepath();
  const std::string& filepath() const;
  void set_filepath(const std::string& value);
  void set_filepath(std::string&& value);
  void set_filepath(const char* value);
  void set_filepath(const char* value, size_t size);
  std::string* mutable_filepath();
  std::string* release_filepath();
  void set_allocated_filepath(std::string* filepath);
  private:
  const std::string& _internal_filepath() const;
  void _internal_set_filepath(const std::string& value);
  std::string* _internal_mutable_filepath();
  public:

  // uint32 n_frames = 2;
  void clear_n_frames();
  ::PROTOBUF_NAMESPACE_ID::uint32 n_frames() const;
  void set_n_frames(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_n_frames() const;
  void _internal_set_n_frames(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // .arrus.proto.ProbeModel.Id probe_id = 3;
  bool has_probe_id() const;
  private:
  bool _internal_has_probe_id() const;
  public:
  void clear_probe_id();
  const ::arrus::proto::ProbeModel_Id& probe_id() const;
  ::arrus::proto::ProbeModel_Id* release_probe_id();
  ::arrus::proto::ProbeModel_Id* mutable_probe_id();
  void set_allocated_probe_id(::arrus::proto::ProbeModel_Id* probe_id);
  private:
  const ::arrus::proto::ProbeModel_Id& _internal_probe_id() const;
  ::arrus::proto::ProbeModel_Id* _internal_mutable_probe_id();
  public:

  // .arrus.proto.ProbeModel probe = 4;
  bool has_probe() const;
  private:
  bool _internal_has_probe() const;
  public:
  void clear_probe();
  const ::arrus::proto::ProbeModel& probe() const;
  ::arrus::proto::ProbeModel* release_probe();
  ::arrus::proto::ProbeModel* mutable_probe();
  void set_allocated_probe(::arrus::proto::ProbeModel* probe);
  private:
  const ::arrus::proto::ProbeModel& _internal_probe() const;
  ::arrus::proto::ProbeModel* _internal_mutable_probe();
  public:

  void clear_one_of_probe_representation();
  OneOfProbeRepresentationCase one_of_probe_representation_case() const;
  // @@protoc_insertion_point(class_scope:arrus.proto.FileSettings)
 private:
  class _Internal;
  void set_has_probe_id();
  void set_has_probe();

  inline bool has_one_of_probe_representation() const;
  inline void clear_has_one_of_probe_representation();

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr filepath_;
  ::PROTOBUF_NAMESPACE_ID::uint32 n_frames_;
  union OneOfProbeRepresentationUnion {
    OneOfProbeRepresentationUnion() {}
    ::arrus::proto::ProbeModel_Id* probe_id_;
    ::arrus::proto::ProbeModel* probe_;
  } one_of_probe_representation_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::uint32 _oneof_case_[1];

  friend struct ::TableStruct_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FileSettings

// string filepath = 1;
inline void FileSettings::clear_filepath() {
  filepath_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& FileSettings::filepath() const {
  // @@protoc_insertion_point(field_get:arrus.proto.FileSettings.filepath)
  return _internal_filepath();
}
inline void FileSettings::set_filepath(const std::string& value) {
  _internal_set_filepath(value);
  // @@protoc_insertion_point(field_set:arrus.proto.FileSettings.filepath)
}
inline std::string* FileSettings::mutable_filepath() {
  // @@protoc_insertion_point(field_mutable:arrus.proto.FileSettings.filepath)
  return _internal_mutable_filepath();
}
inline const std::string& FileSettings::_internal_filepath() const {
  return filepath_.GetNoArena();
}
inline void FileSettings::_internal_set_filepath(const std::string& value) {
  
  filepath_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
}
inline void FileSettings::set_filepath(std::string&& value) {
  
  filepath_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:arrus.proto.FileSettings.filepath)
}
inline void FileSettings::set_filepath(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  filepath_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:arrus.proto.FileSettings.filepath)
}
inline void FileSettings::set_filepath(const char* value, size_t size) {
  
  filepath_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:arrus.proto.FileSettings.filepath)
}
inline std::string* FileSettings::_internal_mutable_filepath() {
  
  return filepath_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* FileSettings::release_filepath() {
  // @@protoc_insertion_point(field_release:arrus.proto.FileSettings.filepath)
  
  return filepath_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void FileSettings::set_allocated_filepath(std::string* filepath) {
  if (filepath != nullptr) {
    
  } else {
    
  }
  filepath_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), filepath);
  // @@protoc_insertion_point(field_set_allocated:arrus.proto.FileSettings.filepath)
}

// uint32 n_frames = 2;
inline void FileSettings::clear_n_frames() {
  n_frames_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 FileSettings::_internal_n_frames() const {
  return n_frames_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 FileSettings::n_frames() const {
  // @@protoc_insertion_point(field_get:arrus.proto.FileSettings.n_frames)
  return _internal_n_frames();
}
inline void FileSettings::_internal_set_n_frames(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  n_frames_ = value;
}
inline void FileSettings::set_n_frames(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_n_frames(value);
  // @@protoc_insertion_point(field_set:arrus.proto.FileSettings.n_frames)
}

// .arrus.proto.ProbeModel.Id probe_id = 3;
inline bool FileSettings::_internal_has_probe_id() const {
  return one_of_probe_representation_case() == kProbeId;
}
inline bool FileSettings::has_probe_id() const {
  return _internal_has_probe_id();
}
inline void FileSettings::set_has_probe_id() {
  _oneof_case_[0] = kProbeId;
}
inline ::arrus::proto::ProbeModel_Id* FileSettings::release_probe_id() {
  // @@protoc_insertion_point(field_release:arrus.proto.FileSettings.probe_id)
  if (_internal_has_probe_id()) {
    clear_has_one_of_probe_representation();
      ::arrus::proto::ProbeModel_Id* temp = one_of_probe_representation_.probe_id_;
    one_of_probe_representation_.probe_id_ = nullptr;
    return temp;
  } else {
    return nullptr;
  }
}
inline const ::arrus::proto::ProbeModel_Id& FileSettings::_internal_probe_id() const {
  return _internal_has_probe_id()
      ? *one_of_probe_representation_.probe_id_
      : *reinterpret_cast< ::arrus::proto::ProbeModel_Id*>(&::arrus::proto::_ProbeModel_Id_default_instance_);
}
inline const ::arrus::proto::ProbeModel_Id& FileSettings::probe_id() const {
  // @@protoc_insertion_point(field_get:arrus.proto.FileSettings.probe_id)
  return _internal_probe_id();
}
inline ::arrus::proto::ProbeModel_Id* FileSettings::_internal_mutable_probe_id() {
  if (!_internal_has_probe_id()) {
    clear_one_of_probe_representation();
    set_has_probe_id();
    one_of_probe_representation_.probe_id_ = CreateMaybeMessage< ::arrus::proto::ProbeModel_Id >(
        GetArenaNoVirtual());
  }
  return one_of_probe_representation_.probe_id_;
}
inline ::arrus::proto::ProbeModel_Id* FileSettings::mutable_probe_id() {
  // @@protoc_insertion_point(field_mutable:arrus.proto.FileSettings.probe_id)
  return _internal_mutable_probe_id();
}

// .arrus.proto.ProbeModel probe = 4;
inline bool FileSettings::_internal_has_probe() const {
  return one_of_probe_representation_case() == kProbe;
}
inline bool FileSettings::has_probe() const {
  return _internal_has_probe();
}
inline void FileSettings::set_has_probe() {
  _oneof_case_[0] = kProbe;
}
inline ::arrus::proto::ProbeModel* FileSettings::release_probe() {
  // @@protoc_insertion_point(field_release:arrus.proto.FileSettings.probe)
  if (_internal_has_probe()) {
    clear_has_one_of_probe_representation();
      ::arrus::proto::ProbeModel* temp = one_of_probe_representation_.probe_;
    one_of_probe_representation_.probe_ = nullptr;
    return temp;
  } else {
    return nullptr;
  }
}
inline const ::arrus::proto::ProbeModel& FileSettings::_internal_probe() const {
  return _internal_has_probe()
      ? *one_of_probe_representation_.probe_
      : *reinterpret_cast< ::arrus::proto::ProbeModel*>(&::arrus::proto::_ProbeModel_default_instance_);
}
inline const ::arrus::proto::ProbeModel& FileSettings::probe() const {
  // @@protoc_insertion_point(field_get:arrus.proto.FileSettings.probe)
  return _internal_probe();
}
inline ::arrus::proto::ProbeModel* FileSettings::_internal_mutable_probe() {
  if (!_internal_has_probe()) {
    clear_one_of_probe_representation();
    set_has_probe();
    one_of_probe_representation_.probe_ = CreateMaybeMessage< ::arrus::proto::ProbeModel >(
        GetArenaNoVirtual());
  }
  return one_of_probe_representation_.probe_;
}
inline ::arrus::proto::ProbeModel* FileSettings::mutable_probe() {
  // @@protoc_insertion_point(field_mutable:arrus.proto.FileSettings.probe)
  return _internal_mutable_probe();
}

inline bool FileSettings::has_one_of_probe_representation() const {
  return one_of_probe_representation_case() != ONE_OF_PROBE_REPRESENTATION_NOT_SET;
}
inline void FileSettings::clear_has_one_of_probe_representation() {
  _oneof_case_[0] = ONE_OF_PROBE_REPRESENTATION_NOT_SET;
}
inline FileSettings::OneOfProbeRepresentationCase FileSettings::one_of_probe_representation_case() const {
  return FileSettings::OneOfProbeRepresentationCase(_oneof_case_[0]);
}
#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace proto
}  // namespace arrus

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_io_2fproto_2fdevices_2ffile_2fFileSettings_2eproto
