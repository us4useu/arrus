// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: io/proto/common/IntervalDouble.proto

#include "io/proto/common/IntervalDouble.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace arrus {
namespace proto {
class IntervalDoubleDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<IntervalDouble> _instance;
} _IntervalDouble_default_instance_;
}  // namespace proto
}  // namespace arrus
static void InitDefaultsscc_info_IntervalDouble_io_2fproto_2fcommon_2fIntervalDouble_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::arrus::proto::_IntervalDouble_default_instance_;
    new (ptr) ::arrus::proto::IntervalDouble();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::arrus::proto::IntervalDouble::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_IntervalDouble_io_2fproto_2fcommon_2fIntervalDouble_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_IntervalDouble_io_2fproto_2fcommon_2fIntervalDouble_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_io_2fproto_2fcommon_2fIntervalDouble_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_io_2fproto_2fcommon_2fIntervalDouble_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_io_2fproto_2fcommon_2fIntervalDouble_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_io_2fproto_2fcommon_2fIntervalDouble_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::arrus::proto::IntervalDouble, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::arrus::proto::IntervalDouble, begin_),
  PROTOBUF_FIELD_OFFSET(::arrus::proto::IntervalDouble, end_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::arrus::proto::IntervalDouble)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::arrus::proto::_IntervalDouble_default_instance_),
};

const char descriptor_table_protodef_io_2fproto_2fcommon_2fIntervalDouble_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n$io/proto/common/IntervalDouble.proto\022\013"
  "arrus.proto\",\n\016IntervalDouble\022\r\n\005begin\030\001"
  " \001(\001\022\013\n\003end\030\002 \001(\001b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_sccs[1] = {
  &scc_info_IntervalDouble_io_2fproto_2fcommon_2fIntervalDouble_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_once;
static bool descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto = {
  &descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_initialized, descriptor_table_protodef_io_2fproto_2fcommon_2fIntervalDouble_2eproto, "io/proto/common/IntervalDouble.proto", 105,
  &descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_once, descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_sccs, descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_io_2fproto_2fcommon_2fIntervalDouble_2eproto::offsets,
  file_level_metadata_io_2fproto_2fcommon_2fIntervalDouble_2eproto, 1, file_level_enum_descriptors_io_2fproto_2fcommon_2fIntervalDouble_2eproto, file_level_service_descriptors_io_2fproto_2fcommon_2fIntervalDouble_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_io_2fproto_2fcommon_2fIntervalDouble_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_io_2fproto_2fcommon_2fIntervalDouble_2eproto), true);
namespace arrus {
namespace proto {

// ===================================================================

void IntervalDouble::InitAsDefaultInstance() {
}
class IntervalDouble::_Internal {
 public:
};

IntervalDouble::IntervalDouble()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:arrus.proto.IntervalDouble)
}
IntervalDouble::IntervalDouble(const IntervalDouble& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&begin_, &from.begin_,
    static_cast<size_t>(reinterpret_cast<char*>(&end_) -
    reinterpret_cast<char*>(&begin_)) + sizeof(end_));
  // @@protoc_insertion_point(copy_constructor:arrus.proto.IntervalDouble)
}

void IntervalDouble::SharedCtor() {
  ::memset(&begin_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&end_) -
      reinterpret_cast<char*>(&begin_)) + sizeof(end_));
}

IntervalDouble::~IntervalDouble() {
  // @@protoc_insertion_point(destructor:arrus.proto.IntervalDouble)
  SharedDtor();
}

void IntervalDouble::SharedDtor() {
}

void IntervalDouble::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const IntervalDouble& IntervalDouble::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_IntervalDouble_io_2fproto_2fcommon_2fIntervalDouble_2eproto.base);
  return *internal_default_instance();
}


void IntervalDouble::Clear() {
// @@protoc_insertion_point(message_clear_start:arrus.proto.IntervalDouble)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&begin_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&end_) -
      reinterpret_cast<char*>(&begin_)) + sizeof(end_));
  _internal_metadata_.Clear();
}

const char* IntervalDouble::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // double begin = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 9)) {
          begin_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // double end = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          end_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* IntervalDouble::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:arrus.proto.IntervalDouble)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // double begin = 1;
  if (!(this->begin() <= 0 && this->begin() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(1, this->_internal_begin(), target);
  }

  // double end = 2;
  if (!(this->end() <= 0 && this->end() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_end(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:arrus.proto.IntervalDouble)
  return target;
}

size_t IntervalDouble::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:arrus.proto.IntervalDouble)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // double begin = 1;
  if (!(this->begin() <= 0 && this->begin() >= 0)) {
    total_size += 1 + 8;
  }

  // double end = 2;
  if (!(this->end() <= 0 && this->end() >= 0)) {
    total_size += 1 + 8;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void IntervalDouble::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:arrus.proto.IntervalDouble)
  GOOGLE_DCHECK_NE(&from, this);
  const IntervalDouble* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<IntervalDouble>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:arrus.proto.IntervalDouble)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:arrus.proto.IntervalDouble)
    MergeFrom(*source);
  }
}

void IntervalDouble::MergeFrom(const IntervalDouble& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:arrus.proto.IntervalDouble)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (!(from.begin() <= 0 && from.begin() >= 0)) {
    _internal_set_begin(from._internal_begin());
  }
  if (!(from.end() <= 0 && from.end() >= 0)) {
    _internal_set_end(from._internal_end());
  }
}

void IntervalDouble::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:arrus.proto.IntervalDouble)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void IntervalDouble::CopyFrom(const IntervalDouble& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:arrus.proto.IntervalDouble)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool IntervalDouble::IsInitialized() const {
  return true;
}

void IntervalDouble::InternalSwap(IntervalDouble* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(begin_, other->begin_);
  swap(end_, other->end_);
}

::PROTOBUF_NAMESPACE_ID::Metadata IntervalDouble::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace proto
}  // namespace arrus
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::arrus::proto::IntervalDouble* Arena::CreateMaybeMessage< ::arrus::proto::IntervalDouble >(Arena* arena) {
  return Arena::CreateInternal< ::arrus::proto::IntervalDouble >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
