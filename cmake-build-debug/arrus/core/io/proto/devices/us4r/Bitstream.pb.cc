// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: io/proto/devices/us4r/Bitstream.proto

#include "io/proto/devices/us4r/Bitstream.pb.h"

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
class BitstreamDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Bitstream> _instance;
} _Bitstream_default_instance_;
}  // namespace proto
}  // namespace arrus
static void InitDefaultsscc_info_Bitstream_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::arrus::proto::_Bitstream_default_instance_;
    new (ptr) ::arrus::proto::Bitstream();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::arrus::proto::Bitstream::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_Bitstream_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_Bitstream_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::arrus::proto::Bitstream, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::arrus::proto::Bitstream, levels_),
  PROTOBUF_FIELD_OFFSET(::arrus::proto::Bitstream, periods_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::arrus::proto::Bitstream)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::arrus::proto::_Bitstream_default_instance_),
};

const char descriptor_table_protodef_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n%io/proto/devices/us4r/Bitstream.proto\022"
  "\013arrus.proto\",\n\tBitstream\022\016\n\006levels\030\001 \003("
  "\r\022\017\n\007periods\030\002 \003(\rb\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_sccs[1] = {
  &scc_info_Bitstream_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_once;
static bool descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto = {
  &descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_initialized, descriptor_table_protodef_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto, "io/proto/devices/us4r/Bitstream.proto", 106,
  &descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_once, descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_sccs, descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto::offsets,
  file_level_metadata_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto, 1, file_level_enum_descriptors_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto, file_level_service_descriptors_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto), true);
namespace arrus {
namespace proto {

// ===================================================================

void Bitstream::InitAsDefaultInstance() {
}
class Bitstream::_Internal {
 public:
};

Bitstream::Bitstream()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:arrus.proto.Bitstream)
}
Bitstream::Bitstream(const Bitstream& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      levels_(from.levels_),
      periods_(from.periods_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:arrus.proto.Bitstream)
}

void Bitstream::SharedCtor() {
}

Bitstream::~Bitstream() {
  // @@protoc_insertion_point(destructor:arrus.proto.Bitstream)
  SharedDtor();
}

void Bitstream::SharedDtor() {
}

void Bitstream::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Bitstream& Bitstream::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Bitstream_io_2fproto_2fdevices_2fus4r_2fBitstream_2eproto.base);
  return *internal_default_instance();
}


void Bitstream::Clear() {
// @@protoc_insertion_point(message_clear_start:arrus.proto.Bitstream)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  levels_.Clear();
  periods_.Clear();
  _internal_metadata_.Clear();
}

const char* Bitstream::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated uint32 levels = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedUInt32Parser(_internal_mutable_levels(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8) {
          _internal_add_levels(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated uint32 periods = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedUInt32Parser(_internal_mutable_periods(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16) {
          _internal_add_periods(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr));
          CHK_(ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* Bitstream::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:arrus.proto.Bitstream)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated uint32 levels = 1;
  {
    int byte_size = _levels_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteUInt32Packed(
          1, _internal_levels(), byte_size, target);
    }
  }

  // repeated uint32 periods = 2;
  {
    int byte_size = _periods_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteUInt32Packed(
          2, _internal_periods(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:arrus.proto.Bitstream)
  return target;
}

size_t Bitstream::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:arrus.proto.Bitstream)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated uint32 levels = 1;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      UInt32Size(this->levels_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _levels_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // repeated uint32 periods = 2;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      UInt32Size(this->periods_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _periods_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Bitstream::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:arrus.proto.Bitstream)
  GOOGLE_DCHECK_NE(&from, this);
  const Bitstream* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Bitstream>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:arrus.proto.Bitstream)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:arrus.proto.Bitstream)
    MergeFrom(*source);
  }
}

void Bitstream::MergeFrom(const Bitstream& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:arrus.proto.Bitstream)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  levels_.MergeFrom(from.levels_);
  periods_.MergeFrom(from.periods_);
}

void Bitstream::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:arrus.proto.Bitstream)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Bitstream::CopyFrom(const Bitstream& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:arrus.proto.Bitstream)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Bitstream::IsInitialized() const {
  return true;
}

void Bitstream::InternalSwap(Bitstream* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  levels_.InternalSwap(&other->levels_);
  periods_.InternalSwap(&other->periods_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Bitstream::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace proto
}  // namespace arrus
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::arrus::proto::Bitstream* Arena::CreateMaybeMessage< ::arrus::proto::Bitstream >(Arena* arena) {
  return Arena::CreateInternal< ::arrus::proto::Bitstream >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
