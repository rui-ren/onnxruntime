// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_DMLGRAPHDESC_DML_IR_H_
#define FLATBUFFERS_GENERATED_DMLGRAPHDESC_DML_IR_H_

#include "flatbuffers/flatbuffers.h"

#include "OperatorFieldTypes_generated.h"

namespace dml {
namespace ir {

struct ConstantRawData;
struct ConstantRawDataBuilder;

struct ConstantName;
struct ConstantNameBuilder;

struct ConstantNodeDesc;
struct ConstantNodeDescBuilder;

struct DmlBufferTensorDesc;
struct DmlBufferTensorDescBuilder;

struct OperatorNodeDesc;
struct OperatorNodeDescBuilder;

struct DmlGraphNode;
struct DmlGraphNodeBuilder;

struct DmlGraphDesc;
struct DmlGraphDescBuilder;

enum ConstantNodeDescDetail {
  ConstantNodeDescDetail_NONE = 0,
  ConstantNodeDescDetail_ConstantName = 1,
  ConstantNodeDescDetail_ConstantRawData = 2,
  ConstantNodeDescDetail_MIN = ConstantNodeDescDetail_NONE,
  ConstantNodeDescDetail_MAX = ConstantNodeDescDetail_ConstantRawData
};

inline const ConstantNodeDescDetail (&EnumValuesConstantNodeDescDetail())[3] {
  static const ConstantNodeDescDetail values[] = {
    ConstantNodeDescDetail_NONE,
    ConstantNodeDescDetail_ConstantName,
    ConstantNodeDescDetail_ConstantRawData
  };
  return values;
}

inline const char * const *EnumNamesConstantNodeDescDetail() {
  static const char * const names[4] = {
    "NONE",
    "ConstantName",
    "ConstantRawData",
    nullptr
  };
  return names;
}

inline const char *EnumNameConstantNodeDescDetail(ConstantNodeDescDetail e) {
  if (flatbuffers::IsOutRange(e, ConstantNodeDescDetail_NONE, ConstantNodeDescDetail_ConstantRawData)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesConstantNodeDescDetail()[index];
}

template<typename T> struct ConstantNodeDescDetailTraits {
  static const ConstantNodeDescDetail enum_value = ConstantNodeDescDetail_NONE;
};

template<> struct ConstantNodeDescDetailTraits<dml::ir::ConstantName> {
  static const ConstantNodeDescDetail enum_value = ConstantNodeDescDetail_ConstantName;
};

template<> struct ConstantNodeDescDetailTraits<dml::ir::ConstantRawData> {
  static const ConstantNodeDescDetail enum_value = ConstantNodeDescDetail_ConstantRawData;
};

bool VerifyConstantNodeDescDetail(flatbuffers::Verifier &verifier, const void *obj, ConstantNodeDescDetail type);
bool VerifyConstantNodeDescDetailVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types);

enum NodeDesc {
  NodeDesc_NONE = 0,
  NodeDesc_OperatorNodeDesc = 1,
  NodeDesc_ConstantNodeDesc = 2,
  NodeDesc_MIN = NodeDesc_NONE,
  NodeDesc_MAX = NodeDesc_ConstantNodeDesc
};

inline const NodeDesc (&EnumValuesNodeDesc())[3] {
  static const NodeDesc values[] = {
    NodeDesc_NONE,
    NodeDesc_OperatorNodeDesc,
    NodeDesc_ConstantNodeDesc
  };
  return values;
}

inline const char * const *EnumNamesNodeDesc() {
  static const char * const names[4] = {
    "NONE",
    "OperatorNodeDesc",
    "ConstantNodeDesc",
    nullptr
  };
  return names;
}

inline const char *EnumNameNodeDesc(NodeDesc e) {
  if (flatbuffers::IsOutRange(e, NodeDesc_NONE, NodeDesc_ConstantNodeDesc)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesNodeDesc()[index];
}

template<typename T> struct NodeDescTraits {
  static const NodeDesc enum_value = NodeDesc_NONE;
};

template<> struct NodeDescTraits<dml::ir::OperatorNodeDesc> {
  static const NodeDesc enum_value = NodeDesc_OperatorNodeDesc;
};

template<> struct NodeDescTraits<dml::ir::ConstantNodeDesc> {
  static const NodeDesc enum_value = NodeDesc_ConstantNodeDesc;
};

bool VerifyNodeDesc(flatbuffers::Verifier &verifier, const void *obj, NodeDesc type);
bool VerifyNodeDescVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types);

struct ConstantRawData FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConstantRawDataBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATA = 4
  };
  const flatbuffers::Vector<uint8_t> *data() const {
    return GetPointer<const flatbuffers::Vector<uint8_t> *>(VT_DATA);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_DATA) &&
           verifier.VerifyVector(data()) &&
           verifier.EndTable();
  }
};

struct ConstantRawDataBuilder {
  typedef ConstantRawData Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_data(flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data) {
    fbb_.AddOffset(ConstantRawData::VT_DATA, data);
  }
  explicit ConstantRawDataBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ConstantRawDataBuilder &operator=(const ConstantRawDataBuilder &);
  flatbuffers::Offset<ConstantRawData> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConstantRawData>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConstantRawData> CreateConstantRawData(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data = 0) {
  ConstantRawDataBuilder builder_(_fbb);
  builder_.add_data(data);
  return builder_.Finish();
}

inline flatbuffers::Offset<ConstantRawData> CreateConstantRawDataDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<uint8_t> *data = nullptr) {
  auto data__ = data ? _fbb.CreateVector<uint8_t>(*data) : 0;
  return dml::ir::CreateConstantRawData(
      _fbb,
      data__);
}

struct ConstantName FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConstantNameBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NAME = 4
  };
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           verifier.EndTable();
  }
};

struct ConstantNameBuilder {
  typedef ConstantName Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(ConstantName::VT_NAME, name);
  }
  explicit ConstantNameBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ConstantNameBuilder &operator=(const ConstantNameBuilder &);
  flatbuffers::Offset<ConstantName> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConstantName>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConstantName> CreateConstantName(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> name = 0) {
  ConstantNameBuilder builder_(_fbb);
  builder_.add_name(name);
  return builder_.Finish();
}

inline flatbuffers::Offset<ConstantName> CreateConstantNameDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *name = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  return dml::ir::CreateConstantName(
      _fbb,
      name__);
}

struct ConstantNodeDesc FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ConstantNodeDescBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATA_TYPE = 4,
    VT_DATA = 6
  };
  dml::ir::ConstantNodeDescDetail data_type() const {
    return static_cast<dml::ir::ConstantNodeDescDetail>(GetField<uint8_t>(VT_DATA_TYPE, 0));
  }
  const void *data() const {
    return GetPointer<const void *>(VT_DATA);
  }
  template<typename T> const T *data_as() const;
  const dml::ir::ConstantName *data_as_ConstantName() const {
    return data_type() == dml::ir::ConstantNodeDescDetail_ConstantName ? static_cast<const dml::ir::ConstantName *>(data()) : nullptr;
  }
  const dml::ir::ConstantRawData *data_as_ConstantRawData() const {
    return data_type() == dml::ir::ConstantNodeDescDetail_ConstantRawData ? static_cast<const dml::ir::ConstantRawData *>(data()) : nullptr;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_DATA_TYPE) &&
           VerifyOffset(verifier, VT_DATA) &&
           VerifyConstantNodeDescDetail(verifier, data(), data_type()) &&
           verifier.EndTable();
  }
};

template<> inline const dml::ir::ConstantName *ConstantNodeDesc::data_as<dml::ir::ConstantName>() const {
  return data_as_ConstantName();
}

template<> inline const dml::ir::ConstantRawData *ConstantNodeDesc::data_as<dml::ir::ConstantRawData>() const {
  return data_as_ConstantRawData();
}

struct ConstantNodeDescBuilder {
  typedef ConstantNodeDesc Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_data_type(dml::ir::ConstantNodeDescDetail data_type) {
    fbb_.AddElement<uint8_t>(ConstantNodeDesc::VT_DATA_TYPE, static_cast<uint8_t>(data_type), 0);
  }
  void add_data(flatbuffers::Offset<void> data) {
    fbb_.AddOffset(ConstantNodeDesc::VT_DATA, data);
  }
  explicit ConstantNodeDescBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ConstantNodeDescBuilder &operator=(const ConstantNodeDescBuilder &);
  flatbuffers::Offset<ConstantNodeDesc> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ConstantNodeDesc>(end);
    return o;
  }
};

inline flatbuffers::Offset<ConstantNodeDesc> CreateConstantNodeDesc(
    flatbuffers::FlatBufferBuilder &_fbb,
    dml::ir::ConstantNodeDescDetail data_type = dml::ir::ConstantNodeDescDetail_NONE,
    flatbuffers::Offset<void> data = 0) {
  ConstantNodeDescBuilder builder_(_fbb);
  builder_.add_data(data);
  builder_.add_data_type(data_type);
  return builder_.Finish();
}

struct DmlBufferTensorDesc FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DmlBufferTensorDescBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DATATYPE = 4,
    VT_SIZES = 6,
    VT_STRIDES = 8,
    VT_TOTALTENSORSIZEINBYTES = 10
  };
  const flatbuffers::String *dataType() const {
    return GetPointer<const flatbuffers::String *>(VT_DATATYPE);
  }
  const flatbuffers::Vector<uint32_t> *sizes() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_SIZES);
  }
  const flatbuffers::Vector<uint32_t> *strides() const {
    return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_STRIDES);
  }
  uint64_t totalTensorSizeInBytes() const {
    return GetField<uint64_t>(VT_TOTALTENSORSIZEINBYTES, 0);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_DATATYPE) &&
           verifier.VerifyString(dataType()) &&
           VerifyOffset(verifier, VT_SIZES) &&
           verifier.VerifyVector(sizes()) &&
           VerifyOffset(verifier, VT_STRIDES) &&
           verifier.VerifyVector(strides()) &&
           VerifyField<uint64_t>(verifier, VT_TOTALTENSORSIZEINBYTES) &&
           verifier.EndTable();
  }
};

struct DmlBufferTensorDescBuilder {
  typedef DmlBufferTensorDesc Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_dataType(flatbuffers::Offset<flatbuffers::String> dataType) {
    fbb_.AddOffset(DmlBufferTensorDesc::VT_DATATYPE, dataType);
  }
  void add_sizes(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> sizes) {
    fbb_.AddOffset(DmlBufferTensorDesc::VT_SIZES, sizes);
  }
  void add_strides(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> strides) {
    fbb_.AddOffset(DmlBufferTensorDesc::VT_STRIDES, strides);
  }
  void add_totalTensorSizeInBytes(uint64_t totalTensorSizeInBytes) {
    fbb_.AddElement<uint64_t>(DmlBufferTensorDesc::VT_TOTALTENSORSIZEINBYTES, totalTensorSizeInBytes, 0);
  }
  explicit DmlBufferTensorDescBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DmlBufferTensorDescBuilder &operator=(const DmlBufferTensorDescBuilder &);
  flatbuffers::Offset<DmlBufferTensorDesc> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DmlBufferTensorDesc>(end);
    return o;
  }
};

inline flatbuffers::Offset<DmlBufferTensorDesc> CreateDmlBufferTensorDesc(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> dataType = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> sizes = 0,
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> strides = 0,
    uint64_t totalTensorSizeInBytes = 0) {
  DmlBufferTensorDescBuilder builder_(_fbb);
  builder_.add_totalTensorSizeInBytes(totalTensorSizeInBytes);
  builder_.add_strides(strides);
  builder_.add_sizes(sizes);
  builder_.add_dataType(dataType);
  return builder_.Finish();
}

inline flatbuffers::Offset<DmlBufferTensorDesc> CreateDmlBufferTensorDescDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *dataType = nullptr,
    const std::vector<uint32_t> *sizes = nullptr,
    const std::vector<uint32_t> *strides = nullptr,
    uint64_t totalTensorSizeInBytes = 0) {
  auto dataType__ = dataType ? _fbb.CreateString(dataType) : 0;
  auto sizes__ = sizes ? _fbb.CreateVector<uint32_t>(*sizes) : 0;
  auto strides__ = strides ? _fbb.CreateVector<uint32_t>(*strides) : 0;
  return dml::ir::CreateDmlBufferTensorDesc(
      _fbb,
      dataType__,
      sizes__,
      strides__,
      totalTensorSizeInBytes);
}

struct OperatorNodeDesc FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef OperatorNodeDescBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_TYPE = 4,
    VT_INPUTS = 6,
    VT_OUTPUTS = 8,
    VT_ATTRIBUTES = 10
  };
  const flatbuffers::String *type() const {
    return GetPointer<const flatbuffers::String *>(VT_TYPE);
  }
  const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *inputs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *>(VT_INPUTS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *outputs() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *>(VT_OUTPUTS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> *attributes() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> *>(VT_ATTRIBUTES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_TYPE) &&
           verifier.VerifyString(type()) &&
           VerifyOffset(verifier, VT_INPUTS) &&
           verifier.VerifyVector(inputs()) &&
           verifier.VerifyVectorOfTables(inputs()) &&
           VerifyOffset(verifier, VT_OUTPUTS) &&
           verifier.VerifyVector(outputs()) &&
           verifier.VerifyVectorOfTables(outputs()) &&
           VerifyOffset(verifier, VT_ATTRIBUTES) &&
           verifier.VerifyVector(attributes()) &&
           verifier.VerifyVectorOfTables(attributes()) &&
           verifier.EndTable();
  }
};

struct OperatorNodeDescBuilder {
  typedef OperatorNodeDesc Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_type(flatbuffers::Offset<flatbuffers::String> type) {
    fbb_.AddOffset(OperatorNodeDesc::VT_TYPE, type);
  }
  void add_inputs(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>> inputs) {
    fbb_.AddOffset(OperatorNodeDesc::VT_INPUTS, inputs);
  }
  void add_outputs(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>> outputs) {
    fbb_.AddOffset(OperatorNodeDesc::VT_OUTPUTS, outputs);
  }
  void add_attributes(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>> attributes) {
    fbb_.AddOffset(OperatorNodeDesc::VT_ATTRIBUTES, attributes);
  }
  explicit OperatorNodeDescBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  OperatorNodeDescBuilder &operator=(const OperatorNodeDescBuilder &);
  flatbuffers::Offset<OperatorNodeDesc> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<OperatorNodeDesc>(end);
    return o;
  }
};

inline flatbuffers::Offset<OperatorNodeDesc> CreateOperatorNodeDesc(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> type = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>> inputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>> outputs = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>> attributes = 0) {
  OperatorNodeDescBuilder builder_(_fbb);
  builder_.add_attributes(attributes);
  builder_.add_outputs(outputs);
  builder_.add_inputs(inputs);
  builder_.add_type(type);
  return builder_.Finish();
}

inline flatbuffers::Offset<OperatorNodeDesc> CreateOperatorNodeDescDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *type = nullptr,
    const std::vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *inputs = nullptr,
    const std::vector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>> *outputs = nullptr,
    const std::vector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>> *attributes = nullptr) {
  auto type__ = type ? _fbb.CreateString(type) : 0;
  auto inputs__ = inputs ? _fbb.CreateVector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>(*inputs) : 0;
  auto outputs__ = outputs ? _fbb.CreateVector<flatbuffers::Offset<dml::ir::DmlBufferTensorDesc>>(*outputs) : 0;
  auto attributes__ = attributes ? _fbb.CreateVector<flatbuffers::Offset<dml::ir::operatorFieldTypes::AttributeDesc>>(*attributes) : 0;
  return dml::ir::CreateOperatorNodeDesc(
      _fbb,
      type__,
      inputs__,
      outputs__,
      attributes__);
}

struct DmlGraphNode FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DmlGraphNodeBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_DESC_TYPE = 4,
    VT_DESC = 6,
    VT_NAME = 8,
    VT_INPUTNAMES = 10,
    VT_OUTPUTNAMES = 12
  };
  dml::ir::NodeDesc desc_type() const {
    return static_cast<dml::ir::NodeDesc>(GetField<uint8_t>(VT_DESC_TYPE, 0));
  }
  const void *desc() const {
    return GetPointer<const void *>(VT_DESC);
  }
  template<typename T> const T *desc_as() const;
  const dml::ir::OperatorNodeDesc *desc_as_OperatorNodeDesc() const {
    return desc_type() == dml::ir::NodeDesc_OperatorNodeDesc ? static_cast<const dml::ir::OperatorNodeDesc *>(desc()) : nullptr;
  }
  const dml::ir::ConstantNodeDesc *desc_as_ConstantNodeDesc() const {
    return desc_type() == dml::ir::NodeDesc_ConstantNodeDesc ? static_cast<const dml::ir::ConstantNodeDesc *>(desc()) : nullptr;
  }
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *inputNames() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_INPUTNAMES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *outputNames() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_OUTPUTNAMES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<uint8_t>(verifier, VT_DESC_TYPE) &&
           VerifyOffset(verifier, VT_DESC) &&
           VerifyNodeDesc(verifier, desc(), desc_type()) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyOffset(verifier, VT_INPUTNAMES) &&
           verifier.VerifyVector(inputNames()) &&
           verifier.VerifyVectorOfStrings(inputNames()) &&
           VerifyOffset(verifier, VT_OUTPUTNAMES) &&
           verifier.VerifyVector(outputNames()) &&
           verifier.VerifyVectorOfStrings(outputNames()) &&
           verifier.EndTable();
  }
};

template<> inline const dml::ir::OperatorNodeDesc *DmlGraphNode::desc_as<dml::ir::OperatorNodeDesc>() const {
  return desc_as_OperatorNodeDesc();
}

template<> inline const dml::ir::ConstantNodeDesc *DmlGraphNode::desc_as<dml::ir::ConstantNodeDesc>() const {
  return desc_as_ConstantNodeDesc();
}

struct DmlGraphNodeBuilder {
  typedef DmlGraphNode Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_desc_type(dml::ir::NodeDesc desc_type) {
    fbb_.AddElement<uint8_t>(DmlGraphNode::VT_DESC_TYPE, static_cast<uint8_t>(desc_type), 0);
  }
  void add_desc(flatbuffers::Offset<void> desc) {
    fbb_.AddOffset(DmlGraphNode::VT_DESC, desc);
  }
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(DmlGraphNode::VT_NAME, name);
  }
  void add_inputNames(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> inputNames) {
    fbb_.AddOffset(DmlGraphNode::VT_INPUTNAMES, inputNames);
  }
  void add_outputNames(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> outputNames) {
    fbb_.AddOffset(DmlGraphNode::VT_OUTPUTNAMES, outputNames);
  }
  explicit DmlGraphNodeBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DmlGraphNodeBuilder &operator=(const DmlGraphNodeBuilder &);
  flatbuffers::Offset<DmlGraphNode> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DmlGraphNode>(end);
    return o;
  }
};

inline flatbuffers::Offset<DmlGraphNode> CreateDmlGraphNode(
    flatbuffers::FlatBufferBuilder &_fbb,
    dml::ir::NodeDesc desc_type = dml::ir::NodeDesc_NONE,
    flatbuffers::Offset<void> desc = 0,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> inputNames = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> outputNames = 0) {
  DmlGraphNodeBuilder builder_(_fbb);
  builder_.add_outputNames(outputNames);
  builder_.add_inputNames(inputNames);
  builder_.add_name(name);
  builder_.add_desc(desc);
  builder_.add_desc_type(desc_type);
  return builder_.Finish();
}

inline flatbuffers::Offset<DmlGraphNode> CreateDmlGraphNodeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    dml::ir::NodeDesc desc_type = dml::ir::NodeDesc_NONE,
    flatbuffers::Offset<void> desc = 0,
    const char *name = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *inputNames = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *outputNames = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto inputNames__ = inputNames ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*inputNames) : 0;
  auto outputNames__ = outputNames ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*outputNames) : 0;
  return dml::ir::CreateDmlGraphNode(
      _fbb,
      desc_type,
      desc,
      name__,
      inputNames__,
      outputNames__);
}

struct DmlGraphDesc FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef DmlGraphDescBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_NODES = 4,
    VT_GRAPHINPUTNAMES = 6,
    VT_GRAPHOUTPUTNAMES = 8
  };
  const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlGraphNode>> *nodes() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlGraphNode>> *>(VT_NODES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *graphInputNames() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_GRAPHINPUTNAMES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *graphOutputNames() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_GRAPHOUTPUTNAMES);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_NODES) &&
           verifier.VerifyVector(nodes()) &&
           verifier.VerifyVectorOfTables(nodes()) &&
           VerifyOffset(verifier, VT_GRAPHINPUTNAMES) &&
           verifier.VerifyVector(graphInputNames()) &&
           verifier.VerifyVectorOfStrings(graphInputNames()) &&
           VerifyOffset(verifier, VT_GRAPHOUTPUTNAMES) &&
           verifier.VerifyVector(graphOutputNames()) &&
           verifier.VerifyVectorOfStrings(graphOutputNames()) &&
           verifier.EndTable();
  }
};

struct DmlGraphDescBuilder {
  typedef DmlGraphDesc Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_nodes(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlGraphNode>>> nodes) {
    fbb_.AddOffset(DmlGraphDesc::VT_NODES, nodes);
  }
  void add_graphInputNames(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> graphInputNames) {
    fbb_.AddOffset(DmlGraphDesc::VT_GRAPHINPUTNAMES, graphInputNames);
  }
  void add_graphOutputNames(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> graphOutputNames) {
    fbb_.AddOffset(DmlGraphDesc::VT_GRAPHOUTPUTNAMES, graphOutputNames);
  }
  explicit DmlGraphDescBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  DmlGraphDescBuilder &operator=(const DmlGraphDescBuilder &);
  flatbuffers::Offset<DmlGraphDesc> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<DmlGraphDesc>(end);
    return o;
  }
};

inline flatbuffers::Offset<DmlGraphDesc> CreateDmlGraphDesc(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<dml::ir::DmlGraphNode>>> nodes = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> graphInputNames = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> graphOutputNames = 0) {
  DmlGraphDescBuilder builder_(_fbb);
  builder_.add_graphOutputNames(graphOutputNames);
  builder_.add_graphInputNames(graphInputNames);
  builder_.add_nodes(nodes);
  return builder_.Finish();
}

inline flatbuffers::Offset<DmlGraphDesc> CreateDmlGraphDescDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<flatbuffers::Offset<dml::ir::DmlGraphNode>> *nodes = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *graphInputNames = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *graphOutputNames = nullptr) {
  auto nodes__ = nodes ? _fbb.CreateVector<flatbuffers::Offset<dml::ir::DmlGraphNode>>(*nodes) : 0;
  auto graphInputNames__ = graphInputNames ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*graphInputNames) : 0;
  auto graphOutputNames__ = graphOutputNames ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*graphOutputNames) : 0;
  return dml::ir::CreateDmlGraphDesc(
      _fbb,
      nodes__,
      graphInputNames__,
      graphOutputNames__);
}

inline bool VerifyConstantNodeDescDetail(flatbuffers::Verifier &verifier, const void *obj, ConstantNodeDescDetail type) {
  switch (type) {
    case ConstantNodeDescDetail_NONE: {
      return true;
    }
    case ConstantNodeDescDetail_ConstantName: {
      auto ptr = reinterpret_cast<const dml::ir::ConstantName *>(obj);
      return verifier.VerifyTable(ptr);
    }
    case ConstantNodeDescDetail_ConstantRawData: {
      auto ptr = reinterpret_cast<const dml::ir::ConstantRawData *>(obj);
      return verifier.VerifyTable(ptr);
    }
    default: return true;
  }
}

inline bool VerifyConstantNodeDescDetailVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types) {
  if (!values || !types) return !values && !types;
  if (values->size() != types->size()) return false;
  for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
    if (!VerifyConstantNodeDescDetail(
        verifier,  values->Get(i), types->GetEnum<ConstantNodeDescDetail>(i))) {
      return false;
    }
  }
  return true;
}

inline bool VerifyNodeDesc(flatbuffers::Verifier &verifier, const void *obj, NodeDesc type) {
  switch (type) {
    case NodeDesc_NONE: {
      return true;
    }
    case NodeDesc_OperatorNodeDesc: {
      auto ptr = reinterpret_cast<const dml::ir::OperatorNodeDesc *>(obj);
      return verifier.VerifyTable(ptr);
    }
    case NodeDesc_ConstantNodeDesc: {
      auto ptr = reinterpret_cast<const dml::ir::ConstantNodeDesc *>(obj);
      return verifier.VerifyTable(ptr);
    }
    default: return true;
  }
}

inline bool VerifyNodeDescVector(flatbuffers::Verifier &verifier, const flatbuffers::Vector<flatbuffers::Offset<void>> *values, const flatbuffers::Vector<uint8_t> *types) {
  if (!values || !types) return !values && !types;
  if (values->size() != types->size()) return false;
  for (flatbuffers::uoffset_t i = 0; i < values->size(); ++i) {
    if (!VerifyNodeDesc(
        verifier,  values->Get(i), types->GetEnum<NodeDesc>(i))) {
      return false;
    }
  }
  return true;
}

inline const dml::ir::DmlGraphDesc *GetDmlGraphDesc(const void *buf) {
  return flatbuffers::GetRoot<dml::ir::DmlGraphDesc>(buf);
}

inline const dml::ir::DmlGraphDesc *GetSizePrefixedDmlGraphDesc(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<dml::ir::DmlGraphDesc>(buf);
}

inline bool VerifyDmlGraphDescBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<dml::ir::DmlGraphDesc>(nullptr);
}

inline bool VerifySizePrefixedDmlGraphDescBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<dml::ir::DmlGraphDesc>(nullptr);
}

inline void FinishDmlGraphDescBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<dml::ir::DmlGraphDesc> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedDmlGraphDescBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<dml::ir::DmlGraphDesc> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace ir
}  // namespace dml

#endif  // FLATBUFFERS_GENERATED_DMLGRAPHDESC_DML_IR_H_
