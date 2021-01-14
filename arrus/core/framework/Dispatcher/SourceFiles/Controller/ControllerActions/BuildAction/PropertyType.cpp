#include "Controller/ControllerActions/BuildAction/PropertyType.h"
#include "Utils/DispatcherLogger.h"

PropertyType::PropertyType()
{
	this->initiateRegexes();
	this->innerType = InnerType::STRING;
}

PropertyType::PropertyType(const VariableAnyValue& val)
{
	if (val.getAnyValue().type() == boost::typeindex::type_id<bool>())
	{
		this->innerType = InnerType::BOOL;
	}
	else if (val.getAnyValue().type() == boost::typeindex::type_id<float>())
	{
		this->innerType = InnerType::FLOAT;
	}
	else if (val.getAnyValue().type() == boost::typeindex::type_id<int>())
	{
		this->innerType = InnerType::INT;
	}
	else if (val.getAnyValue().type() == boost::typeindex::type_id<std::string>())
	{
		this->innerType = InnerType::STRING;
	}
	else if (val.getAnyValue().type() == boost::typeindex::type_id<std::unordered_map<std::string, VariableAnyValue>>())
	{
		this->innerType = InnerType::MAP;
	}
	else if (val.getAnyValue().type() == boost::typeindex::type_id<std::vector<VariableAnyValue>>())
	{
		this->innerType = InnerType::ARRAY;
	}
}

PropertyType::PropertyType(const boost::property_tree::ptree& prop)
{
	this->initiateRegexes();
	this->innerType = this->getInnerType(prop);
}

PropertyType::PropertyType(InnerType type)
{
	this->initiateRegexes();
	this->innerType = type;
}

void PropertyType::initiateRegexes()
{
	this->pointer_regex = std::regex("^#([0-9]+)#(.+)$");
	this->int_regex = std::regex("^[+-]?[0-9]+$");
	this->float_regex = std::regex("^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$");
	this->bool_regex = std::regex("^(true)|(false)$");
}

PropertyType::InnerType PropertyType::getInnerType(const boost::property_tree::ptree& prop)
{
	if (std::regex_match(prop.data(), this->bool_regex))
		return InnerType::BOOL;

	if (std::regex_match(prop.data(), this->int_regex))
		return InnerType::INT;

	if (std::regex_match(prop.data(), this->float_regex))
		return InnerType::FLOAT;

	if (std::regex_match(prop.data(), this->pointer_regex))
		return InnerType::POINTER;

	if (prop.data().empty())
	{
		if (prop.size() == 0)
			return InnerType::MAP_OR_ARRAY_OR_STRING;

		if (!prop.front().first.empty())
			return InnerType::MAP;

		return InnerType::ARRAY;
	}

	return InnerType::STRING;
}

std::string PropertyType::toString()
{
	switch (this->innerType) 
	{
		case InnerType::BOOL:
			return "bool";

		case InnerType::ARRAY:
			return "array";

		case InnerType::FLOAT:
			return "float";

		case InnerType::INT:
			return "int";

		case InnerType::MAP:
			return "map";

		case InnerType::POINTER:
			return "pointer";

		case InnerType::MAP_OR_ARRAY_OR_STRING:
			return "map or array or string";

		case InnerType::STRING:
			return "string";
	}
	return "string";
}

bool PropertyType::operator == (const PropertyType& type) const
{
	switch (this->innerType)
	{
		case InnerType::BOOL:
			return ((type.innerType == InnerType::BOOL) || (type.innerType == InnerType::POINTER));

		case InnerType::INT:
			return ((type.innerType == InnerType::INT) || (type.innerType == InnerType::POINTER));

		case InnerType::FLOAT:
			return ((type.innerType == InnerType::FLOAT) || (type.innerType == InnerType::POINTER));

		case InnerType::STRING:
			return ((type.innerType == InnerType::STRING) || (type.innerType == InnerType::POINTER));

		case InnerType::MAP:
			return (type.innerType == InnerType::MAP);

		case InnerType::ARRAY:
			return (type.innerType == InnerType::ARRAY);

		case InnerType::MAP_OR_ARRAY_OR_STRING:
			return ((type.innerType == InnerType::MAP_OR_ARRAY_OR_STRING) || (type.innerType == InnerType::ARRAY)
				|| (type.innerType == InnerType::MAP) || (type.innerType == InnerType::STRING));

		case InnerType::POINTER:
			return ((type.innerType == InnerType::POINTER) || (type.innerType == InnerType::BOOL) || (type.innerType == InnerType::INT)
				|| (type.innerType == InnerType::FLOAT) || (type.innerType == InnerType::STRING));
	}
	return false;
}

bool PropertyType::operator != (const PropertyType& type) const
{
	return !(*this == type);
}