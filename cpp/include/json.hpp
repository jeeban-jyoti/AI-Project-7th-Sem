// Minimal, dependency-free JSON value with parser and serializer.
// Objects keep insertion order (vector of pairs), which also keeps output stable.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tp {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), b_(b) {}
    Json(int v) : type_(Type::Number), n_(v) {}
    Json(long v) : type_(Type::Number), n_(static_cast<double>(v)) {}
    Json(long long v) : type_(Type::Number), n_(static_cast<double>(v)) {}
    Json(unsigned v) : type_(Type::Number), n_(v) {}
    Json(unsigned long v) : type_(Type::Number), n_(static_cast<double>(v)) {}
    Json(unsigned long long v) : type_(Type::Number), n_(static_cast<double>(v)) {}
    Json(double v) : type_(Type::Number), n_(v) {}
    Json(const char* s) : type_(Type::String), s_(s) {}
    Json(std::string s) : type_(Type::String), s_(std::move(s)) {}

    static Json array() { Json j; j.type_ = Type::Array; return j; }
    static Json object() { Json j; j.type_ = Type::Object; return j; }
    static Json parse(const std::string& text);

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }
    bool isArray() const { return type_ == Type::Array; }
    bool isObject() const { return type_ == Type::Object; }
    bool isBool() const { return type_ == Type::Bool; }

    double num(double def = 0.0) const { return type_ == Type::Number ? n_ : (type_ == Type::Bool ? (b_ ? 1.0 : 0.0) : def); }
    int integer(int def = 0) const { return type_ == Type::Number ? static_cast<int>(n_ < 0 ? n_ - 0.5 : n_ + 0.5) : def; }
    bool boolean(bool def = false) const { return type_ == Type::Bool ? b_ : (type_ == Type::Number ? n_ != 0.0 : def); }
    const std::string& str() const { return s_; }

    // Object access
    bool has(const std::string& key) const;
    const Json& operator[](const std::string& key) const;  // null if missing
    Json& operator[](const std::string& key);              // creates key
    const Json& operator[](const char* key) const { return (*this)[std::string(key)]; }
    Json& operator[](const char* key) { return (*this)[std::string(key)]; }

    double getNum(const std::string& key, double def) const;
    int getInt(const std::string& key, int def) const;
    bool getBool(const std::string& key, bool def) const;
    std::string getStr(const std::string& key, const std::string& def) const;

    // Array access
    std::size_t size() const;
    const Json& at(std::size_t i) const;
    Json& at(std::size_t i);
    void push(Json v);

    const std::vector<Json>& items() const { return a_; }
    const std::vector<std::pair<std::string, Json>>& fields() const { return o_; }

    std::string dump() const;

private:
    void dumpTo(std::string& out) const;

    Type type_ = Type::Null;
    bool b_ = false;
    double n_ = 0.0;
    std::string s_;
    std::vector<Json> a_;
    std::vector<std::pair<std::string, Json>> o_;
};

}  // namespace tp
