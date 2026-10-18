#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tp {

namespace {

const Json& nullJson() {
    static const Json kNull;
    return kNull;
}

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Json parseDocument() {
        Json v = parseValue();
        skipWs();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("JSON parse error: ") + what + " at offset " + std::to_string(pos_));
    }

    void skipWs() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
    }

    char peek() {
        skipWs();
        if (pos_ >= s_.size()) fail("unexpected end");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++pos_;
    }

    Json parseValue() {
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return Json(parseString());
        if (c == 't') { matchWord("true"); return Json(true); }
        if (c == 'f') { matchWord("false"); return Json(false); }
        if (c == 'n') { matchWord("null"); return Json(); }
        return parseNumber();
    }

    void matchWord(const char* w) {
        std::size_t n = std::char_traits<char>::length(w);
        if (s_.compare(pos_, n, w) != 0) fail("invalid literal");
        pos_ += n;
    }

    Json parseNumber() {
        skipWs();
        std::size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') ++pos_;
            else break;
        }
        if (start == pos_) fail("invalid value");
        std::string tok = s_.substr(start, pos_ - start);
        char* end = nullptr;
        double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str()) fail("invalid number");
        return Json(v);
    }

    static void appendUtf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') break;
            if (c != '\\') { out += c; continue; }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("bad unicode escape");
                    unsigned cp = static_cast<unsigned>(std::strtoul(s_.substr(pos_, 4).c_str(), nullptr, 16));
                    pos_ += 4;
                    appendUtf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }

    Json parseArray() {
        expect('[');
        Json arr = Json::array();
        if (peek() == ']') { ++pos_; return arr; }
        while (true) {
            arr.push(parseValue());
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; break; }
            fail("expected , or ]");
        }
        return arr;
    }

    Json parseObject() {
        expect('{');
        Json obj = Json::object();
        if (peek() == '}') { ++pos_; return obj; }
        while (true) {
            if (peek() != '"') fail("expected key");
            std::string key = parseString();
            expect(':');
            obj[key] = parseValue();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; break; }
            fail("expected , or }");
        }
        return obj;
    }

    const std::string& s_;
    std::size_t pos_ = 0;
};

void dumpString(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
}

}  // namespace

Json Json::parse(const std::string& text) { return Parser(text).parseDocument(); }

bool Json::has(const std::string& key) const {
    if (type_ != Type::Object) return false;
    for (const auto& kv : o_) if (kv.first == key) return true;
    return false;
}

const Json& Json::operator[](const std::string& key) const {
    if (type_ != Type::Object) return nullJson();
    for (const auto& kv : o_) if (kv.first == key) return kv.second;
    return nullJson();
}

Json& Json::operator[](const std::string& key) {
    if (type_ == Type::Null) type_ = Type::Object;
    if (type_ != Type::Object) throw std::runtime_error("JSON value is not an object");
    for (auto& kv : o_) if (kv.first == key) return kv.second;
    o_.emplace_back(key, Json());
    return o_.back().second;
}

double Json::getNum(const std::string& key, double def) const {
    const Json& v = (*this)[key];
    return v.isNumber() || v.isBool() ? v.num(def) : def;
}

int Json::getInt(const std::string& key, int def) const {
    const Json& v = (*this)[key];
    return v.isNumber() ? v.integer(def) : def;
}

bool Json::getBool(const std::string& key, bool def) const {
    const Json& v = (*this)[key];
    return v.isBool() || v.isNumber() ? v.boolean(def) : def;
}

std::string Json::getStr(const std::string& key, const std::string& def) const {
    const Json& v = (*this)[key];
    return v.isString() ? v.str() : def;
}

std::size_t Json::size() const {
    if (type_ == Type::Array) return a_.size();
    if (type_ == Type::Object) return o_.size();
    return 0;
}

const Json& Json::at(std::size_t i) const {
    if (type_ != Type::Array || i >= a_.size()) return nullJson();
    return a_[i];
}

Json& Json::at(std::size_t i) {
    if (type_ != Type::Array || i >= a_.size()) throw std::runtime_error("JSON array index out of range");
    return a_[i];
}

void Json::push(Json v) {
    if (type_ == Type::Null) type_ = Type::Array;
    if (type_ != Type::Array) throw std::runtime_error("JSON value is not an array");
    a_.push_back(std::move(v));
}

std::string Json::dump() const {
    std::string out;
    out.reserve(256);
    dumpTo(out);
    return out;
}

void Json::dumpTo(std::string& out) const {
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += b_ ? "true" : "false"; break;
        case Type::Number: {
            if (!std::isfinite(n_)) { out += "null"; break; }
            char buf[32];
            if (std::fabs(n_ - std::round(n_)) < 1e-9 && std::fabs(n_) < 1e15)
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(std::llround(n_)));
            else
                std::snprintf(buf, sizeof(buf), "%.6g", n_);
            out += buf;
            break;
        }
        case Type::String: dumpString(out, s_); break;
        case Type::Array: {
            out += '[';
            for (std::size_t i = 0; i < a_.size(); ++i) {
                if (i) out += ',';
                a_[i].dumpTo(out);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            for (std::size_t i = 0; i < o_.size(); ++i) {
                if (i) out += ',';
                dumpString(out, o_[i].first);
                out += ':';
                o_[i].second.dumpTo(out);
            }
            out += '}';
            break;
        }
    }
}

}  // namespace tp
