// This file was autogenerated by eigen_lcm
// DO NOT MODIFY BY HAND!
#pragma once
#include <utility>

#include <lcm/lcm_coretypes.h>
#ifdef SKYMARSHAL_PRINTING_ENABLED
#include <lcm/lcm_reflection_eigen.hpp>
#endif
#include <Eigen/Core>

namespace eigen_lcm {

class Matrix2d : public Eigen::Matrix<double, 2, 2, Eigen::DontAlign> {
 public:
  // Default constructor. Unlike Eigen, eigen_lcm types initialize themselves to zero.
  // This is to prevent all sorts of uninitialized memory footguns.
  // Since the X sized variants are zero-sized by default, it is OK to default construct them.
  Matrix2d() : Eigen::Matrix<double, 2, 2, Eigen::DontAlign>() {
    *this = Matrix2d::Zero();
  }

  // Pass through constructor
  template <typename... Args>
  Matrix2d(Args&&... args) : Eigen::Matrix<double, 2, 2, Eigen::DontAlign>(args...) {}

  // This method allows you to assign Eigen expressions to eigen_lcm::Matrix2d
  template <typename OtherDerived>
  eigen_lcm::Matrix2d& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    Eigen::Matrix<double, 2, 2, Eigen::DontAlign>::operator=(other);
    return *this;
  }

  // Stream formatting operator
  friend std::ostream& operator<<(std::ostream& os, const eigen_lcm::Matrix2d& obj) {
    os << "Matrix2d(";
    for (size_t i = 0; i < obj.size(); ++i) {
      os << obj.data()[i];
      if (i + 1 != obj.size()) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }

 private:
  // Disable comma initialization magic for eigen_lcm types
  template <typename T>
  Eigen::Matrix<double, 2, 2, Eigen::DontAlign>& operator<<(T other);

 public:
  /**
   * Encode a message into binary form.
   *
   * @param buf The output buffer.
   * @param offset Encoding starts at thie byte offset into @p buf.
   * @param maxlen Maximum number of bytes to write.  This should generally be
   *  equal to getEncodedSize().
   * @return The number of bytes encoded, or <0 on error.
   */
  inline __lcm_buffer_size encode(void* buf, __lcm_buffer_size offset,
                                  __lcm_buffer_size maxlen) const;

  /**
   * Check how many bytes are required to encode this message.
   */
  inline __lcm_buffer_size getEncodedSize() const;

  /**
   * Decode a message from binary form into this instance.
   *
   * @param buf The buffer containing the encoded message.
   * @param offset The byte offset into @p buf where the encoded message starts.
   * @param maxlen The maximum number of bytes to read while decoding.
   * @return The number of bytes decoded, or <0 if an error occured.
   */
  inline __lcm_buffer_size decode(const void* buf, __lcm_buffer_size offset,
                                  __lcm_buffer_size maxlen);

  /**
   * Retrieve the 64-bit fingerprint identifying the structure of the message.
   * Note that the fingerprint is the same for all instances of the same
   * message type, and is a fingerprint on the message type definition, not on
   * the message contents.
   */
  constexpr static int64_t getHash();

  using type_name_array_t = const char[9];

  inline static constexpr type_name_array_t* getTypeNameArrayPtr();

  /**
   * Returns "Matrix2d"
   */
  inline static constexpr const char* getTypeName();

  /**
   * Returns "eigen_lcm"
   */
  inline static constexpr const char* getPackageName();

  // LCM support functions. Users should not call these
  inline __lcm_buffer_size _encodeNoHash(void* buf, __lcm_buffer_size offset,
                                         __lcm_buffer_size maxlen) const;
  inline __lcm_buffer_size _getEncodedSizeNoHash() const;
  inline __lcm_buffer_size _decodeNoHash(const void* buf, __lcm_buffer_size offset,
                                         __lcm_buffer_size maxlen);
  constexpr static uint64_t _computeHash(const __lcm_hash_ptr*) {
    uint64_t hash = 0x7e2b2f4ef42123f2LL;
    return (hash << 1) + ((hash >> 63) & 1);
  }
};

__lcm_buffer_size Matrix2d::encode(void* buf, __lcm_buffer_size offset,
                                   __lcm_buffer_size maxlen) const {
  __lcm_buffer_size pos = 0, tlen;
  int64_t hash = (int64_t)getHash();

  tlen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  tlen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix2d::decode(const void* buf, __lcm_buffer_size offset,
                                   __lcm_buffer_size maxlen) {
  __lcm_buffer_size pos = 0, thislen;

  int64_t msg_hash;
  thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
  if (thislen < 0)
    return thislen;
  else
    pos += thislen;
  if (msg_hash != getHash())
    return -1;

  thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
  if (thislen < 0)
    return thislen;
  else
    pos += thislen;

  return pos;
}

__lcm_buffer_size Matrix2d::getEncodedSize() const {
  return 8 + _getEncodedSizeNoHash();
}

constexpr int64_t Matrix2d::getHash() {
  return static_cast<int64_t>(_computeHash(NULL));
}

constexpr Matrix2d::type_name_array_t* Matrix2d::getTypeNameArrayPtr() {
  return &"Matrix2d";
}

constexpr const char* Matrix2d::getTypeName() {
  return *Matrix2d::getTypeNameArrayPtr();
}

constexpr const char* Matrix2d::getPackageName() {
  return "eigen_lcm";
}

__lcm_buffer_size Matrix2d::_encodeNoHash(void* buf, __lcm_buffer_size offset,
                                          __lcm_buffer_size maxlen) const {
  __lcm_buffer_size pos = 0, tlen;

  tlen = __double_encode_array(buf, offset + pos, maxlen - pos, this->data(), 4);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix2d::_decodeNoHash(const void* buf, __lcm_buffer_size offset,
                                          __lcm_buffer_size maxlen) {
  __lcm_buffer_size pos = 0, tlen;

  tlen = __double_decode_array(buf, offset + pos, maxlen - pos, this->data(), 4);
  if (tlen < 0)
    return tlen;
  else
    pos += tlen;

  return pos;
}

__lcm_buffer_size Matrix2d::_getEncodedSizeNoHash() const {
  __lcm_buffer_size enc_size = 0;
  enc_size += __double_encoded_array_size(NULL, 4);
  return enc_size;
}

}  // namespace eigen_lcm
