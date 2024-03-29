/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

/** @file
 * Declarations for maths routines
 */

#ifndef NTA_MATH_HPP
#define NTA_MATH_HPP

#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <vector>

#include <boost/concept_check.hpp>

#include <nupic/math/Utils.hpp>
#include <nupic/types/Types.hpp>

//--------------------------------------------------------------------------------
/**
 * Macros to make it easier to work with Boost concept checks
 */

// Assert that It obeys the STL forward iterator concept
#define ASSERT_INPUT_ITERATOR(It)                                              \
  boost::function_requires<boost::InputIteratorConcept<It>>();

// Assert that It obeys the STL forward iterator concept
#define ASSERT_OUTPUT_ITERATOR(It, T)                                          \
  boost::function_requires<boost::OutputIteratorConcept<It, T>>();

// Assert that UnaryPredicate obeys the STL unary predicate concept
#define ASSERT_UNARY_PREDICATE(UnaryPredicate, Arg1)                           \
  boost::function_requires<                                                    \
      boost::UnaryPredicateConcept<UnaryPredicate, Arg1>>();

// Assert that UnaryFunction obeys the STL unary function concept
#define ASSERT_UNARY_FUNCTION(UnaryFunction, Ret, Arg1)                        \
  boost::function_requires<                                                    \
      boost::UnaryFunctionConcept<UnaryFunction, Ret, Arg1>>();

// Assert that BinaryFunction obeys the STL binary function concept
#define ASSERT_BINARY_FUNCTION(BinaryFunction, Ret, Arg1, Arg2)                \
  boost::function_requires<                                                    \
      boost::BinaryFunctionConcept<BinaryFunction, Ret, Arg1, Arg2>>();

//--------------------------------------------------------------------------------
namespace nupic {

//--------------------------------------------------------------------------------
// ASSERTIONS
//--------------------------------------------------------------------------------
/**
 * This is used to check that a boolean condition holds, and to send a message
 * on NTA_INFO if it doesn't. It's compiled in only when NTA_ASSERTIONS_ON is
 * true.
 */
inline bool INVARIANT(bool cond, const char *msg) {
#ifdef NTA_ASSERTIONS_ON
  if (!(cond)) {
    NTA_INFO << msg;
    return false;
  }
#endif
  return true;
}

//--------------------------------------------------------------------------------
/**
 * An assert used to validate that a range defined by two iterators is valid,
 * that is: begin <= end. We could say that end < begin means the range is
 * empty, but for debugging purposes, we want to know when the iterators
 * are crossed.
 */
template <typename It>
void ASSERT_VALID_RANGE(It begin, It end, const char *message) {
  NTA_ASSERT(begin <= end) << "Invalid iterators: " << message;
}

//--------------------------------------------------------------------------------
/**
 *  Epsilon is defined for the whole math and algorithms of the Numenta
 * Platform, independently of the concrete type chosen to handle floating point
 * numbers. numeric_limits<float>::epsilon() == 1.19209e-7
 *   numeric_limits<double>::epsilon() == 2.22045e-16
 */
static const nupic::Real Epsilon = nupic::Real(1e-6);

//--------------------------------------------------------------------------------
/**
 * Functions that test for positivity or negativity based on nupic::Epsilon.
 */
template <typename T> inline bool strictlyNegative(const T &a) {
  return a < -nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool strictlyPositive(const T &a) {
  return a > nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool negative(const T &a) {
  return a <= nupic::Epsilon;
}

//--------------------------------------------------------------------------------
template <typename T> inline bool positive(const T &a) {
  return a >= -nupic::Epsilon;
}

//--------------------------------------------------------------------------------
/**
 * A functions that implements the distance to zero function as a functor.
 * Defining argument_type and result_type directly here instead of inheriting
 * from std::unary_function so that we have an easier time in SWIG Python
 * wrapping.
 */
template <typename T> struct DistanceToZero {
  typedef T argument_type;
  typedef T result_type;

  inline T operator()(const T &x) const { return x >= 0 ? x : -x; }
};

//--------------------------------------------------------------------------------
/**
 * A specialization for UInts, where we only need one test (more efficient).
 */
template <> inline UInt DistanceToZero<UInt>::operator()(const UInt &x) const {
  return x;
}

//--------------------------------------------------------------------------------
/**
 * Use this functor if T is guaranteed to be positive only.
 */
template <typename T>
struct DistanceToZeroPositive {
  inline T operator()(const T &x) const { return x; }
};

//--------------------------------------------------------------------------------
/**
 * This computes the distance to 1 rather than to 0.
 */
template <typename T> struct DistanceToOne {
  typedef T argument_type;
  typedef T result_type;

  inline T operator()(const T &x) const {
    return x > (T)1 ? x - (T)1 : (T)1 - x;
  }
};

//--------------------------------------------------------------------------------
/**
 * This functor decides whether a number is almost zero or not, using the
 * platform-wide nupic::Epsilon.
 */
template <typename D> struct IsNearlyZero {
  typedef typename D::result_type value_type;

  D dist_;

  // In the case where D::result_type is integral
  // we convert nupic::Epsilon to zero!
  inline IsNearlyZero() : dist_() {}

  inline IsNearlyZero(const IsNearlyZero &other) : dist_(other.dist_) {}

  inline IsNearlyZero &operator=(const IsNearlyZero &other) {
    if (this != &other)
      dist_ = other.dist_;

    return *this;
  }

  inline bool operator()(const typename D::argument_type &x) const {
    return dist_(x) <= nupic::Epsilon;
  }
};

//--------------------------------------------------------------------------------
/**
 * @b Responsibility:
 *  Tell whether an arithmetic value is zero or not, within some precision,
 *  or whether two values are equal or not, within some precision.
 *
 * @b Parameters:
 *  epsilon: accuracy of the comparison
 *
 * @b Returns:
 *  true if |a| <= epsilon
 *  false otherwise
 *
 * @b Requirements:
 *  T arithmetic
 *  T comparable with operator<= AND operator >=
 *
 * @b Restrictions:
 *  Doesn't compile if T is not arithmetic.
 *  In debug mode, NTA_ASSERTs if |a| > 10
 *  In debug mode, NTA_ASSERTs if a == infinity, quiet_NaN or signaling_NaN
 *
 * @b Notes:
 *  Comparing floating point numbers is a pretty tricky business. Knuth's got
 *  many pages devoted to it in Vol II. Boost::test has a special function to
 *  handle that. One of the problems is that when more bits are allocated to
 *  the integer part as the number gets bigger, there is inherently less
 *  precision in the decimals. But, for comparisons to zero, we can continue
 *  using an absolute epsilon (instead of multiplying epsilon by the number).
 *  In our application, we are anticipating numbers mostly between 0 and 1,
 *  because they represent probabilities.
 *
 *  Not clear why this is namespace std instead of nupic , but FA says there was
 *  a "good, ugly" reason to do it this way that he can't remember. - WCS 0206
 */
template <typename T>
inline bool nearlyZero(const T &a, const T &epsilon = T(nupic::Epsilon)) {
  return a >= -epsilon && a <= epsilon;
}

//--------------------------------------------------------------------------------
template <typename T>
inline bool nearlyEqual(const T &a, const T &b,
                        const T &epsilon = nupic::Epsilon) {
  return nearlyZero((b - a), epsilon);
}

//--------------------------------------------------------------------------------
/**
 * Euclidean modulo function.
 *
 * Returns x % m, but keeps the value positive
 * (similar to Python's modulo function).
 */
inline int emod(int x, int m) {
  int r = x % m;
  if (r < 0)
    return r + m;
  else
    return r;
}

//--------------------------------------------------------------------------------
/**
 * A boolean functor that returns true if the element's selected value is found
 * in the (associative) container (needs to support find).
 *
 * Example:
 * =======
 * typedef std::pair<UInt, UInt> IdxVal;
 * std::list<IdxVal> row;
 * std::set<UInt> alreadyGrouped;
 * row.remove_if(SetIncludes<std::set<UInt>, select1st<IdxVal>
 * >(alreadyGrouped));
 *
 * Will remove from row (a list of pairs) the pairs whose first element is
 * already contained in alreadyGrouped.
 */
template <typename C1, typename Selector, bool f = false> struct IsIncluded {
  IsIncluded(const C1 &container) : sel_(), container_(container) {}

  template <typename T> inline bool operator()(const T &p) const {
    if (f)
      return container_.find(sel_(p)) == container_.end();
    else
      return container_.find(sel_(p)) != container_.end();
  }

  Selector sel_;
  const C1 &container_;
};

//--------------------------------------------------------------------------------
// PAIRS AND TRIPLETS
//--------------------------------------------------------------------------------

/**
 * Lexicographic order:
 * (1,1) < (1,2) < (1,10) < (2,5) < (3,6) < (3,7) ...
 */
template <typename T1, typename T2>
struct lexicographic_2 {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    if (a.first < b.first)
      return true;
    else if (a.first == b.first)
      if (a.second < b.second)
        return true;
    return false;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the first member of a pair only:
 * (1, 3.5) < (2, 5.6) < (10, 7.1) < (11, 8.5)
 */
template <typename T1, typename T2>
struct less_1st {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.first < b.first;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the second member of a pair only:
 * (10, 3.5) < (1, 5.6) < (2, 7.1) < (11, 8.5)
 */
template <typename T1, typename T2>
struct less_2nd {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.second < b.second;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the first member of a pair only:
 * (10, 3.5) > (8, 5.6) > (2, 7.1) > (1, 8.5)
 */
template <typename T1, typename T2>
struct greater_1st {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.first > b.first;
  }
};

//--------------------------------------------------------------------------------
/**
 * Order based on the second member of a pair only:
 * (10, 3.5) > (1, 5.6) > (2, 7.1) > (11, 8.5)
 */
template <typename T1, typename T2>
struct greater_2nd {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    return a.second > b.second;
  }
};

//--------------------------------------------------------------------------------
template <typename T1, typename T2>
struct greater_2nd_p {
  inline bool operator()(const std::pair<T1, T2 *> &a,
                         const std::pair<T1, T2 *> &b) const {
    return *(a.second) > *(b.second);
  }
};

//--------------------------------------------------------------------------------
/**
 * A greater 2nd order that breaks ties, useful for debugging.
 */
template <typename T1, typename T2>
struct greater_2nd_no_ties {
  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    if (a.second > b.second)
      return true;
    else if (a.second == b.second)
      if (a.first < b.first)
        return true;
    return false;
  }
};

//--------------------------------------------------------------------------------
template <typename T1, typename T2, typename RND>
struct greater_2nd_rnd_ties {
  RND &rng;

  inline greater_2nd_rnd_ties(RND &_rng) : rng(_rng) {}

  inline bool operator()(const std::pair<T1, T2> &a,
                         const std::pair<T1, T2> &b) const {
    T2 val_a = a.second, val_b = b.second;
    if (val_a > val_b)
      return true;
    else if (val_a == val_b)
      return rng.getReal64() >= .5;
    return false;
  }
};

//--------------------------------------------------------------------------------
// A class used to work with lists of non-zeros represented in i,j,v format
//--------------------------------------------------------------------------------
/**
 * This class doesn't implement any algorithm, it just stores i,j and v.
 */
template <typename T1, typename T2> class ijv {
  typedef T1 size_type;
  typedef T2 value_type;

private:
  size_type i_, j_;
  value_type v_;

public:
  inline ijv() : i_(0), j_(0), v_(0) {}

  inline ijv(size_type i, size_type j, value_type v) : i_(i), j_(j), v_(v) {}

  inline ijv(const ijv &o) : i_(o.i_), j_(o.j_), v_(o.v_) {}

  inline ijv &operator=(const ijv &o) {
    i_ = o.i_;
    j_ = o.j_;
    v_ = o.v_;
    return *this;
  }

  inline size_type i() const { return i_; }
  inline size_type j() const { return j_; }
  inline value_type v() const { return v_; }
  inline void i(size_type ii) { i_ = ii; }
  inline void j(size_type jj) { j_ = jj; }
  inline void v(value_type vv) { v_ = vv; }

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct lexicographic {
    inline bool operator()(const ijv &a, const ijv &b) const {
      if (a.i() < b.i())
        return true;
      else if (a.i() == b.i())
        if (a.j() < b.j())
          return true;
      return false;
    }
  };

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct less_value {
    inline bool operator()(const ijv &a, const ijv &b) const {
      return a.v() < b.v();
    }
  };

  //--------------------------------------------------------------------------------
  /**
   * See just above for definition.
   */
  struct greater_value {
    inline bool operator()(const ijv &a, const ijv &b) const {
      return a.v() > b.v();
    }
  };
};

//--------------------------------------------------------------------------------
/**
 * These templates allow the implementation to use the right function depending
 * on the data type, which yield significant speed-ups when working with floats.
 * They also extend STL's arithmetic operations.
 */
//--------------------------------------------------------------------------------
// Unary functions
//--------------------------------------------------------------------------------

template <typename T> struct Identity {
  inline T &operator()(T &x) const { return x; }
  inline const T &operator()(const T &x) const { return x; }
};

template <typename T> struct Negate {
  inline T operator()(const T &x) const { return -x; }
};

template <typename T> struct Abs {
  inline T operator()(const T &x) const { return x > 0.0 ? x : -x; }
};

template <typename T> struct Square {
  inline T operator()(const T &x) const { return x * x; }
};

template <typename T> struct Cube {
  inline T operator()(const T &x) const { return x * x * x; }
};

template <typename T> struct Inverse {
  inline T operator()(const T &x) const { return 1.0 / x; }
};

template <typename T> struct Sqrt {};

template <> struct Sqrt<float> {
  inline float operator()(const float &x) const { return sqrtf(x); }
};

template <> struct Sqrt<double> {
  inline double operator()(const double &x) const { return sqrt(x); }
};

template <typename T> struct Exp {};

template <> struct Exp<float> {
  // On x86_64, there is a bug in glibc that makes expf very slow
  // (more than it should be), so we continue using exp on that
  // platform as a workaround.
  // https://bugzilla.redhat.com/show_bug.cgi?id=521190
  // To force the compiler to use exp instead of expf, the return
  // type (and not the argument type!) needs to be double.
  inline float operator()(const float &x) const { return expf(x); }
};

template <> struct Exp<double> {
  inline double operator()(const double &x) const { return exp(x); }
};

template <typename T> struct Log {};

template <> struct Log<float> {
  inline float operator()(const float &x) const { return logf(x); }
};

template <> struct Log<double> {
  inline double operator()(const double &x) const { return log(x); }
};

template <typename T> struct Log2 {};

template <> struct Log2<float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)(log(x) / log(2.0));
#else
    return log2f(x);
#endif
  }
};

template <> struct Log2<double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(x) / log(2.0);
#else
    return log2(x);
#endif
  }
};

template <typename T> struct Log10 {};

template <> struct Log10<float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)(log(x) / log(10.0));
#else
    return log10f(x);
#endif
  }
};

template <> struct Log10<double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(x) / log(10.0);
#else
    return log10(x);
#endif
  }
};

template <typename T> struct Log1p {};

template <> struct Log1p<float> {
  inline float operator()(const float &x) const {
#if defined(NTA_OS_WINDOWS)
    return (float)log(1.0 + x);
#else
    return log1pf(x);
#endif
  }
};

template <> struct Log1p<double> {
  inline double operator()(const double &x) const {
#if defined(NTA_OS_WINDOWS)
    return log(1.0 + x);
#else
    return log1p(x);
#endif
  }
};

/**
 * Numerical approximation of derivative.
 * Error is h^4 y^5/30.
 */
template <typename Float, typename F>
struct Derivative {
  Derivative(const F &f) : f_(f) {}

  F f_;

  /**
   * Approximates the derivative of F at x.
   */
  inline const Float operator()(const Float &x) const {
    const Float h = nupic::Epsilon;
    return (-f_(x + 2 * h) + 8 * f_(x + h) - 8 * f_(x - h) + f_(x - 2 * h)) /
           (12 * h);
  }
};

//--------------------------------------------------------------------------------
// Binary functions
//--------------------------------------------------------------------------------
template <typename T> struct Assign {
  inline T operator()(T &x, const T &y) const {
    x = y;
    return x;
  }
};

template <typename T> struct Plus {
  inline T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T> struct Minus {
  inline T operator()(const T &x, const T &y) const { return x - y; }
};

template <typename T> struct Multiplies {
  inline T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T> struct Divides {
  inline T operator()(const T &x, const T &y) const { return x / y; }
};

template <typename T> struct Pow {};

template <>
struct Pow<float> {
  inline float operator()(const float &x, const float &y) const {
    return powf(x, y);
  }
};

template <>
struct Pow<double> {
  inline double operator()(const double &x, const double &y) const {
    return pow(x, y);
  }
};

template <typename T> struct Logk {};

template <>
struct Logk<float> {
  inline float operator()(const float &x, const float &y) const {
    return logf(x) / logf(y);
  }
};

template <>
struct Logk<double> {
  inline double operator()(const double &x, const double &y) const {
    return log(x) / log(y);
  }
};

template <typename T> struct Max {
  inline T operator()(const T &x, const T &y) const { return x > y ? x : y; }
};

template <typename T> struct Min {
  inline T operator()(const T &x, const T &y) const { return x < y ? x : y; }
};

//--------------------------------------------------------------------------------
/**
 * Gaussian:
 * y = 1/(sigma * sqrt(2*pi)) * exp(-(x-mu)^2/(2*sigma^2)) as a functor.
 */
template <typename T> struct Gaussian {
  T k1, k2, mu;

  inline Gaussian(T m, T s) : k1(0.0), k2(0.0), mu(m) {
    // For some reason, SWIG cannot parse 1 / (x), the parentheses in the
    // denominator don't agree with it, so we have to initialize those
    // constants here.
    k1 = 1.0 / sqrt(2.0 * 3.1415926535);
    k2 = -1.0 / (2.0 * s * s);
  }

  inline Gaussian(const Gaussian &o) : k1(o.k1), k2(o.k2), mu(o.mu) {}

  inline Gaussian &operator=(const Gaussian &o) {
    if (&o != this) {
      k1 = o.k1;
      k2 = o.k2;
      mu = o.mu;
    }

    return *this;
  }

  inline T operator()(T x) const {
    T v = x - mu;
    return k1 * exp(k2 * v * v);
  }
};

//--------------------------------------------------------------------------------
/**
 * 2D Gaussian
 */
template <typename T>
struct Gaussian2D
{
  T c_x, c_y, s00, s01, s10, s11, s2, k1;

  inline Gaussian2D(T c_x_, T c_y_, T s00_, T s01_, T s10_, T s11_)
      : c_x(c_x_), c_y(c_y_), s00(s00_), s01(s01_), s10(s10_), s11(s11_),
        s2(s10 + s01), k1(0.0) {
    // For some reason, SWIG cannot parse 1 / (x), the parentheses in the
    // denominator don't agree with it, so we have to initialize those
    // constants here.
    k1 = 1.0 / (2.0 * 3.1415926535 * sqrt(s00 * s11 - s10 * s01));
    T d = -2.0 * (s00 * s11 - s10 * s01);
    s00 /= d;
    s01 /= d;
    s10 /= d;
    s11 /= d;
    s2 /= d;
  }

  inline Gaussian2D(const Gaussian2D &o)
      : c_x(o.c_x), c_y(o.c_y), s00(o.s00), s01(o.s01), s10(o.s10), s11(o.s11),
        s2(o.s2), k1(o.k1) {}

  inline Gaussian2D &operator=(const Gaussian2D &o) {
    if (&o != this) {
      c_x = o.c_x;
      c_y = o.c_y;
      s00 = o.s00;
      s01 = o.s01;
      s10 = o.s10;
      s11 = o.s11;
      s2 = o.s2;
      k1 = o.k1;
    }

    return *this;
  }

  inline T operator()(T x, T y) const {
    T v0 = x - c_x, v1 = y - c_y;
    return k1 * exp(s11 * v0 * v0 + s2 * v0 * v1 + s00 * v1 * v1);
  }
};

//--------------------------------------------------------------------------------
/**
 * Compose two unary functions.
 */
template <typename F1, typename F2>
struct unary_compose {
  typedef typename F1::argument_type argument_type;
  typedef typename F2::result_type result_type;

  F1 f1;
  F2 f2;

  inline result_type operator()(const argument_type &x) const {
    return f2(f1(x));
  }
};

//--------------------------------------------------------------------------------
/**
 * Compose an order predicate and a binary selector, so that we can write:
 * sort(x.begin(), x.end(), compose<less<float>, select2nd<pair<int, float> >
 * >()); to sort pairs in increasing order of their second element.
 */
template <typename O, typename S>
struct predicate_compose {
  typedef bool result_type;
  typedef typename S::argument_type argument_type;

  O o;
  S s;

  inline result_type operator()(const argument_type &x,
                                const argument_type &y) const {
    return o(s(x), s(y));
  }
};

//--------------------------------------------------------------------------------
/**
 *  When dividing by a value less than min_exponent10, inf will be generated.
 *   numeric_limits<float>::min_exponent10 = -37
 *   numeric_limits<double>::min_exponent10 = -307
 */
template <typename T> inline bool isSafeForDivision(const T &x) {
  Log<T> log_f;
  return log_f(x) >= std::numeric_limits<T>::min_exponent10;
}

//--------------------------------------------------------------------------------
/**
 * Returns the value passed in or a threshold if the value is >= threshold.
 */
template <typename T> struct ClipAbove {
  inline ClipAbove(const T &val) : val_(val) {}

  inline ClipAbove(const ClipAbove &c) : val_(c.val_) {}

  inline ClipAbove &operator=(const ClipAbove &c) {
    if (this != &c)
      val_ = c.val_;

    return *this;
  }

  inline T operator()(const T &x) const { return x >= val_ ? val_ : x; }

  T val_;
};

//--------------------------------------------------------------------------------
/**
 * Returns the value passed in or a threshold if the value is < threshold.
 */
template <typename T> struct ClipBelow {
  inline ClipBelow(const T &val) : val_(val) {}

  inline ClipBelow(const ClipBelow &c) : val_(c.val_) {}

  inline ClipBelow &operator=(const ClipBelow &c) {
    if (this != &c)
      val_ = c.val_;

    return *this;
  }

  inline T operator()(const T &x) const { return x < val_ ? val_ : x; }

  T val_;
};

//--------------------------------------------------------------------------------

}; // namespace nupic

#endif // NTA_MATH_HPP
