/*
Slearp (structured learning and prediction) is the structured learning and
predict toolkit for tasks such as g2p conversion, based on discriminative
leaning.
Copyright (C) 2013, 2014 Keigo Kubo

Slearp is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

Slearp is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with slearp.  If not, see <http://www.gnu.org/licenses/>.

date:   2014/3/03
author: Keigo Kubo
e-mail: keigokubo{@}gmail.com   << Please transform {@} into @
*/

#pragma once
#include "hashFea.h"
#include <vector>
#include <sstream>
#include <string.h>
#include <iostream>

extern char unknownChar;
extern char separateChar;
extern char joinChar;
extern char delInsChar;
extern char escapeChar;

extern const char interBound;
extern const char interUnknown;
extern const char interSeparate;
extern const char interJoin;
extern const char interDelIns;

typedef struct localOptions {
  char *train;
  char *dev;
  char *eval;
  char *read;
  char *readRule;
  const char *write;
  char *method;
  bool useContext;
  bool useTrans;
  bool useChain;
  bool useJoint;
  bool useFJoint;
  unsigned short cngram;
  unsigned short jngram;
  unsigned short fjngram;
  bool useOpt; // use optimize
  unsigned short numOfThread;
  float learningRate;
  float C;
  float unSupWeight;
  unsigned short numGrad;
  const char *lossFuncTyp;
  bool useFullCovariance;
  float r;
  float b;
  float initVar;
  unsigned short trainNbest;
  bool useLocal;
  bool useAvg;
  unsigned short beamSize;
  unsigned short outputNbest;
  unsigned int iter;
  unsigned int minIter;
  unsigned short stopCond;
  unsigned short maxTrialIter;
  unsigned int hashFeaTableSize;
  unsigned int covarianceTableSize;
  unsigned int hashTrainDataTableSize;
  unsigned int hashDevDataTableSize;
  unsigned int hashPronunceRuleTableSize;
} LocalOpt;

template <typename T> inline string toString(const T &x) {
  ostringstream stream;
  stream << x;
  return stream.str();
}

template <typename T> inline T fromString(const std::string &s) {
  istringstream stream(s);
  char c;
  T x;
  if (!(stream >> x) || stream.get(c)) {
    throw "Can not convert";
  }
  return x;
}

template <typename T> inline T ABS(const T &x) { return (x > 0) ? x : -x; }

template <typename T> inline T SIGN(const T &x, const T &y) {
  return (y > 0) ? x : -x;
}

template <class T> inline void SWAP(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

