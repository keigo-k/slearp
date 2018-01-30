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
#include <string>
#include <stdlib.h>
#include <string.h>

template <typename T, typename K> inline T MAX(const T &x, const K &y) {
  return (x > y) ? x : y;
}

template <typename T, typename K> inline T MIN(const T &x, const K &y) {
  return (x < y) ? x : y;
}

using namespace std;

typedef float ParaVal;
class Fea {
public:
  char *fea;
  ParaVal w;

  void *next;
};

class hashFea {
protected:
  unsigned int hashRow;
  unsigned int hashCol;

  inline void **hash(const char *targetfea);
  inline int keyequal(const char *fea, const char *targetfea);

public:
  void ***FeaTable;
  unsigned int numFea;

  hashFea(void);
  ~hashFea(void);

  void initialize(unsigned int row, unsigned int col);

  Fea *get(const string &target);
  Fea *refer(const string &target);
  ParaVal getParaVal(const string &target);
  void writeFeatures(const char *write);
};

class crfceFea : public Fea {
public:
  ParaVal tmpW;
  ParaVal g;
  ParaVal tmpG;
  ParaVal unSupG;
  ParaVal dir;
  ParaVal *y; // (g2-g1) in l-bgfs
  ParaVal *s; // (w2-w1) in l-bgfs
};

class hashCrfceFea : public hashFea {
  float *a;
  float *r;
  template <typename T> inline T SIGN(const T &x, const T &y) {
    return (y > 0) ? x : -x;
  }

public:
  unsigned short numGrad;

  hashCrfceFea(void);
  ~hashCrfceFea(void);

  void initialize(unsigned int row, unsigned int col, unsigned short numgrad);

  crfceFea *get(const string &target);
  void setWeight(const string &target, ParaVal val);

  // for supervised
  void updateWeightForSup(float rate, float C);
  void nextSettingInIter1ForSup();
  void updateWeightForSupByLBGFS(float rate, float C, unsigned short storeSize);
  void nextSettingForSup(float C, unsigned short storeSize);

  // for unsupervised
  void updateWeight(float supWeight, float unSupWeight, float rate, float C);
  void nextSettingInIter1(float supWeight, float unSupWeight);
  void updateWeightByLBGFS(float supWeight, float unSupWeight, float rate,
                           float C, unsigned short storeSize);
  void nextSetting(float supWeight, float unSupWeight, float C,
                   unsigned short storeSize);

  void addGrad(const string &target, ParaVal val);
  void addUnSupGrad(const string &target, ParaVal val);
};

class cwFea : public Fea {
public:
  ParaVal avgw;
  ParaVal v;
  ParaVal tdelta;
};

class Covariance {
public:
  cwFea *a;
  cwFea *b;
  ParaVal cov;
  Covariance *next;
};

class hashCwFea : public hashFea {
protected:
  inline Covariance **hashCov(cwFea *a, cwFea *b);

public:
  unsigned int hashCovRow;
  unsigned int hashCovCol;
  unsigned long int numCov;
  float initVar;
  float initCov;
  Covariance ***covariance;

  hashCwFea(void);
  ~hashCwFea(void);

  void initialize(unsigned int row, unsigned int col, unsigned int covRow,
                  unsigned int covCol, float initVar);

  void averageWeight();
  void swapWeight();
  void oneStepRegularization(float lamda);

  cwFea *get(const string &target);
  void setWeight(const string &target, ParaVal val);

  Covariance *getCovariance(cwFea *a, cwFea *b);
  void setCovariance(cwFea *a, cwFea *b, ParaVal val);
  void writeAvgFeatures(const char *write);
  void writeCovariance();
};

class ssvmFea : public Fea {
public:
  ParaVal g;
  ParaVal avgw;
};

class hashSsvmFea : public hashFea {
public:
  hashSsvmFea(void);
  ~hashSsvmFea(void);

  void updateWeightL1(const float C);
  void updateWeightL2(const float C);
  void averageWeight();
  void swapWeight();
  ssvmFea *get(const string &target);
  void setWeight(const string &target, ParaVal val);

  void writeAvgFeatures(const char *write);
};

