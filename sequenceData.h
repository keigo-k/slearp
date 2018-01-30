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
#include <vector>
#include <iostream>
#include <stdlib.h>
#include "utility.h"

using namespace std;

typedef struct {
  char **segx;
  char **segy;
  unsigned short lenseg;
  char **nosegy;
  unsigned short leny;
} YInfo;

class YInfoVec : public vector<YInfo> {
public:
  YInfoVec(void) {}
  ~YInfoVec(void) {
    YInfoVec::iterator it;
    YInfoVec::iterator endit = (*this).end();
    for (it = (*this).begin(); it != endit; ++it) {
      free(it->segx);
    }
  }
};

class sequenceData {
public:
  char **nosegx;
  unsigned short lenx;
  YInfoVec yInfoVec;

  sequenceData *next;

  ~sequenceData(void) {
    if (nosegx) {
      free(nosegx);
    }
  };
};

class hashSequenceData {
  sequenceData **hashWithlen(const char *nosegx, unsigned short lenx);
  int keyequalWithlen(const char *x, const char *nosegx, unsigned short lenx);

public:
  unsigned int hashRow;
  unsigned int hashCol;

  sequenceData ***sequenceDataTable;
  unsigned int numData;
  unsigned int numUniqData;

  hashSequenceData(void);
  ~hashSequenceData(void);

  void initialize(unsigned int row, unsigned int col);

  YInfo regist(const string &str);
  sequenceData *refer(const char *nosegx, unsigned short lenx);
  static void mkSequenceData(const string &str, sequenceData &data);
};

