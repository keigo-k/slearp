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
#include <vector>
#include <stdlib.h>

using namespace std;

typedef short RuleLen;
typedef struct pronunce {
  const char *y;
  RuleLen leny;
} Pronunce;

typedef struct pronunceRule {
  const char *x;
  RuleLen lenx;
  vector<Pronunce> pronunceVec;

  struct pronunceRule *next;
} PronunceRule;

class hashPronunceRule {
  unsigned int hashRow;
  unsigned int hashCol;
  vector<char *> allStringVec;

  RuleLen rulelen(const char *target);

  PronunceRule **hashWithlen(const char *targetx, RuleLen lenTargetx);

  int keyequalWithlen(const char *x, const char *targetx, RuleLen lenTargetx);

public:
  PronunceRule ***PronunceRuleTable;
  unsigned int numRule;
  RuleLen maxlenx;
  RuleLen maxleny;

  hashPronunceRule(void);
  ~hashPronunceRule(void);

  void initialize(unsigned int row, unsigned int col);

  void regist(const char *targetx, const char *targety);
  PronunceRule *registWithMemory(const char *targetx, unsigned short storeSizex,
                                 const char *targety,
                                 unsigned short storeSizey);

  PronunceRule *refer(const char *targetx, RuleLen lenTargetx);
  void writePronunceRules(const char *write);
};

