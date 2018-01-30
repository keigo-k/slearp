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

#include "hashPronunceRule.h"
#include "utility.h"
#include <iostream>
#include <fstream>

hashPronunceRule::hashPronunceRule() : PronunceRuleTable(NULL) {}

hashPronunceRule::~hashPronunceRule() {
  if (PronunceRuleTable == NULL) {
    return;
  }

  vector<char *>::iterator it = allStringVec.begin();
  vector<char *>::iterator endit = allStringVec.end();
  for (; it != endit; ++it) {
    free(*it);
  }

  PronunceRule *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = PronunceRuleTable[i][j]; p != NULL; p = tmp) {
        tmp = p->next;
        delete p;
      }
    }
  }
  free(PronunceRuleTable);
}

RuleLen hashPronunceRule::rulelen(const char *target) {
  RuleLen len = 0;
  while (*target != '\0') {
    if (*target == joinChar || *target == unknownChar) {
      ++len;
    }
    ++target;
  }

  ++len;
  return len;
}

PronunceRule **hashPronunceRule::hashWithlen(const char *targetx,
                                             RuleLen lenTargetx) {

  PronunceRule **p;
  unsigned int value = 0;
  const char *tmp = targetx;
  char c;
  while (lenTargetx) {
    --lenTargetx;
    while ((c = *targetx) != unknownChar && c != joinChar &&
           c != separateChar && c != '\0') {
      value += (c * c) * (127 << lenTargetx);
      ++targetx;
    }
    ++targetx;
    value /= 7;
  }
  p = PronunceRuleTable[value % hashRow];

  while (targetx != tmp) {
    while ((c = *tmp) != unknownChar && c != joinChar && c != separateChar &&
           c != '\0') {
      value += (c * c) * (127 << lenTargetx);
      ++tmp;
    }
    value /= 3;
    ++lenTargetx;
    ++tmp;
  }

  return &(p[value % hashCol]);
}

int hashPronunceRule::keyequalWithlen(const char *x, const char *targetx,
                                      RuleLen lenTargetx) {
  // have already verified that two length is equal.

  char c;
  while (lenTargetx) {
    while ((c = *targetx) != unknownChar && c != joinChar &&
           c != separateChar && c != '\0') {
      if (*x != c) {
        return 0;
      }
      ++x;
      ++targetx;
    }
    if (*x != unknownChar && *x != joinChar && *x != separateChar &&
        *x != '\0') {
      return 0;
    }
    ++x;
    ++targetx;
    --lenTargetx;
  }

  return 1;
}

void hashPronunceRule::initialize(unsigned int row, unsigned int col) {
  unsigned int i, j;

  numRule = 0;
  maxlenx = 1;
  maxleny = 1;
  hashRow = row;
  hashCol = col;

  if ((PronunceRuleTable = (PronunceRule ***)malloc(
           row * sizeof(PronunceRule **) +
           row * col * sizeof(PronunceRule *))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < row; ++i) {
    PronunceRuleTable[i] = (PronunceRule **)(PronunceRuleTable + row) + i * col;
    for (j = 0; j < col; ++j) {
      PronunceRuleTable[i][j] = NULL;
    }
  }
}

void hashPronunceRule::regist(const char *targetx, const char *targety) {
  RuleLen lenTargetx = rulelen(targetx);
  RuleLen lenTargety = rulelen(targety);
  PronunceRule **pp = hashWithlen(targetx, lenTargetx);
  PronunceRule *p = *pp;

  for (; p != NULL; p = p->next) {
    if (p->lenx == lenTargetx && keyequalWithlen(p->x, targetx, lenTargetx)) {

      vector<Pronunce>::iterator it;
      vector<Pronunce>::iterator endit = p->pronunceVec.end();
      for (it = p->pronunceVec.begin(); it != endit; ++it) {
        if ((*it).leny == lenTargety &&
            keyequalWithlen((*it).y, targety, lenTargety)) {

          return;
        }
      }

      if (lenTargety > maxleny) {
        maxleny = lenTargety;
      }

      Pronunce pronunce = { targety, lenTargety };
      p->pronunceVec.push_back(pronunce);

      ++numRule;
      return;
    }
  }

  p = new PronunceRule;
  p->next = *pp;
  *pp = p;

  p->x = targetx;
  p->lenx = lenTargetx;
  if (lenTargetx > maxlenx) {
    maxlenx = lenTargetx;
  }

  if (lenTargety > maxleny) {
    maxleny = lenTargety;
  }
  Pronunce pronunce = { targety, lenTargety };
  p->pronunceVec.push_back(pronunce);

  ++numRule;
}

PronunceRule *hashPronunceRule::registWithMemory(const char *targetx,
                                                 unsigned short storeSizex,
                                                 const char *targety,
                                                 unsigned short storeSizey) {
  RuleLen lenTargetx = rulelen(targetx);
  RuleLen lenTargety = rulelen(targety);

  PronunceRule **pp = hashWithlen(targetx, lenTargetx);
  PronunceRule *p = *pp;

  for (; p != NULL; p = p->next) {
    if (p->lenx == lenTargetx && keyequalWithlen(p->x, targetx, lenTargetx)) {

      vector<Pronunce>::iterator it;
      vector<Pronunce>::iterator endit = p->pronunceVec.end();
      for (it = p->pronunceVec.begin(); it != endit; ++it) {
        if ((*it).leny == lenTargety &&
            keyequalWithlen((*it).y, targety, lenTargety)) {
          return p;
        }
      }

      if (lenTargety > maxleny) {
        maxleny = lenTargety;
      }

      char *pstr;
      if ((pstr = (char *)malloc(storeSizey)) == NULL) {
        cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
                "memory.\n";
        exit(EXIT_FAILURE);
      }

      memcpy(pstr, targety, storeSizey);
      Pronunce pronunce = { pstr, lenTargety };
      p->pronunceVec.push_back(pronunce);
      allStringVec.push_back(pstr);

      ++numRule;
      return p;
    }
  }

  p = new PronunceRule;

  p->next = *pp;
  *pp = p;

  char *pstr;
  if ((pstr = (char *)malloc(storeSizex + storeSizey)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }
  memcpy(pstr, targetx, storeSizex);
  p->x = pstr;
  p->lenx = lenTargetx;
  if (lenTargetx > maxlenx) {
    maxlenx = lenTargetx;
  }

  if (lenTargety > maxleny) {
    maxleny = lenTargety;
  }
  memcpy(pstr + storeSizex, targety, storeSizey);
  Pronunce pronunce = { pstr + storeSizex, lenTargety };
  p->pronunceVec.push_back(pronunce);
  allStringVec.push_back(pstr);

  ++numRule;
  return p;
}

PronunceRule *hashPronunceRule::refer(const char *targetx, RuleLen lenTargetx) {
  PronunceRule **pp = hashWithlen(targetx, lenTargetx);
  PronunceRule *p = *pp;

  for (; p != NULL; p = p->next) {
    if (p->lenx == lenTargetx && keyequalWithlen(p->x, targetx, lenTargetx)) {
      return p;
    }
  }
  return NULL;
}

void hashPronunceRule::writePronunceRules(const char *write) {
  ofstream WRITE;
  string writeFile(".rule");
  writeFile = write + writeFile;
  WRITE.open(writeFile.c_str(), ios_base::trunc);
  if (!WRITE) {
    cerr << "ERROR:Can not write pronunce rule: " << writeFile << endl;
    exit(EXIT_FAILURE);
  } else {
    PronunceRule *p;
    unsigned int i, j;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = PronunceRuleTable[i][j]; p != NULL; p = p->next) {
          vector<Pronunce>::iterator it;
          vector<Pronunce>::iterator endit = p->pronunceVec.end();
          for (it = p->pronunceVec.begin(); it != endit; ++it) {
            WRITE << p->x;
            WRITE << "\t";
            WRITE << (*it).y;
            WRITE << "\n";
          }
        }
      }
    }
    WRITE.close();
  }
}

