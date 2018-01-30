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
#include <iostream>
#include <stdlib.h>
#include <string.h>
#ifndef _INCLUDE_STRCMP_
#define _INCLUDE_STRCMP_
#define STRCMP(a, b) (*(a) == *(b) && !strcmp(a, b))
#endif

using namespace std;

class OptInfo {
public:
  char *optname;
  char *type;
  void *address;
  OptInfo(char *n, char *t, void *a) : optname(n), type(t), address(a) {}
  ~OptInfo(void) {}
  OptInfo set(char *n, char *t, void *a) {
    optname = n;
    type = t;
    address = a;
    return *this;
  }
};

class parseOption {
public:
  vector<OptInfo> optionVec;

  parseOption(short numOption);
  ~parseOption(void) {}
  void parseArgv(char **argv);
};

