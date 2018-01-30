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

#include <math.h>
#include "utility.h"

class QPsolver {
protected:
  float EPSILON;
  short ITER_MAX;

  void gaussElimination(vector<vector<double> > &l, vector<double> &r, int n);

public:
  QPsolver(void) : EPSILON(1.0E-5), ITER_MAX(30) {};

  // For linear constrains (II) x = c, where I is identify matrix and x has n*2
  // dimention.
  // This constrains represent \alpha+\beta=C in L1 sscw.
  // second order form: x^T \hat{q} x + x^T \hat{v}, s.t. (II) x = c
  void solveWithIILinearConstraint(vector<vector<double> > &q,
                                   vector<double> &v, vector<double> &x,
                                   double C);

  // second order form: x^T q x + x^T v
  void solveWithoutLinearConstraint(vector<vector<double> > &q,
                                    vector<double> &v, vector<double> &x);
};

