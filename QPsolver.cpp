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

#include <iostream>
#include <vector>
#include <sstream>
#include "QPsolver.h"

using namespace std;

void QPsolver::gaussElimination(vector<vector<double> > &l, vector<double> &r,
                                int n) {
  vector<int> irs, ics;
  irs.reserve(n);
  ics.reserve(n);

  int i, j, k;
  for (i = 0; i < n; i++) {
    irs.push_back(i);
    ics.push_back(i);
  }

  double tmp;
  double pmax;
  int ir, ic, itmp;

  /*
  for(i=0;i<n;++i){
          for(j=0;j<n;++j){
                  cerr << l[i][j] << " ";
          }
          cerr << "= " << r[i];
          cerr << endl << endl;
  }
  */

  bool changePivot = false;
  for (i = 0; i < n; i++) {
    itmp = ics[i];
    ir = i;
    pmax = ABS(l[irs[i]][itmp]);

    for (j = i + 1; j < n; j++) {
      tmp = ABS(l[irs[j]][itmp]);
      if (pmax < tmp) {
        pmax = tmp;
        ir = j;
        changePivot = true;
      }
    }

    if (pmax == 0.0) {
      changePivot = true;

      itmp = irs[i];
      ic = i;
      for (j = i + 1; j < n; j++) {
        tmp = ABS(l[itmp][ics[j]]);
        if (pmax < tmp) {
          pmax = tmp;
          ic = j;
        }
      }

      if (pmax == 0.0) {

        for (j = i + 1; j < n; j++) {
          for (k = i + 1; k < n; k++) {
            tmp = ABS(l[irs[j]][ics[k]]);
            if (pmax < tmp) {
              pmax = tmp;
              ir = j;
              ic = k;
            }
          }
        }

        if (pmax == 0.0) {
          break;
        } else {
          itmp = ics[i];
          ics[i] = ics[ic];
          ics[ic] = itmp;

          itmp = irs[i];
          irs[i] = irs[ir];
          irs[ir] = itmp;
        }

      } else if (ic != i) {
        itmp = ics[i];
        ics[i] = ics[ic];
        ics[ic] = itmp;
      }

    } else if (ir != i) {
      itmp = irs[i];
      irs[i] = irs[ir];
      irs[ir] = itmp;
    }

    ic = ics[i];
    ir = irs[i];
    for (j = i + 1; j < n; j++) {
      itmp = irs[j];
      tmp = l[itmp][ic] / l[ir][ic];
      l[itmp][ic] = 0.0;

      for (k = i + 1; k < n; k++) {
        l[itmp][ics[k]] -= l[ir][ics[k]] * tmp;
      }
      r[itmp] -= r[ir] * tmp;
    }

    /*
    for(int y=0;y<n;++y){
            for(int x=0;x<n;++x){
                    cerr << l[irs[y]][ics[x]] << " ";
            }
            cerr << "= " << r[irs[y]];
            cerr << endl << endl;
    }
    */
  }

  // Elimination
  for (i = n - 1; i >= 0; i--) {
    ic = ics[i];
    ir = irs[i];

    if (l[ir][ic] == 0.0) {
      r[ir] = 0.0;
      continue;
    }

    for (j = i + 1; j < n; j++) {
      r[ir] -= l[ir][ics[j]] * r[irs[j]];
      l[ir][ics[j]] = 0.0;
    }

    r[ir] /= l[ir][ic];
    l[ir][ic] = 1.0;
  }

  if (changePivot == true) {
    vector<double> tmp = r;

    for (i = n - 1; i >= 0; i--) {
      r[ics[i]] = tmp[irs[i]];
    }
  }
}

// For linear constrains (II) x = c, where I is identify matrix and x has n*2
// dimention.
// This constrains represent \alpha+\beta=C in L1 sscw.
void QPsolver::solveWithIILinearConstraint(vector<vector<double> > &q,
                                           vector<double> &v, vector<double> &x,
                                           double C) {
  return;
}

// L2 sscw solver.
void QPsolver::solveWithoutLinearConstraint(vector<vector<double> > &q,
                                            vector<double> &v,
                                            vector<double> &x) {

  int size = v.size();
  double step, xz, BETA = 999999;
  vector<double> z(size, 0), delta_z(size, 0), r(size, 0);
  vector<vector<double> > l;

  // set initial value
  int i = 0, j = 0;
  bool infeasible = false;
  xz = 0;
  while (i < size) {
    double tmp_z = 0;
    j = 0;
    while (j < size) {
      tmp_z += q[i][j] * x[j]; //  z = \Sigma_j q[i][j]*x[j]+v[i]
      q[i][j] = -q[i][j];
      ++j;
    }

    tmp_z += v[i];

    if (tmp_z < 0) {
      infeasible = true;
      z[i] = tmp_z; // set z[i] to 1.0E-4 in the following step
      tmp_z = x[i] * 1.0E-4;
    } else {
      z[i] = tmp_z;
      tmp_z *= x[i];
    }
    xz += tmp_z; // xz = \Sigma_i z[i]*x[i]
    if (BETA > tmp_z) {
      BETA = tmp_z;
    }
    ++i;
  }

  if (xz <= EPSILON) {
    if (!infeasible) {
      return;
    }
    xz = EPSILON;
  }

  BETA = 1 - (BETA * size) / xz;
  if (BETA < 0.9) {
    BETA = 0.9;
  }

  int k = 0;
  float gamma = ((float)size) / (size + sqrt(size));
  float tmp_gamma = gamma;
  if (!infeasible) {
    while (xz > EPSILON && k < ITER_MAX) {
      double mu = gamma * xz / size;
      gamma = 1.0 / xz;

      // prepare gauss elimnation
      l = q;

      i = 0;
      while (i < size) {
        double tmp = 1 / x[i];
        l[i][i] -= z[i] * tmp;
        r[i] = (x[i] * z[i] - mu) * tmp; // -(-x[i]*z[i]+mu)*tmp
        ++i;
      }

      // run gauss elimination
      gaussElimination(l, r, size);

      // calculate delta z and determine step size
      step = 0.0;
      double tmp_step = 0.0;
      double acc_a = 0.0;
      double acc_b = 0.0;
      double acc_c = 0.0;

      double max_step = 9999999;
      i = 0;
      while (i < size) {
        double tmp_dx = r[i]; // where r means delta_x;
        double tmp_dz = (-x[i] * z[i] + mu - z[i] * tmp_dx) / x[i];
        delta_z[i] = tmp_dz;

        tmp_step = -x[i] / tmp_dx;
        if (max_step > tmp_step && 0 <= tmp_step) {
          max_step = tmp_step;
        }

        tmp_step = -z[i] / tmp_dz;
        if (max_step > tmp_step && 0 <= tmp_step) {
          max_step = tmp_step;
        }

        acc_a += tmp_dx * tmp_dz;
        acc_b += x[i] * tmp_dz + tmp_dx * z[i];
        acc_c += x[i] * z[i];
        ++i;
      }

      step = -acc_b / (2 * acc_a); // opt position

      double max_step1 = 999999;
      double max_step2 = 999999;
      double min_step1 = 0;
      double min_step2 = 0;
      bool stopFlag = false;
      i = 0;
      while (i < size) {
        double tmp_dx = r[i]; // where r means delta_x;
        double tmp_dz = delta_z[i];

        double a = size * tmp_dx * tmp_dz - (1 - BETA) * acc_a;
        double b = size * (x[i] * tmp_dz + tmp_dx * z[i]) - (1 - BETA) * acc_b;
        double c = size * x[i] * z[i] - (1 - BETA) * acc_c;

        if (c < 0) { // Calculation error
          c = 0.0;
        }

        double d = b * b - 4 * a * c;
        if (d < 0) {
          if (a < 0) {
            stopFlag = true;
            break;
          }
          ++i;
          continue;
        }

        if (a != 0.0) {
          d = sqrt(d);

          double tmp_right = (-b - d) / (2.0 * a);
          double tmp_left = (-b + d) / (2.0 * a);
          if (tmp_right < tmp_left) {
            tmp_step = tmp_right;
            tmp_right = tmp_left;
            tmp_left = tmp_step;
          }

          if (tmp_left > 0) {
            if (a < 0 && min_step1 < tmp_left) { // concave
              min_step1 = tmp_left;
              if (min_step2 < tmp_left) {
                min_step2 = tmp_left;
              }
            } else if (a > 0 && max_step2 > tmp_left) { // convex
              max_step2 = tmp_left;
            }
          }

          if (tmp_right <= 0) {
            if (a < 0) {
              stopFlag = true;
              break;
            }
          } else {
            if (a < 0 && max_step1 > tmp_right) { // concave
              max_step1 = tmp_right;
              if (max_step2 > tmp_right) {
                max_step2 = tmp_right;
              }
            } else if (a > 0 && min_step2 < tmp_right) { // convex
              min_step2 = tmp_right;
            }
          }
        } else {
          tmp_step = -c / b;
          if (c < 0 && b < 0) {
            stopFlag = true;
            break;
          } else if (c > 0 && b < 0) {  // right down
            if (max_step1 > tmp_step) { // right down
              max_step1 = tmp_step;
              if (max_step2 > tmp_step) {
                max_step2 = tmp_step;
              }
            }
          } else if (c < 0 && b > 0 && min_step1 < tmp_step) { // right up
            min_step1 = tmp_step;
            if (min_step2 < tmp_step) {
              min_step2 = tmp_step;
            }
          }
        }

        ++i;
      }

      tmp_step = max_step * 0.99;
      if (tmp_step < step) {
        step = tmp_step;
      }

      if (max_step1 < min_step2 && max_step2 < min_step1) {
        stopFlag = true;
      } else if (max_step1 < min_step2) {
        if (max_step2 < step) {
          step = max_step2;
        } else if (step < min_step1) {
          step = min_step1;
          if (tmp_step < min_step1) {
            tmp_step = max_step * 0.9999;
          }
        }
      } else if (max_step2 < min_step1) {
        if (max_step1 < step) {
          step = max_step1;
        } else if (step < min_step2) {
          step = min_step2;
          if (tmp_step < min_step2) {
            tmp_step = max_step * 0.9999;
          }
        }
      } else {
        if (min_step1 > step) {
          step = min_step1;
          if (tmp_step < min_step1) {
            tmp_step = max_step * 0.9999;
          }
        } else if (max_step1 < step) {
          step = max_step1;
        } else if (max_step2 < step && min_step2 > step) {
          if (step - max_step2 < min_step2 - step || tmp_step < min_step2) {
            step = max_step2;
          } else {
            step = min_step2;
          }
        }
      }

      if (tmp_step < step) {
        step = tmp_step;
      }

      if (stopFlag) { // Can not exist in new neighborhood
        cerr << "Warrning: Can not exist in new neighborhood." << endl;
        break;
      }

      // calculate update x, z
      i = 0;
      xz = 0;
      while (i < size) {
        x[i] += step * r[i];
        z[i] += step * delta_z[i];
        xz += x[i] * z[i];
        ++i;
      }

      gamma *= xz;
      if (gamma < 0.99) {
        gamma *= gamma * gamma;
      } else {
        gamma = tmp_gamma;
      }

      ++k;
    }
    if (ITER_MAX == k) {
      cerr << "Reach ITER_MAX" << endl;
    }

  } else { // infeasible point

    // calculation -qx+z-v. Now qx+v is include z.
    double _qxz_v = 0.0;
    vector<double> init_v(size, 0);
    i = 0;
    while (i < size) {
      if (z[i] <= 0) {
        init_v[i] = 1.0E-4 - z[i]; // z - q x -c
        r[i] = -init_v[i];         // q x - z + c
        z[i] = 1.0E-4;
        _qxz_v += init_v[i] * init_v[i];
      } else {
        r[i] = 0;
      }
      ++i;
    }

    _qxz_v = sqrt(_qxz_v);
    double theta_k = 1;
    double init_mu = xz / size;
    while ((xz > EPSILON || _qxz_v > EPSILON) && k < ITER_MAX) {
      double theta = gamma * theta_k;
      double mu = theta * init_mu;

      gamma = 1.0 / xz;

      // prepare gauss elimnation
      l = q;

      i = 0;
      while (i < size) {
        double tmp = 1 / x[i];
        l[i][i] -= z[i] * tmp;
        r[i] += theta * init_v[i] + (x[i] * z[i] - mu) * tmp; // q x - z + c +
                                                              // theta*\bar{c} -
                                                              // ( -x[i]*z[i]+mu
                                                              // ) * tmp

        ++i;
      }

      // run gauss elimination
      gaussElimination(l, r, size);

      // calculate delta z and determine step size
      step = 0.0;
      xz = 0.0; // where xz meanz \Sigma_i delta_x_i * delta_z_i
      double tmp_step = 0.0;
      double max_step = 1;
      double max_step1 = 1;
      double max_step2 = 1;
      double min_step1 = 0;
      double min_step2 = 0;
      bool stopFlag = false;

      i = 0;
      while (i < size) {
        double tmp_dx = r[i]; // where r means delta_x;
        double tmp_dz = (-x[i] * z[i] + mu - z[i] * tmp_dx) / x[i];
        delta_z[i] = tmp_dz;

        tmp_step = -x[i] / tmp_dx;
        if (max_step > tmp_step && 0 <= tmp_step) {
          max_step = tmp_step;
        }

        tmp_step = -z[i] / tmp_dz;
        if (max_step > tmp_step && 0 <= tmp_step) {
          max_step = tmp_step;
        }

        double a = tmp_dx * tmp_dz;
        xz += a;

        double b = x[i] * tmp_dz + tmp_dx * z[i];
        step += b;
        b -= (1 - BETA) * (theta * init_mu - theta_k * init_mu);

        double c = x[i] * z[i] - (1 - BETA) * theta_k * init_mu;
        if (c < 0) { // Calculation error
          c = 0.0;
        }

        double d = b * b - 4 * a * c;
        if (d < 0) {
          if (a < 0) {
            stopFlag = true;
            break;
          }
          ++i;
          continue;
        }

        if (a != 0.0) {
          d = sqrt(d);

          double tmp_right = (-b - d) / (2.0 * a);
          double tmp_left = (-b + d) / (2.0 * a);
          if (tmp_right < tmp_left) {
            tmp_step = tmp_right;
            tmp_right = tmp_left;
            tmp_left = tmp_step;
          }

          if (tmp_left > 0) {
            if (a < 0 && min_step1 < tmp_left) { // concave
              min_step1 = tmp_left;
              if (min_step2 < tmp_left) {
                min_step2 = tmp_left;
              }
            } else if (a > 0 && max_step2 > tmp_left) { // convex
              max_step2 = tmp_left;
            }
          }

          if (tmp_right <= 0) {
            if (a < 0) {
              stopFlag = true;
              break;
            }
          } else {
            if (a < 0 && max_step1 > tmp_right) { // concave
              max_step1 = tmp_right;
              if (max_step2 > tmp_right) {
                max_step2 = tmp_right;
              }
            } else if (a > 0 && min_step2 < tmp_right) { // convex
              min_step2 = tmp_right;
            }
          }
        } else {
          tmp_step = -c / b;
          if (c < 0 && b < 0) {
            stopFlag = true;
            break;
          } else if (c > 0 && b < 0) {  // right down
            if (max_step1 > tmp_step) { // right down
              max_step1 = tmp_step;
              if (max_step2 > tmp_step) {
                max_step2 = tmp_step;
              }
            }
          } else if (c < 0 && b > 0 && min_step1 < tmp_step) { // right up
            min_step1 = tmp_step;
            if (min_step2 < tmp_step) {
              min_step2 = tmp_step;
            }
          }
        }

        ++i;
      }

      tmp_step = -theta_k / (theta - theta_k);
      if (max_step > tmp_step) {
        max_step = tmp_step;
      }

      step = -step / (2 * xz); // opt position

      tmp_step = max_step * 0.99;
      // if(tmp_step<step){
      step = tmp_step;
      //}

      if (max_step1 < min_step2 && max_step2 < min_step1) {
        stopFlag = true;
      } else if (max_step1 < min_step2) {
        if (max_step2 < step) {
          step = max_step2;
        } else if (step < min_step1) {
          step = min_step1;
          if (tmp_step < min_step1) {
            tmp_step = max_step * 0.9999;
          }
        }
      } else if (max_step2 < min_step1) {
        if (max_step1 < step) {
          step = max_step1;
        } else if (step < min_step2) {
          step = min_step2;
          if (tmp_step < min_step2) {
            tmp_step = max_step * 0.9999;
          }
        }
      } else {
        if (min_step1 > step) {
          step = min_step1;
          if (tmp_step < min_step1) {
            tmp_step = max_step * 0.9999;
          }
        } else if (max_step1 < step) {
          step = max_step1;
        } else if (max_step2 < step && min_step2 > step) {
          if (step - max_step2 < min_step2 - step || tmp_step < min_step2) {
            step = max_step2;
          } else {
            step = min_step2;
          }
        }
      }

      if (tmp_step < step) {
        step = tmp_step;
      }

      if (stopFlag) { // Can not exist in new neighborhood
        break;
      }

      // calculate update x, z
      i = 0;
      xz = 0;
      while (i < size) {
        x[i] += step * r[i];
        z[i] += step * delta_z[i];
        xz += x[i] * z[i];
        ++i;
      }

      i = 0;
      _qxz_v = 0;
      while (i < size) {
        double tmp = 0;
        j = 0;
        while (j < size) {
          // -q[i][j]
          tmp += q[i][j] * x[j]; // -q x
          ++j;
        }

        tmp += z[i] - v[i]; // -q x + z - v
        r[i] = -tmp;        // q x - z + v
        _qxz_v += tmp * tmp;
        ++i;
      }
      _qxz_v = sqrt(_qxz_v);

      theta_k += step * (theta - theta_k);

      gamma *= xz;
      if (gamma < 0.99) {
        gamma *= gamma * gamma;
      } else {
        gamma = tmp_gamma;
      }

      ++k;
    }
  }
}

