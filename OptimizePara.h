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

#include <vector>
#include <iostream>
#include "utility.h"

using namespace std;

class OptimizePara {
public:
  typedef struct range {
    float upper; // no restrict if upper==lower
    float lower;
  } Range;

protected:
  float GOLD;
  float NONZERO;
  float ITMAX;
  float P_ITMAX;
  float CGOLD;
  float ZEPS;
  float TOL;

  float ax;
  float bx;
  float cx;
  float fa;
  float fb;
  float fc;
  float fpos;
  float wmin;

  float mkRestrictedPosition(const vector<float> &pos, float &w,
                             const float *dir, vector<float> &tmpPos,
                             const vector<Range> &ranges) {
    unsigned short size = pos.size();
    tmpPos.resize(size);
    float over = 0;
    for (unsigned short i = 0; i < size; ++i) {
      tmpPos[i] = pos[i] + w * dir[i];
      if (ranges[i].upper != ranges[i].lower) {
        float tmp;
        if ((tmp = tmpPos[i] - ranges[i].upper) > 0.0) {

          if (ABS(tmp) > ABS(over)) {
            over = tmp;
          }
        } else if ((tmp = tmpPos[i] - ranges[i].lower) < 0.0) {

          if (ABS(tmp) > ABS(over)) {
            over = tmp;
          }
        }
      }
    }

    if (over != 0) {
      w -= over;
      for (unsigned short i = 0; i < size; ++i) {
        tmpPos[i] = pos[i] + w * dir[i];
      }
    }

    return over;
  }

  void mkPosition(const vector<float> &pos, float w, const float *dir,
                  vector<float> &tmpPos) {
    unsigned short size = pos.size();
    tmpPos.resize(size);
    for (unsigned short i = 0; i < size; ++i) {
      tmpPos[i] = pos[i] + w * dir[i];
    }
  }

public:
  OptimizePara(void)
      : GOLD(1.618034), NONZERO(1.0E-20), ITMAX(20), P_ITMAX(40),
        CGOLD(0.3819660), ZEPS(2.0E-4), TOL(2.0E-4) {}

  ~OptimizePara(void) {}

  template <typename F>
  float powell(F &func, vector<float> &pos, const vector<Range> &ranges,
               float initStep) {

    unsigned short size = pos.size();
    float **dires;
    if ((dires = (float **)malloc(sizeof(float *) * size +
                                  sizeof(float) * size * (size + 1))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    for (unsigned short i = 0; i < size; ++i) {
      dires[i] = ((float *)(dires + size)) + size * i;
      for (unsigned short j = 0; j < size; ++j) {
        if (i == j) {
          dires[i][j] = 1;
        } else {
          dires[i][j] = 0;
        }
      }
    }

    float *dir = dires[size - 1] + size;

    char iter = 0;
    vector<float> post(pos);
    vector<float> postt;
    postt.resize(size);

    float fpos = func(pos);
    float fpost;
    float fpostt;

    while (1) {
      fpost = fpos;
      unsigned short ibig = 0;
      float del = 0.0;
      for (unsigned short i = 0; i < size; ++i) {
        fpostt = fpos;
        fa = fpos;

        ax = 0.0;
        bx = initStep;
        float *tmpd = dires[i];
        seekRange(func, pos, tmpd, ranges, 0);
        brent(func, pos, tmpd);

        for (unsigned short j = 0; j < size; ++j) {
          pos[j] += wmin * tmpd[j];
        }

        if (ABS(fpostt - fpos) > del) {
          del = ABS(fpostt - fpos);
          ibig = i;
        }
      }

      if (2.0 * ABS(fpost - fpos) <= TOL * (ABS(fpost) + ABS(fpos))) {
        return fpos;
      }

      if (iter >= P_ITMAX) {
        cerr << "WARNING:Reached the maximum number of iteration in powell\n";
        return fpos;
      }

      for (unsigned short i = 0; i < size; ++i) {
        postt[i] = 2.0 * pos[i] - post[i];
        dir[i] = pos[i] - post[i];
        post[i] = pos[i];
      }

      fpostt = func(postt);
      if (fpostt < fpost) {
        float tmp1 = fpost - fpos - del;
        tmp1 = 2.0 * (fpost - 2.0 * fpos + fpostt) * tmp1 * tmp1;
        float tmp2 = fpost - fpostt;
        if (tmp1 - del * tmp2 * tmp2 < 0.0) {
          fa = fpos;
          ax = 0.0;
          bx = initStep;
          seekRange(func, pos, dir, ranges, 0);
          brent(func, pos, dir);

          float *delDir = dires[ibig];
          float *aliveDir = dires[size - 1];
          for (unsigned short i = 0; i < size; ++i) {
            delDir[i] = aliveDir[i];
            aliveDir[i] = dir[i];
            pos[i] += wmin * dir[i];
          }
        }
      }

      iter++;
    }

    free(dires);
  }

  template <typename F>
  void seekRange(F &func, vector<float> &pos, const float *dir,
                 const vector<Range> &ranges, char calcfaFlag) {

    float tmp = 0;
    float fu;

    vector<float> tmpPos;
    if (calcfaFlag) {
      tmp = mkRestrictedPosition(pos, ax, dir, tmpPos, ranges);
      fa = func(tmpPos);
    }

    if (tmp > 0) {
      bx = -bx;
      tmp = mkRestrictedPosition(pos, bx, dir, tmpPos, ranges);
      fb = func(tmpPos);

      if (fb > fa) {
        cx = bx;
        fc = fb;

        bx = cx + CGOLD * (ax - cx);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      } else if (tmp != 0) {
        cx = bx;
        fc = fb;

        bx = ax - CGOLD * (ax - cx);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      }
    } else if (tmp < 0) {
      tmp = mkRestrictedPosition(pos, bx, dir, tmpPos, ranges);
      fb = func(tmpPos);

      if (fb > fa) {
        cx = bx;
        fc = fb;

        bx = cx + CGOLD * (ax - cx);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      } else if (tmp != 0) {
        cx = bx;
        fc = fb;

        bx = ax - CGOLD * (ax - cx);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      }
    } else {
      tmp = mkRestrictedPosition(pos, bx, dir, tmpPos, ranges);
      if (ax == bx) {
        bx = -bx;
        tmp = mkRestrictedPosition(pos, bx, dir, tmpPos, ranges);
        fb = func(tmpPos);

        if (fb > fa) {
          cx = bx;
          fc = fb;

          bx = cx + CGOLD * (ax - cx);
          mkPosition(pos, bx, dir, tmpPos);
          fb = func(tmpPos);

          return;
        } else if (tmp != 0) {
          cx = bx;
          fc = fb;

          bx = ax - CGOLD * (ax - cx);
          mkPosition(pos, bx, dir, tmpPos);
          fb = func(tmpPos);

          return;
        }
      } else {

        fb = func(tmpPos);

        if (fb > fa) {
          tmp = ax;
          ax = bx;
          bx = tmp;

          tmp = fa;
          fa = fb;
          fb = tmp;
        } else if (tmp != 0) {
          cx = bx;
          fc = fb;

          bx = ax + CGOLD * (cx - ax);
          mkPosition(pos, bx, dir, tmpPos);
          fb = func(tmpPos);

          return;
        }
      }
    }

    float tmpfa = fa;
    float tmpfb = fb;
    float tmpax = ax;
    float tmpbx = bx;
    char tmpfaFlag = 0;
    if (fb == fa) {
      tmpfaFlag = 1;
    }

    cx = bx + GOLD * (bx - ax);
    tmp = mkRestrictedPosition(pos, cx, dir, tmpPos, ranges);
    if (tmp != 0) {
      if (cx == bx) {
        cx = bx;
        fc = fb;

        bx = ax + CGOLD * (cx - ax);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      } else {
        fc = func(tmpPos);
        return;
      }
    }

    fc = func(tmpPos);

    while (fb > fc) {
      float r = (bx - ax) * (fb - fc);
      float q = (bx - cx) * (fb - fa);
      tmp = SIGN(MAX(ABS(q - r), NONZERO), q - r);
      float u = bx - ((bx - cx) * q - (bx - ax) * r) / (2.0 * tmp);
      if ((bx - u) * (u - cx) > 0.0) {
        mkPosition(pos, u, dir, tmpPos);
        fu = func(tmpPos);
        if (fu < fc) {
          ax = bx;
          bx = u;

          fa = fb;
          fb = fu;
          return;
        } else if (fu > fb) {
          bx = u;
          fb = fu;

          u = cx + GOLD * (cx - bx);
          tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
          if (tmp != 0) {
            if (u == cx) {
              ax = bx;
              fa = fb;

              bx = ax + CGOLD * (cx - ax);
              mkPosition(pos, bx, dir, tmpPos);
              fb = func(tmpPos);

              return;
            } else {
              ax = bx;
              bx = cx;
              cx = u;

              fa = fb;
              fb = fc;
              fc = func(tmpPos);
              return;
            }
          }
          fu = func(tmpPos);
        } else {
          tmp = u;
          u = cx;
          cx = tmp;

          tmp = fu;
          fu = fc;
          fc = tmp;
        }
      } else if ((bx - cx) * (cx - u) > 0.0) {
        tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
        if (tmp != 0) {
          if (u == cx) {
            ax = bx;
            fa = fb;

            bx = ax + CGOLD * (cx - ax);
            mkPosition(pos, bx, dir, tmpPos);
            fb = func(tmpPos);

            return;
          } else {
            ax = bx;
            bx = cx;
            cx = u;

            fa = fb;
            fb = fc;
            fc = func(tmpPos);
            return;
          }
        }
        fu = func(tmpPos);
        if (fu < fc) {
          bx = cx;
          cx = u;
          u = cx + GOLD * (cx - bx);

          fb = fc;
          fc = fu;
          tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
          if (tmp != 0) {
            if (u == cx) {
              ax = bx;
              fa = fb;

              bx = ax + CGOLD * (cx - ax);
              mkPosition(pos, bx, dir, tmpPos);
              fb = func(tmpPos);

              return;
            } else {
              ax = bx;
              bx = cx;
              cx = u;

              fa = fb;
              fb = fc;
              fc = func(tmpPos);
              return;
            }
          }

          fu = func(tmpPos);
        }
      } else {
        u = cx + GOLD * (cx - bx);
        tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
        if (tmp != 0) {
          if (u == cx) {
            ax = bx;
            fa = fb;

            bx = ax + CGOLD * (cx - ax);
            mkPosition(pos, bx, dir, tmpPos);
            fb = func(tmpPos);

            return;
          } else {
            ax = bx;
            bx = cx;
            cx = u;

            fa = fb;
            fb = fc;
            fc = func(tmpPos);
            return;
          }
        }
        fu = func(tmpPos);
      }

      ax = bx;
      bx = cx;
      cx = u;

      fa = fb;
      fb = fc;
      fc = fu;
    }

    if (fa != fb || fa != fc || !tmpfaFlag) {
      return;
    }

    fa = tmpfb;
    fb = tmpfa;
    ax = tmpbx;
    bx = tmpax;

    cx = bx + GOLD * (bx - ax);
    tmp = mkRestrictedPosition(pos, cx, dir, tmpPos, ranges);
    if (tmp != 0) {
      if (cx == bx) {
        cx = bx;
        fc = fb;

        bx = ax + CGOLD * (cx - ax);
        mkPosition(pos, bx, dir, tmpPos);
        fb = func(tmpPos);

        return;
      } else {
        fc = func(tmpPos);
        return;
      }
    }
    fc = func(tmpPos);

    while (fb > fc) {
      float r = (bx - ax) * (fb - fc);
      float q = (bx - cx) * (fb - fa);
      tmp = SIGN(MAX(ABS(q - r), NONZERO), q - r);
      float u = bx - ((bx - cx) * q - (bx - ax) * r) / (2.0 * tmp);
      if ((bx - u) * (u - cx) > 0.0) {
        mkPosition(pos, u, dir, tmpPos);
        fu = func(tmpPos);
        if (fu < fc) {
          ax = bx;
          bx = u;

          fa = fb;
          fb = fu;
          return;
        } else if (fu > fb) {
          bx = u;
          fb = fu;

          u = cx + GOLD * (cx - bx);
          tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
          if (tmp != 0) {
            if (u == cx) {
              ax = bx;
              fa = fb;

              bx = ax + CGOLD * (cx - ax);
              mkPosition(pos, bx, dir, tmpPos);
              fb = func(tmpPos);

              return;
            } else {
              ax = bx;
              bx = cx;
              cx = u;

              fa = fb;
              fb = fc;
              fc = func(tmpPos);
              return;
            }
          }
          fu = func(tmpPos);
        } else {
          tmp = u;
          u = cx;
          cx = tmp;

          tmp = fu;
          fu = fc;
          fc = tmp;
        }
      } else if ((bx - cx) * (cx - u) > 0.0) {
        tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
        if (tmp != 0) {
          if (u == cx) {
            ax = bx;
            fa = fb;

            bx = ax + CGOLD * (cx - ax);
            mkPosition(pos, bx, dir, tmpPos);
            fb = func(tmpPos);

            return;
          } else {
            ax = bx;
            bx = cx;
            cx = u;

            fa = fb;
            fb = fc;
            fc = func(tmpPos);
            return;
          }
        }
        fu = func(tmpPos);
        if (fu < fc) {
          bx = cx;
          cx = u;
          u = cx + GOLD * (cx - bx);

          fb = fc;
          fc = fu;
          tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
          if (tmp != 0) {
            if (u == cx) {
              ax = bx;
              fa = fb;

              bx = ax + CGOLD * (cx - ax);
              mkPosition(pos, bx, dir, tmpPos);
              fb = func(tmpPos);

              return;
            } else {
              ax = bx;
              bx = cx;
              cx = u;

              fa = fb;
              fb = fc;
              fc = func(tmpPos);
              return;
            }
          }

          fu = func(tmpPos);
        }
      } else {
        u = cx + GOLD * (cx - bx);
        tmp = mkRestrictedPosition(pos, u, dir, tmpPos, ranges);
        if (tmp != 0) {
          if (u == cx) {
            ax = bx;
            fa = fb;

            bx = ax + CGOLD * (cx - ax);
            mkPosition(pos, bx, dir, tmpPos);
            fb = func(tmpPos);

            return;
          } else {
            ax = bx;
            bx = cx;
            cx = u;

            fa = fb;
            fb = fc;
            fc = func(tmpPos);
            return;
          }
        }
        fu = func(tmpPos);
      }

      ax = bx;
      bx = cx;
      cx = u;

      fa = fb;
      fb = fc;
      fc = fu;
    }
  }

  template <typename F>
  void brent(F &func, vector<float> &pos, const float *dir) {

    float a;
    float b;
    if (ax < cx) {
      a = ax;
      b = cx;
    } else {
      a = cx;
      b = ax;
    }

    float x;
    float w;
    float v;
    float fx;
    float fw;
    float fv;

    if (fb <= fa) {
      if (fb <= fc) {
        x = bx;
        fx = fb;
        if (fa >= fc) {
          w = cx;
          fw = fc;
          v = ax;
          fv = fa;
        } else {
          w = ax;
          fw = fa;
          v = cx;
          fv = fc;
        }
      } else {
        x = cx;
        fx = fc;
        w = bx;
        fw = fb;
        v = ax;
        fv = fa;
      }
    } else if (fa <= fc) {
      x = ax;
      fx = fa;
      if (fb >= fc) {
        w = cx;
        fw = fc;
        v = bx;
        fv = fb;
      } else {
        w = bx;
        fw = fb;
        v = cx;
        fv = fc;
      }
    } else {
      x = cx;
      fx = fc;
      w = ax;
      fw = fa;
      v = bx;
      fv = fb;
    }

    float u;
    float fu;

    float iter = 0;
    float xm = 0.5 * (a + b);
    float oldest_d;
    float older_d = (x >= xm) ? a - x : b - x;
    float d = CGOLD * older_d;
    float flatCount = 0;
    vector<float> tmpPos;

    while (iter < ITMAX) {
      float tol1 = TOL * ABS(x) + ZEPS;
      float tol2 = 2.0 * tol1;
      if (ABS(x - xm) <= (tol2 - 0.5 * (b - a))) {
        wmin = x;
        fpos = fx;
        return;
      }

      if (fx == fw && fx == fv) {
        flatCount++;
      }

      if (flatCount >= 2) {
        wmin = x;
        fpos = fx;
        return;
      }

      if (ABS(older_d) > tol1) {
        float r = (x - w) * (fx - fv);
        float q = (x - v) * (fx - fw);
        float p = (x - v) * q - (x - w) * r;

        // check convex
        // avoid zero division
        float convex =
            (r - q) * ((x * x - v * v) * (x - w) - (x * x - w * w) * (x - v));

        q = 2.0 * (q - r);
        if (q > 0.0) {
          p = -p;
        }
        q = ABS(q);
        oldest_d = older_d;
        older_d = d;
        if (convex <= 0 || ABS(p) > ABS(0.5 * q * oldest_d) ||
            p <= q * (a - x) ||
            p >= q * (b - x)) { // limit next point to between a and b

          older_d = (x >= xm) ? a - x : b - x;
          d = CGOLD * older_d;

        } else {
          d = p / q;
          u = x + d;
          if (u - a < tol2 || b - u < tol2) {
            d = SIGN(tol1, xm - x);
          }
        }
      } else {
        older_d = (x >= xm) ? a - x : b - x;
        d = CGOLD * older_d;
      }

      u = (ABS(d) >= tol1) ? x + d : x + SIGN(tol1, d);
      mkPosition(pos, u, dir, tmpPos);
      fu = func(tmpPos);

      if (fu == fx && fu == fw) {
        if (ABS(v - a) < ABS(v - b)) {
          if (u < x) {
            b = x;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
          } else {
            b = u;
            w = u;
            fw = fu;
          }
        } else {
          if (u < x) {
            a = u;
            w = u;
            fw = fu;
          } else {
            a = x;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
          }
        }

        older_d = 0; // next golden
      } else if (fu <= fx) {
        if (u >= x) {
          a = x;
        } else {
          b = x;
        }
        v = w;
        w = x;
        x = u;

        fv = fw;
        fw = fx;
        fx = fu;
      } else {
        if (u < x) {
          a = u;
        } else {
          b = u;
        }

        if (fu <= fw) {
          v = w;
          w = u;
          fv = fw;
          fw = fu;
        } else if (fu <= fv) {
          v = u;
          fv = fu;
        } else {
          older_d = 0; // next golden
        }
      }
      xm = 0.5 * (a + b);
      iter++;
    }

    cerr << "WARNING:Reached the maximum number of iteration in brent.\n";
    wmin = x;
    fpos = fx;
  }
};
