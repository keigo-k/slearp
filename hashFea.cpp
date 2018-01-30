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

#include "hashFea.h"
#include <iostream>
#include <fstream>

hashFea::hashFea() : FeaTable(NULL) {}

hashFea::~hashFea() {
  if (FeaTable == NULL) {
    return;
  }

  Fea *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (Fea *)FeaTable[i][j]; p != NULL; p = (Fea *)tmp) {
        tmp = (Fea *)p->next;
        free(p);
      }
    }
  }
  free(FeaTable);
}

void hashFea::initialize(unsigned int row, unsigned int col) {
  unsigned int i, j;

  numFea = 0;
  hashRow = row;
  hashCol = col;

  if ((FeaTable = (void ***)malloc(row * sizeof(void **) +
                                   row * col * sizeof(void *))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < row; ++i) {
    FeaTable[i] = (void **)(FeaTable + row) + i * col;
    for (j = 0; j < col; ++j) {
      FeaTable[i][j] = NULL;
    }
  }
}

inline void **hashFea::hash(const char *targetfea) {
  const char *tmp = targetfea;
  void **p;
  unsigned int value = 0;
  unsigned short len = 0;
  char c;
  while ((c = *targetfea) != '\0') {
    value += c * (63 << len);
    ++targetfea;
    ++len;
  }
  p = FeaTable[value % hashRow];

  while ((c = *tmp) != '\0') {
    --len;
    value += c * (31 << len);
    ++tmp;
  }

  return &(p[value % hashCol]);
}

inline int hashFea::keyequal(const char *fea, const char *targetfea) {

  char c;
  while ((c = *targetfea) != '\0') {
    if (*fea != c) {
      return 0;
    }
    ++fea;
    ++targetfea;
  }
  if (*fea != '\0') {
    return 0;
  }

  return 1;
}

Fea *hashFea::get(const string &target) {
  const char *targetfea = target.c_str();
  Fea **pp = (Fea **)hash(targetfea);
  Fea *p = *pp;

  for (; p != NULL; p = (Fea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (Fea *)malloc(sizeof(Fea) + sizeof(char) * size)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;

  ++numFea;
  return p;
}

Fea *hashFea::refer(const string &target) {
  const char *targetfea = target.c_str();
  Fea **pp = (Fea **)hash(targetfea);
  Fea *p = *pp;

  for (; p != NULL; p = (Fea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p;
    }
  }

  return NULL;
}

ParaVal hashFea::getParaVal(const string &target) {
  const char *targetfea = target.c_str();

  Fea **pp = (Fea **)hash(targetfea);
  Fea *p = *pp;

  for (; p != NULL; p = (Fea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p->w;
    }
  }

  return 0;
}

void hashFea::writeFeatures(const char *write) {
  ofstream WRITE;
  WRITE.open(write, ios_base::trunc);
  if (!WRITE) {
    cerr << "ERROR:Can not write model: " << write << endl;
    exit(EXIT_FAILURE);
  } else {
    Fea *p;
    unsigned int i, j;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (Fea *)FeaTable[i][j]; p != NULL; p = (Fea *)p->next) {
          WRITE << p->fea;
          WRITE << "\t";
          WRITE << p->w;
          WRITE << "\n";
        }
      }
    }

    WRITE.close();
  }
}

hashCrfceFea::hashCrfceFea() : hashFea() {}

hashCrfceFea::~hashCrfceFea() {}

void hashCrfceFea::initialize(unsigned int row, unsigned int col,
                              unsigned short numgrad) {
  unsigned int i, j;

  numFea = 0;
  hashRow = row;
  hashCol = col;
  numGrad = numgrad;

  if ((FeaTable =
           (void ***)malloc(row * sizeof(void **) + row * col * sizeof(void *) +
                            sizeof(float) * numGrad * 2)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < row; ++i) {
    FeaTable[i] = (void **)(FeaTable + row) + i * col;
    for (j = 0; j < col; ++j) {
      FeaTable[i][j] = NULL;
    }
  }

  if (numGrad > 0) {
    a = (float *)(&(FeaTable[row - 1][col - 1]) + 1);
    for (i = 0; i < numGrad; i++) {
      a[i] = 0;
    }
    r = a + numGrad;
    for (i = 0; i < numGrad; i++) {
      r[i] = 0;
    }
  } else {
    a = NULL;
    r = NULL;
  }
}

crfceFea *hashCrfceFea::get(const string &target) {
  const char *targetfea = target.c_str();
  crfceFea **pp = (crfceFea **)hash(targetfea);
  crfceFea *p = *pp;

  for (; p != NULL; p = (crfceFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (crfceFea *)malloc(sizeof(crfceFea) + sizeof(char) * size +
                              sizeof(ParaVal) * numGrad * 2)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;
  p->tmpW = 0;
  p->g = 0;
  p->tmpG = 0;
  p->unSupG = 0;
  p->dir = 0;

  if (numGrad > 0) {
    p->y = (ParaVal *)(p->fea + size);
    p->s = p->y + numGrad;
    for (int i = 0; i < numGrad; i++) {
      p->y[i] = 0;
      p->s[i] = 0;
    }
  } else {
    p->y = NULL;
    p->s = NULL;
  }

  ++numFea;
  return p;
}

void hashCrfceFea::setWeight(const string &target, ParaVal val) {
  const char *targetfea = target.c_str();
  crfceFea **pp = (crfceFea **)hash(targetfea);
  crfceFea *p = *pp;

  for (; p != NULL; p = (crfceFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      p->w = val;
      return;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (crfceFea *)malloc(sizeof(crfceFea) + sizeof(char) * size +
                              sizeof(ParaVal) * numGrad * 2)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = val;
  p->tmpW = 0;
  p->g = 0;
  p->tmpG = 0;
  p->unSupG = 0;
  p->dir = 0;

  if (numGrad > 0) {
    p->y = (ParaVal *)(p->fea + size);
    p->s = p->y + numGrad;
    for (int i = 0; i < numGrad; i++) {
      p->y[i] = 0;
      p->s[i] = 0;
    }
  } else {
    p->y = NULL;
    p->s = NULL;
  }

  ++numFea;
}

void hashCrfceFea::updateWeightForSup(float rate, float C) {

  crfceFea *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
        continue;
      }
      /*
      for(tmp=p; p==tmp && p!=NULL;){
              p->w += rate*p->g;
              ParaVal sign = p->w;
              p->w -= rate * SIGN(C, sign);
              if(p->w*sign<=0){
                      tmp=(crfceFea *)p->next;
                      free(p);
                      FeaTable[i][j]=(void *)tmp;
                      p=tmp;
              }else{
                      p=(crfceFea *)p->next;
              }
      }
      */

      for (; p != NULL; p = (crfceFea *)tmp->next) {
        p->tmpG = p->g - p->w * C;
        /*
        if(p->w > 1.0E-15 || p->w < -1.0E-15){
                ParaVal sign = p->w;
                p->tmpG -= SIGN(C, sign);
        }
        */

        p->w += rate * p->tmpG;

        tmp = p;
        /*
        if(p->w*sign<=0){
                tmp->next=(void *)p->next;
                free(p);
        }else{
                tmp=p;
        }
        */
      }
    }
  }
}

void hashCrfceFea::nextSettingInIter1ForSup() {
  crfceFea *p;
  unsigned int i, j;
  if (numGrad > 0) {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->s[0] = p->w - p->tmpW;
          p->tmpW = p->w;
          // p->tmpG = p->g;
          p->g = 0;
        }
      }
    }
  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->tmpW = p->w;
          p->g = 0;
        }
      }
    }
  }
}

void hashCrfceFea::updateWeightForSupByLBGFS(float rate, float C,
                                             unsigned short storeSize) {

  crfceFea *p, *tmp;
  unsigned int i, j;
  if (numGrad > 0) {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->w = p->tmpW;

          p->dir = p->g - p->w * C;
          // p->dir = p->g;
          /*
          if(p->w > 1.0E-15 || p->w < -1.0E-15){
                  p->dir -= SIGN(C, p->w);
          }
          */
          p->y[0] = p->dir - p->tmpG;
        }
      }
    }

    unsigned short k;
    for (k = 0; k < storeSize; ++k) {
      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            a[k] += p->s[k] * p->dir;
            r[k] += p->s[k] * p->y[k];
          }
        }
      }

      if (r[k] >= 1.0E-15 || r[k] <= -1.0E-15) {
        r[k] = 1.0 / r[k];
      } else {
        cerr << "WARNING: Division by " << SIGN((float)1.0E-15, r[k])
             << " instead of division by zero in BGFS" << endl;
        r[k] = 1.0 / (SIGN((float)1.0E-15, r[k]));
      }
      a[k] *= r[k];

      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            p->dir -= a[k] * p->y[k];
          }
        }
      }
    }

    while (k != 0) {
      --k;
      float b = 0.0;
      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            b += p->y[k] * p->dir;
          }
        }
      }

      b *= r[k];

      if (k != 0) {
        for (i = 0; i < hashRow; ++i) {
          for (j = 0; j < hashCol; ++j) {
            for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
                 p = (crfceFea *)p->next) {
              p->dir += (a[k] - b) * p->s[k];
            }
          }
        }
      } else {
        for (i = 0; i < hashRow; ++i) {
          for (j = 0; j < hashCol; ++j) {
            if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
              continue;
            }
            /*
                                                            for(tmp=p; p==tmp &&
               p!=NULL;){
                                                                    p->w +=
               p->dir + (a[k]-b)*p->s[k];
                                                                    ParaVal sign
               = p->w;
                                                                    p->w -= rate
               * SIGN(C, sign);
                                                                    if(p->w*sign<=0){
                                                                            tmp=(crfceFea
               *)p->next;
                                                                            free(p);
                                                                            FeaTable[i][j]=(void
               *)tmp;
                                                                            p=tmp;
                                                                    }else{
                                                                            p=(crfceFea
               *)p->next;
                                                                    }
                                                            }
            */
            for (; p != NULL; p = (crfceFea *)tmp->next) {
              p->w += rate * (p->dir + (a[k] - b) * p->s[k]);
              /*ParaVal sign = p->w;
              p->w -= rate * SIGN(C, sign);
              if(p->w*sign<=0){
                      tmp->next=(void *)p->next;
                      free(p);
              }else{*/
              tmp = p;
              //}
            }
          }
        }
      }
    }

    for (k = 0; k < storeSize; ++k) {
      r[k] = 0;
      a[k] = 0;
    }
  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
          continue;
        }
        /*
        for(tmp=p; p==tmp && p!=NULL;){
                p->w=p->tmpW;
                p->w += rate * p->g;
                ParaVal sign = p->w;
                p->w -= rate * SIGN(C, sign);
                if(p->w*sign<=0){
                        tmp=(crfceFea *)p->next;
                        free(p);
                        FeaTable[i][j]=(void *)tmp;
                        p=tmp;
                }else{
                        p=(crfceFea *)p->next;
                }
        }
        */
        for (; p != NULL; p = (crfceFea *)tmp->next) {
          p->w = p->tmpW;

          p->dir = p->g - p->w * C;
          /*
          p->dir = p->g;
          if(p->w > 1.0E-15 || p->w < -1.0E-15){
                  p->dir -= SIGN(C, p->w);
          }
          */

          p->w += rate * p->dir;
          /*ParaVal sign = p->w;
          p->w -= rate * SIGN(C, sign);
          if(p->w*sign<=0){
                  tmp->next=(void *)p->next;
                  free(p);
          }else{*/
          tmp = p;
          //}
        }
      }
    }
  }
}

void hashCrfceFea::nextSettingForSup(float C, unsigned short storeSize) {
  crfceFea *p;
  unsigned int i, j;
  if (numGrad > 0) {
    unsigned short k;
    unsigned short currMaxIndex =
        (numGrad - 1 < storeSize) ? numGrad - 1 : storeSize;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {

          for (k = currMaxIndex; k != 0; --k) {
            p->s[k] = p->s[k - 1];
            p->y[k] = p->y[k - 1];
          }

          p->tmpG = p->g - p->tmpW * C;
          /*
          p->dir = p->g;
          if(p->tmpW > 1.0E-15 || p->tmpW < -1.0E-15){
                  p->dir -= SIGN(C, p->tmpW);
          }
          */
          // p->tmpG = p->dir;
          p->s[0] = p->w - p->tmpW;
          p->tmpW = p->w;
          p->g = 0;
        }
      }
    }

  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->tmpW = p->w;
          p->g = 0;
        }
      }
    }
  }
}

void hashCrfceFea::updateWeight(float supWeight, float unSupWeight, float rate,
                                float C) {

  crfceFea *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
        continue;
      }
      /*
      for(tmp=p; p==tmp && p!=NULL;){
              p->w = rate*(supWeight*p->g + unSupWeight*p->unSupG);
              ParaVal sign = p->w;
              p->w -= rate * SIGN(C, sign);
              if(p->w*sign<=0){
                      tmp=(crfceFea *)p->next;
                      free(p);
                      FeaTable[i][j]=(void *)tmp;
                      p=tmp;
              }else{
                      p=(crfceFea *)p->next;
              }
      }
      */

      for (; p != NULL; p = (crfceFea *)tmp->next) {
        p->tmpG = supWeight * p->g + unSupWeight * p->unSupG - p->w * C;
        p->w += rate * p->tmpG;
        tmp = p;
      }
    }
  }
}

void hashCrfceFea::nextSettingInIter1(float supWeight, float unSupWeight) {
  crfceFea *p;
  unsigned int i, j;
  if (numGrad > 0) {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->s[0] = p->w - p->tmpW;
          p->tmpW = p->w;
          p->g = 0;
          p->unSupG = 0;
        }
      }
    }
  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->tmpW = p->w;
          p->g = 0;
          p->unSupG = 0;
        }
      }
    }
  }
}

void hashCrfceFea::updateWeightByLBGFS(float supWeight, float unSupWeight,
                                       float rate, float C,
                                       unsigned short storeSize) {

  crfceFea *p, *tmp;
  unsigned int i, j;
  if (numGrad > 0) {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->w = p->tmpW;
          p->dir = supWeight * p->g + unSupWeight * p->unSupG - p->w * C;
          p->y[0] = p->dir - p->tmpG;
        }
      }
    }

    unsigned short k;
    for (k = 0; k < storeSize; ++k) {
      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            a[k] += p->s[k] * p->dir;
            r[k] += p->s[k] * p->y[k];
          }
        }
      }

      if (r[k] >= 1.0E-15 || r[k] <= -1.0E-15) {
        r[k] = 1.0 / r[k];
      } else {
        cerr << "WARNING: Division by " << SIGN((float)1.0E-15, r[k])
             << " instead of division by zero in BGFS" << endl;
        cerr << "r[" << k << "]: " << r[k] << " a[" << k << "]: " << a[k]
             << endl;
        for (i = 0; i < hashRow; ++i) {
          for (j = 0; j < hashCol; ++j) {
            for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
                 p = (crfceFea *)p->next) {
              cerr << "w: " << p->w << " g: " << supWeight *p->g
                   << " unsupg: " << unSupWeight *p->unSupG
                   << " y[k]: " << p->y[k] << " s[k]: " << p->s[k]
                   << " dir: " << p->dir << endl;
            }
          }
        }
        exit(1);
        r[k] = 1.0 / (SIGN((float)1.0E-15, r[k]));
      }
      a[k] *= r[k];

      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            p->dir -= a[k] * p->y[k];
          }
        }
      }
    }

    while (k != 0) {
      --k;
      float b = 0.0;
      for (i = 0; i < hashRow; ++i) {
        for (j = 0; j < hashCol; ++j) {
          for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
               p = (crfceFea *)p->next) {
            b += p->y[k] * p->dir;
          }
        }
      }

      b *= r[k];

      if (k != 0) {
        for (i = 0; i < hashRow; ++i) {
          for (j = 0; j < hashCol; ++j) {
            for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
                 p = (crfceFea *)p->next) {
              p->dir += (a[k] - b) * p->s[k];
            }
          }
        }
      } else {
        for (i = 0; i < hashRow; ++i) {
          for (j = 0; j < hashCol; ++j) {
            if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
              continue;
            }
            /*
            for(tmp=p; p==tmp && p!=NULL;){
                    p->w += p->dir + (a[k]-b)*p->s[k];
                    ParaVal sign = p->w;
                    p->w -= rate * SIGN(C, sign);
                    if(p->w*sign<=0){
                            tmp=(crfceFea *)p->next;
                            free(p);
                            FeaTable[i][j]=(void *)tmp;
                            p=tmp;
                    }else{
                            p=(crfceFea *)p->next;
                    }
            }
            */

            for (; p != NULL; p = (crfceFea *)tmp->next) {
              p->w += rate * (p->dir + (a[k] - b) * p->s[k]);
              /*ParaVal sign = p->w;
              p->w -= rate * SIGN(C, sign);
              if(p->w*sign<=0){
                      tmp->next=(void *)p->next;
                      free(p);
              }else{*/
              tmp = p;
              //}
            }
          }
        }
      }
    }

    for (i = 0; i < storeSize; ++i) {
      r[i] = 0;
      a[i] = 0;
    }
  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        if ((p = (crfceFea *)FeaTable[i][j]) == NULL) {
          continue;
        }
        /*
        for(tmp=p; p==tmp && p!=NULL;){
                p->w=p->tmpW;
                p->w += rate * (supWeight*p->g + unSupWeight*p->unSupG);
                ParaVal sign = p->w;
                p->w -= rate * SIGN(C, sign);
                if(p->w*sign<=0){
                        tmp=(crfceFea *)p->next;
                        free(p);
                        FeaTable[i][j]=(void *)tmp;
                        p=tmp;
                }else{
                        p=(crfceFea *)p->next;
                }
        }
        */

        for (; p != NULL; p = (crfceFea *)tmp->next) {
          p->w = p->tmpW;
          p->w +=
              rate * (supWeight * p->g + unSupWeight * p->unSupG - p->w * C);
          /*ParaVal sign = p->w;
          p->w -= rate * SIGN(C, sign);
          if(p->w*sign<=0){
                  tmp->next=(void *)p->next;
                  free(p);
          }else{*/
          tmp = p;
          //}
        }
      }
    }
  }
}

void hashCrfceFea::nextSetting(float supWeight, float unSupWeight, float C,
                               unsigned short storeSize) {
  crfceFea *p;
  unsigned int i, j;
  if (numGrad > 0) {
    unsigned short k;
    unsigned short currMaxIndex =
        (numGrad - 1 < storeSize) ? numGrad - 1 : storeSize;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          for (k = currMaxIndex; k != 0; --k) {
            p->s[k] = p->s[k - 1];
            p->y[k] = p->y[k - 1];
          }

          p->tmpG = supWeight * p->g + unSupWeight * p->unSupG - p->tmpW * C;
          p->s[0] = p->w - p->tmpW;
          p->tmpW = p->w;
          p->g = 0;
          p->unSupG = 0;
        }
      }
    }

  } else {
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (crfceFea *)FeaTable[i][j]; p != NULL;
             p = (crfceFea *)p->next) {
          p->tmpW = p->w;
          p->g = 0;
          p->unSupG = 0;
        }
      }
    }
  }
}

void hashCrfceFea::addGrad(const string &target, ParaVal val) {
  const char *targetfea = target.c_str();
  crfceFea **pp = (crfceFea **)hash(targetfea);
  crfceFea *p = *pp;

  for (; p != NULL; p = (crfceFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      p->g += val;
      return;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (crfceFea *)malloc(sizeof(crfceFea) + sizeof(char) * size +
                              sizeof(ParaVal) * numGrad * 2)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;
  p->tmpW = 0;
  p->g = val;
  p->tmpG = 0;
  p->unSupG = 0;
  p->dir = 0;

  if (numGrad > 0) {
    p->y = (ParaVal *)(p->fea + size);
    p->s = p->y + numGrad;
    for (int i = 0; i < numGrad; i++) {
      p->y[i] = 0;
      p->s[i] = 0;
    }
  } else {
    p->y = NULL;
    p->s = NULL;
  }

  ++numFea;
}

void hashCrfceFea::addUnSupGrad(const string &target, ParaVal val) {
  const char *targetfea = target.c_str();
  crfceFea **pp = (crfceFea **)hash(targetfea);
  crfceFea *p = *pp;

  for (; p != NULL; p = (crfceFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      p->unSupG += val;
      return;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (crfceFea *)malloc(sizeof(crfceFea) + sizeof(char) * size +
                              sizeof(ParaVal) * numGrad * 2)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;
  p->tmpW = 0;
  p->g = 0;
  p->tmpG = 0;
  p->unSupG = val;
  p->dir = 0;

  if (numGrad > 0) {
    p->y = (ParaVal *)(p->fea + size);
    p->s = p->y + numGrad;
    for (int i = 0; i < numGrad; i++) {
      p->y[i] = 0;
      p->s[i] = 0;
    }
  } else {
    p->y = NULL;
    p->s = NULL;
  }

  ++numFea;
}

hashCwFea::hashCwFea(void) {}

hashCwFea::~hashCwFea(void) {
  if (covariance == NULL) {
    return;
  }

  Covariance *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashCovRow; ++i) {
    for (j = 0; j < hashCovCol; ++j) {
      for (p = covariance[i][j]; p != NULL; p = tmp) {
        tmp = (Covariance *)p->next;
        free(p);
      }
    }
  }
  free(covariance);
}

void hashCwFea::initialize(unsigned int row, unsigned int col,
                           unsigned int covRow, unsigned int covCol,
                           float var) {
  unsigned int i, j;

  numFea = 0;
  hashRow = row;
  hashCol = col;
  hashCovRow = covRow;
  hashCovCol = covCol;
  initVar = var;
  initCov = 0;

  if ((FeaTable = (void ***)malloc(row * sizeof(void **) +
                                   row * col * sizeof(void *))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < row; ++i) {
    FeaTable[i] = (void **)(FeaTable + row) + i * col;
    for (j = 0; j < col; ++j) {
      FeaTable[i][j] = NULL;
    }
  }

  if (covRow == 0) {
    return;
  }

  if ((covariance = (Covariance ***)malloc(covRow * sizeof(Covariance **) +
                                           covRow * covCol *
                                               sizeof(Covariance *))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < covRow; ++i) {
    covariance[i] = (Covariance **)(covariance + covRow) + i * covCol;
    for (j = 0; j < covCol; ++j) {
      covariance[i][j] = NULL;
    }
  }
}

void hashCwFea::averageWeight() {
  unsigned int i, j;
  cwFea *p;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (cwFea *)FeaTable[i][j]; p != NULL; p = (cwFea *)p->next) {
        p->avgw += p->w;
      }
    }
  }
}

void hashCwFea::swapWeight() {
  unsigned int i, j;
  ParaVal tmp;
  cwFea *p;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (cwFea *)FeaTable[i][j]; p != NULL; p = (cwFea *)p->next) {
        tmp = p->avgw;
        p->avgw = p->w;
        p->w = tmp;
      }
    }
  }
}

void hashCwFea::oneStepRegularization(float lamda) {
  unsigned int i, j;
  cwFea *p;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (cwFea *)FeaTable[i][j]; p != NULL; p = (cwFea *)p->next) {
        p->w /= (1 + p->v * p->v * lamda);
      }
    }
  }
}

cwFea *hashCwFea::get(const string &target) {
  const char *targetfea = target.c_str();
  cwFea **pp = (cwFea **)hash(targetfea);
  cwFea *p = *pp;

  for (; p != NULL; p = (cwFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (cwFea *)malloc(sizeof(cwFea) + sizeof(char) * size)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;
  p->avgw = 0;
  p->v = initVar;
  p->tdelta = 0;

  ++numFea;
  return p;
}

void hashCwFea::setWeight(const string &target, ParaVal val) {
  const char *targetfea = target.c_str();
  cwFea **pp = (cwFea **)hash(targetfea);
  cwFea *p = *pp;

  for (; p != NULL; p = (cwFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      p->w = val;
      return;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (cwFea *)malloc(sizeof(cwFea) + sizeof(char) * size)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = val;
  p->avgw = 0;
  p->v = initVar;
  p->tdelta = 0;

  ++numFea;
}

inline Covariance **hashCwFea::hashCov(cwFea *a, cwFea *b) {
  Covariance **p;
  unsigned long int i = (unsigned long int)a;
  i += (i >> 13) + (i << 13) + (unsigned long int)b;

  p = covariance[(unsigned int)(i % hashCovRow)];

  i = (unsigned long int)b;
  i += (i >> 13) + (i << 13) + (unsigned long int)a;
  return &(p[(unsigned int)(i % hashCovCol)]);
}

Covariance *hashCwFea::getCovariance(cwFea *a, cwFea *b) {
  if (a > b) {
    cwFea *tmp = a;
    a = b;
    b = tmp;
  }
  Covariance **pp = (Covariance **)hashCov(a, b);
  Covariance *p = *pp;

  for (; p != NULL; p = (Covariance *)p->next) {
    if (p->a == a && p->b == b) {
      return p;
    }
  }

  if ((p = (Covariance *)malloc(sizeof(Covariance))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = *pp;
  *pp = p;

  p->a = a;
  p->b = b;
  p->cov = initCov;

  ++numCov;
  return p;
}

void hashCwFea::setCovariance(cwFea *a, cwFea *b, ParaVal val) {
  if (a > b) {
    cwFea *tmp = a;
    a = b;
    b = tmp;
  }
  Covariance **pp = (Covariance **)hashCov(a, b);
  Covariance *p = *pp;

  for (; p != NULL; p = (Covariance *)p->next) {
    if (p->a == a && p->b == b) {
      p->cov = val;
      return;
    }
  }

  if ((p = (Covariance *)malloc(sizeof(Covariance))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = *pp;
  *pp = p;

  p->a = a;
  p->b = b;
  p->cov = val;

  ++numCov;
}

void hashCwFea::writeAvgFeatures(const char *write) {
  ofstream WRITE;
  WRITE.open(write, ios_base::trunc);
  if (!WRITE) {
    cerr << "ERROR:Can not write model: " << write << endl;
    exit(EXIT_FAILURE);
  } else {
    Fea *p;
    unsigned int i, j;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (Fea *)FeaTable[i][j]; p != NULL; p = (Fea *)p->next) {
          WRITE << p->fea;
          WRITE << "\t";
          WRITE << ((cwFea *)p)->avgw;
          WRITE << "\n";
        }
      }
    }

    WRITE.close();
  }
}

void hashCwFea::writeCovariance() {
  Covariance *p;
  unsigned int i, j;
  for (i = 0; i < hashCovRow; ++i) {
    for (j = 0; j < hashCovCol; ++j) {
      for (p = covariance[i][j]; p != NULL; p = p->next) {
        cerr << "fea1:" << p->a->fea << " fea2:" << p->b->fea
             << " cov:" << p->cov << endl;
      }
    }
  }
}

hashSsvmFea::hashSsvmFea(void) {}

hashSsvmFea::~hashSsvmFea(void) {}

void hashSsvmFea::updateWeightL1(const float C) {
  ssvmFea *p;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      if ((p = (ssvmFea *)FeaTable[i][j]) == NULL) {
        continue;
      }

      for (; p != NULL; p = (ssvmFea *)p->next) {
        p->w = p->w + p->g;

        if (p->w >= 0) {
          p->w = MAX(p->w - C, 0);
        } else {
          p->w = MIN(p->w + C, 0);
        }

        p->g = 0;
      }
    }
  }
}

void hashSsvmFea::updateWeightL2(const float C) {
  ssvmFea *p;
  unsigned int i, j;
  float invC = 1.0 / (1 + C);
  std::cerr << "invC:" << invC << endl;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      if ((p = (ssvmFea *)FeaTable[i][j]) == NULL) {
        continue;
      }

      for (; p != NULL; p = (ssvmFea *)p->next) {
        p->w = (p->w + p->g) * invC;
        p->g = 0;
      }
    }
  }
}

void hashSsvmFea::averageWeight() {
  unsigned int i, j;
  ssvmFea *p;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (ssvmFea *)FeaTable[i][j]; p != NULL; p = (ssvmFea *)p->next) {
        p->avgw += p->w;
      }
    }
  }
}

void hashSsvmFea::swapWeight() {
  unsigned int i, j;
  ParaVal tmp;
  ssvmFea *p;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = (ssvmFea *)FeaTable[i][j]; p != NULL; p = (ssvmFea *)p->next) {
        tmp = p->avgw;
        p->avgw = p->w;
        p->w = tmp;
      }
    }
  }
}

ssvmFea *hashSsvmFea::get(const string &target) {

  const char *targetfea = target.c_str();
  ssvmFea **pp = (ssvmFea **)hash(targetfea);
  ssvmFea *p = *pp;

  for (; p != NULL; p = (ssvmFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      return p;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (ssvmFea *)malloc(sizeof(ssvmFea) + sizeof(char) * size)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = 0;
  p->avgw = 0;
  p->g = 0;

  ++numFea;
  return p;
}

void hashSsvmFea::setWeight(const string &target, ParaVal val) {

  const char *targetfea = target.c_str();
  ssvmFea **pp = (ssvmFea **)hash(targetfea);
  ssvmFea *p = *pp;

  for (; p != NULL; p = (ssvmFea *)p->next) {
    if (keyequal(p->fea, targetfea)) {
      p->w = val;
      return;
    }
  }

  unsigned short size = (unsigned short)target.size() + 1;
  if ((p = (ssvmFea *)malloc(sizeof(ssvmFea) + sizeof(char) * size)) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  p->next = (void *)*pp;
  *pp = p;

  p->fea = (char *)(p + 1);
  strcpy(p->fea, targetfea);
  p->w = val;
  p->avgw = 0;
  p->g = 0;

  ++numFea;
}

void hashSsvmFea::writeAvgFeatures(const char *write) {

  ofstream WRITE;
  WRITE.open(write, ios_base::trunc);
  if (!WRITE) {
    cerr << "ERROR:Can not write model: " << write << endl;
    exit(EXIT_FAILURE);
  } else {
    Fea *p;
    unsigned int i, j;
    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = (Fea *)FeaTable[i][j]; p != NULL; p = (Fea *)p->next) {
          WRITE << p->fea;
          WRITE << "\t";
          WRITE << ((ssvmFea *)p)->avgw;
          WRITE << "\n";
        }
      }
    }

    WRITE.close();
  }
}

