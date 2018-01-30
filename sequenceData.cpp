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

#include "sequenceData.h"
#include "utility.h"

hashSequenceData::hashSequenceData() : sequenceDataTable(NULL) {}

hashSequenceData::~hashSequenceData() {
  if (sequenceDataTable == NULL) {
    return;
  }

  sequenceData *p, *tmp;
  unsigned int i, j;
  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = sequenceDataTable[i][j]; p != NULL; p = tmp) {
        tmp = p->next;
        delete p;
      }
    }
  }
  free(sequenceDataTable);
}

sequenceData **hashSequenceData::hashWithlen(const char *nosegx,
                                             unsigned short lenx) {

  sequenceData **p;
  unsigned int value = 0;
  const char *tmp = nosegx;
  char c;
  while (lenx) {
    --lenx;
    while ((c = *nosegx) != unknownChar && c != joinChar && c != separateChar &&
           c != '\0') {
      value += (c * c) * (127 << lenx);
      ++nosegx;
    }
    ++nosegx;
    value /= 7;
  }
  p = sequenceDataTable[value % hashRow];

  while (nosegx != tmp) {
    while ((c = *tmp) != unknownChar && c != joinChar && c != separateChar &&
           c != '\0') {
      value += (c * c) * (127 << lenx);
      ++tmp;
    }
    value /= 3;
    ++lenx;
    ++tmp;
  }

  return &(p[value % hashCol]);
}

int hashSequenceData::keyequalWithlen(const char *x, const char *nosegx,
                                      unsigned short lenx) {

  char c;
  while (lenx) {
    while ((c = *nosegx) != unknownChar && c != joinChar && c != separateChar &&
           c != '\0') {
      if (*x != c) {
        return 0;
      }
      ++x;
      ++nosegx;
    }
    if (*x != unknownChar && *x != joinChar && *x != separateChar &&
        *x != '\0') {
      return 0;
    }
    ++x;
    ++nosegx;
    --lenx;
  }

  return 1;
}

void hashSequenceData::initialize(unsigned int row, unsigned int col) {
  unsigned int i, j;

  numData = 0;
  numUniqData = 0;
  hashRow = row;
  hashCol = col;

  if ((sequenceDataTable = (sequenceData ***)malloc(
           row * sizeof(sequenceData **) +
           row * col * sizeof(sequenceData *))) == NULL) {
    cerr << "ERROR:Can not get memory in malloc.\nYou must need more memory.\n";
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < row; ++i) {
    sequenceDataTable[i] = (sequenceData **)(sequenceDataTable + row) + i * col;
    for (j = 0; j < col; ++j) {
      sequenceDataTable[i][j] = NULL;
    }
  }
}

YInfo hashSequenceData::regist(const string &str) {
  int size = str.size();
  const char *sp = str.c_str();
  const char *tmp = sp;

  char **nosegx = NULL;
  unsigned short lenx = 0;
  char **segx = NULL;
  char **segy = NULL;
  unsigned short lenseg = 0;
  char **nosegy = NULL;
  unsigned short leny = 0;

  unsigned short bytex = 0;
  unsigned short bytey = 0;

  unsigned short checklenseg = 0;
  char state = 0;
  while (*tmp != '\0') {
    if (*tmp == separateChar) {
      if (state == 0) {
        if (tmp != sp) {
          if (*(tmp - 1) != delInsChar) {
            ++lenx;
          }
        } else {
          throw "Wrong file format";
        }
        ++lenseg;
      } else {
        if (tmp != sp) {
          if (*(tmp - 1) != delInsChar) {
            ++leny;
          }
        } else {
          throw "Wrong file format";
        }
        ++checklenseg;
      }
    } else if (*tmp == joinChar || *tmp == unknownChar) {
      if (tmp != sp) {
        if (*(tmp - 1) != delInsChar) {
          if (state == 0) {
            ++lenx;
          } else {
            ++leny;
          }
        }
      } else {
        throw "Wrong file format";
      }
    } else if (*tmp == '\t') {
      state++;
      if (tmp != sp) {
        if (state == 1 && *(tmp - 1) != separateChar &&
            *(tmp - 1) != delInsChar) {
          ++lenx;
        }
      } else {
        throw "Wrong file format";
      }
    }
    ++tmp;
    if (state == 0) {
      ++bytex;
    } else {
      ++bytey;
    }
  }

  ++bytex;

  if (tmp != sp && lenseg == checklenseg && *(tmp - 1) != '\t') {
    if (*(tmp - 1) != separateChar && *(tmp - 1) != joinChar &&
        *(tmp - 1) != unknownChar && *(tmp - 1) != delInsChar) {
      if (state == 1) {
        ++leny;
      } else if (state == 0) {
        ++lenx;
      }
    }
  } else {
    throw "Wrong file format";
  }

  if (lenseg > 0) {
    // train or development data
    if ((segx = (char **)malloc(sizeof(char *) * lenseg * 2 +
                                sizeof(char *) * (leny) +
                                sizeof(char) * (size + 1 + bytey))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    segy = segx + lenseg;

    segx[0] = (char *)(segy + lenseg);
    strcpy(segx[0], sp);

    unsigned short tmplen = 0;
    state = 0;

    char *tmp2 = segx[0];
    while (*tmp2 != '\0') {
      if (*tmp2 == '\t') {
        state = 1;
        segy[0] = tmp2 + 1;
        tmplen = 0;
      } else if (*tmp2 == separateChar) {
        if (state == 0) {
          ++tmplen;
          if (tmplen < lenseg) {
            segx[tmplen] = tmp2 + 1;
          }
          *tmp2 = '\0';
        } else {
          ++tmplen;
          if (tmplen < lenseg) {
            segy[tmplen] = tmp2 + 1;
          }
          *tmp2 = '\0';
        }
      }

      tmp2++;
    }

    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    nosegx[0] = (char *)(nosegx + lenx);

    nosegy = (char **)(segx[0] + size + 1);
    nosegy[0] = (char *)(nosegy + leny);

    state = 0;
    tmplen = 0;
    tmp = sp;
    tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == '\t') {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          tmp2 = nosegy[0];
          ++tmp;
        } else {
          tmp2 = nosegy[0];
          ++tmp;
        }
        state = 1;
        tmplen = 0;
      } else if (*tmp == separateChar || *tmp == unknownChar ||
                 *tmp == joinChar) {
        if (state == 0) {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < lenx) {
              nosegx[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        } else {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < leny) {
              nosegy[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else if (lenx > 0 && state == 0) {
    // unsupervised or unlabeled test data
    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    nosegx[0] = (char *)(nosegx + lenx);

    unsigned short tmplen = 0;
    tmp = sp;
    char *tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == unknownChar || *tmp == joinChar) {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          ++tmplen;
          if (tmplen < lenx) {
            nosegx[tmplen] = tmp2 + 1;
          }
          ++tmp2;
          ++tmp;
        } else {
          --tmp2;
          ++tmp;
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else if (lenx > 0 && leny > 0) {
    // labeled test data

    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    if ((nosegy = (char **)malloc(sizeof(char *) * leny +
                                  sizeof(char) * (bytey))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    segx = nosegy; // Due to free function

    nosegx[0] = (char *)(nosegx + lenx);
    nosegy[0] = (char *)(nosegy + leny);

    state = 0;
    unsigned short tmplen = 0;
    tmp = sp;
    char *tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == '\t') {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          tmp2 = nosegy[0];
          ++tmp;
        } else {
          tmp2 = nosegy[0];
          ++tmp;
        }
        state = 1;
        tmplen = 0;
      } else if (*tmp == unknownChar || *tmp == joinChar) {
        if (state == 0) {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < lenx) {
              nosegx[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        } else {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < leny) {
              nosegy[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else {
    throw "Wrong file format";
  }

  sequenceData **pp = hashWithlen(nosegx[0], lenx);
  sequenceData *p = *pp;

  for (; p != NULL; p = p->next) {
    if (p->lenx == lenx && keyequalWithlen(p->nosegx[0], nosegx[0], lenx)) {

      YInfo yinfo = { segx, segy, lenseg, nosegy, leny };
      p->yInfoVec.push_back(yinfo);

      ++numData;
      return yinfo;
    }
  }

  p = new sequenceData;
  p->next = *pp;
  *pp = p;

  p->nosegx = nosegx;
  p->lenx = lenx;

  YInfo yinfo = { segx, segy, lenseg, nosegy, leny };
  p->yInfoVec.push_back(yinfo);

  ++numUniqData;
  ++numData;
  return yinfo;
}

sequenceData *hashSequenceData::refer(const char *nosegx, unsigned short lenx) {
  sequenceData **pp = hashWithlen(nosegx, lenx);
  sequenceData *p = *pp;

  for (; p != NULL; p = p->next) {
    if (p->lenx == lenx && keyequalWithlen(p->nosegx[0], nosegx, lenx)) {
      return p;
    }
  }
  return NULL;
}

void hashSequenceData::mkSequenceData(const string &str, sequenceData &data) {
  int size = str.size();
  const char *sp = str.c_str();
  const char *tmp = sp;

  char **nosegx = NULL;
  unsigned short lenx = 0;
  char **segx = NULL;
  char **segy = NULL;
  unsigned short lenseg = 0;
  char **nosegy = NULL;
  unsigned short leny = 0;

  unsigned short bytex = 0;
  unsigned short bytey = 0;

  unsigned short checklenseg = 0;
  char state = 0;
  while (*tmp != '\0') {
    if (*tmp == separateChar) {
      if (state == 0) {
        if (tmp != sp) {
          if (*(tmp - 1) != delInsChar) {
            ++lenx;
          }
        } else {
          throw "Wrong file format";
        }
        ++lenseg;
      } else {
        if (tmp != sp) {
          if (*(tmp - 1) != delInsChar) {
            ++leny;
          }
        } else {
          throw "Wrong file format";
        }
        ++checklenseg;
      }
    } else if (*tmp == joinChar || *tmp == unknownChar) {
      if (tmp != sp) {
        if (*(tmp - 1) != delInsChar) {
          if (state == 0) {
            ++lenx;
          } else {
            ++leny;
          }
        }
      } else {
        throw "Wrong file format";
      }
    } else if (*tmp == '\t') {
      state++;
      if (tmp != sp) {
        if (state == 1 && *(tmp - 1) != separateChar &&
            *(tmp - 1) != delInsChar) {
          ++lenx;
        }
      } else {
        throw "Wrong file format";
      }
    }
    ++tmp;
    if (state == 0) {
      ++bytex;
    } else {
      ++bytey;
    }
  }

  ++bytex;

  if (tmp != sp && lenseg == checklenseg && *(tmp - 1) != '\t') {
    if (*(tmp - 1) != separateChar && *(tmp - 1) != joinChar &&
        *(tmp - 1) != unknownChar && *(tmp - 1) != delInsChar) {
      if (state == 1) {
        ++leny;
      } else if (state == 0) {
        ++lenx;
      }
    }
  } else {
    throw "Wrong file format";
  }

  if (lenseg > 0) {
    // train or development data
    if ((segx = (char **)malloc(sizeof(char *) * lenseg * 2 +
                                sizeof(char *) * (leny) +
                                sizeof(char) * (size + 1 + bytey))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    segy = segx + lenseg;

    segx[0] = (char *)(segy + lenseg);
    strcpy(segx[0], sp);

    unsigned short tmplen = 0;
    state = 0;

    char *tmp2 = segx[0];
    while (*tmp2 != '\0') {
      if (*tmp2 == '\t') {
        state = 1;
        segy[0] = tmp2 + 1;
        tmplen = 0;
      } else if (*tmp2 == separateChar) {
        if (state == 0) {
          ++tmplen;
          if (tmplen < lenseg) {
            segx[tmplen] = tmp2 + 1;
          }
          *tmp2 = '\0';
        } else {
          ++tmplen;
          if (tmplen < lenseg) {
            segy[tmplen] = tmp2 + 1;
          }
          *tmp2 = '\0';
        }
      }

      tmp2++;
    }

    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    nosegx[0] = (char *)(nosegx + lenx);

    nosegy = (char **)(segx[0] + size + 1);
    nosegy[0] = (char *)(nosegy + leny);

    state = 0;
    tmplen = 0;
    tmp = sp;
    tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == '\t') {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          tmp2 = nosegy[0];
          ++tmp;
        } else {
          tmp2 = nosegy[0];
          ++tmp;
        }
        state = 1;
        tmplen = 0;
      } else if (*tmp == separateChar || *tmp == unknownChar ||
                 *tmp == joinChar) {
        if (state == 0) {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < lenx) {
              nosegx[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        } else {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < leny) {
              nosegy[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else if (lenx > 0 && state == 0) {
    // unsupervised or unlabeled test data
    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    nosegx[0] = (char *)(nosegx + lenx);

    unsigned short tmplen = 0;
    tmp = sp;
    char *tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == unknownChar || *tmp == joinChar) {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          ++tmplen;
          if (tmplen < lenx) {
            nosegx[tmplen] = tmp2 + 1;
          }
          ++tmp2;
          ++tmp;
        } else {
          --tmp2;
          ++tmp;
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else if (lenx > 0 && leny > 0) {
    // labeled test data

    if ((nosegx = (char **)malloc(sizeof(char *) * lenx +
                                  sizeof(char) * (bytex))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    if ((nosegy = (char **)malloc(sizeof(char *) * leny +
                                  sizeof(char) * (bytey))) == NULL) {
      cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
              "memory.\n";
      exit(EXIT_FAILURE);
    }

    segx = nosegy; // Due to free function

    nosegx[0] = (char *)(nosegx + lenx);
    nosegy[0] = (char *)(nosegy + leny);

    state = 0;
    unsigned short tmplen = 0;
    tmp = sp;
    char *tmp2 = nosegx[0];
    while (*tmp != '\0') {
      if (*tmp == '\t') {
        if (*(tmp - 1) != delInsChar) {
          *tmp2 = '\0';
          tmp2 = nosegy[0];
          ++tmp;
        } else {
          tmp2 = nosegy[0];
          ++tmp;
        }
        state = 1;
        tmplen = 0;
      } else if (*tmp == unknownChar || *tmp == joinChar) {
        if (state == 0) {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < lenx) {
              nosegx[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        } else {
          if (*(tmp - 1) != delInsChar) {
            *tmp2 = '\0';
            ++tmplen;
            if (tmplen < leny) {
              nosegy[tmplen] = tmp2 + 1;
            }
            ++tmp2;
            ++tmp;
          } else {
            --tmp2;
            ++tmp;
          }
        }
      } else {
        *tmp2 = *tmp;
        ++tmp2;
        ++tmp;
      }
    }
    *tmp2 = '\0';

  } else {
    throw "Wrong file format";
  }

  data.nosegx = nosegx;
  data.lenx = lenx;

  YInfo yinfo = { segx, segy, lenseg, nosegy, leny };
  data.yInfoVec.push_back(yinfo);
}

