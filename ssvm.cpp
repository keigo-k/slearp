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

#include "discriminativeModel.h"
#include "OptimizePara.h"

ssvm::ssvm(const LocalOpt &lopt) : discriminativeModel(lopt) {
  trainNbest = lopt.trainNbest;

  if (STRCMP(lopt.lossFuncTyp, (const char *)"ploss")) {
    lossFuncTyp = 0;
  } else {
    lossFuncTyp = 1;
  }

  unsigned int tmp1 = (unsigned int)sqrt((double)lopt.hashFeaTableSize);

  feas = new hashSsvmFea();
  ((hashSsvmFea *)feas)->initialize(tmp1, tmp1);
}

ssvm::~ssvm() { delete feas; }

void ssvm::nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                     vector<vector<vector<string> > > &cNgramLists,
                     vector<vector<string> > &fjNgramLists) {
  vector<Pronunce>::iterator proit;
  vector<Pronunce>::iterator proendit;

  // initialize
  Hyp hyp = { tmpBoundChar, tmpBoundChar, -1, 1, 0.0, NULL };
  unsigned short size = data.lenx;
  vector<vectorHyps>::iterator it = hypsTable.begin();
  (*it).push_back(hyp);
  ++it;

  for (int i = 0; i < size; ++i) {

    for (int k = 1; k <= rules.maxlenx && (i + k) <= size; k++) {

      // cerr<<"i:"<<i<<" k:"<<k<<" size:"<<size<<"
      // rules.maxlenx:"<<rules.maxlenx<<endl;
      PronunceRule *rule = rules.refer(data.nosegx[i], k);
      if (rule == NULL) {
        // cerr <<"NULL"<<endl;
        if (k == 1) {
          rule = rules.registWithMemory(
              data.nosegx[i], strlen(data.nosegx[i]) + 1, tmpDelInsChar, 2);
          // cerr <<"make del: "<<rule->x<<endl;
        } else {
          // cerr <<"continue"<<endl;
          continue;
        }
      }

      hyp.pos = i;
      hyp.len = k;
      hyp.x = rule->x;
      if (useContext || useChain) {
        getCNgram(data, hyp.x, i, i + k + 1, cNgramLists[i][k - 1]);
      }

      if (useFJoint && (i + k) < size && fjNgramLists[i + k - 1].empty()) {
        getFJNgram(data, i + k, fjNgramLists[i + k - 1]);
      }

      proendit = rule->pronunceVec.end();
      for (proit = rule->pronunceVec.begin(); proit != proendit; ++proit) {
        hyp.y = proit->y;
        ParaVal nodeScore = 0.0;
        if (useContext) {
          nodeScore = getCFeaturesScore(cNgramLists[i][k - 1], hyp.y, NULL);
        }

        if (useFJoint && (i + k) < size) {
          nodeScore += getFJFeaturesScore(fjNgramLists[i + k - 1], hyp);
        }

        if (i == 0) {
          ParaVal edgeScore = 0.0;
          if (useTrans || useChain) {
            edgeScore =
                getCFeaturesScore(cNgramLists[i][k - 1], hyp.y, tmpBoundChar);
          }

          hyp.score = nodeScore + edgeScore;
          hyp.next = &((*(-1 + it)).front());
          if (useJoint) {
            hyp.score += getJFeaturesScore(hyp);
          }

          (*((k - 1) + it)).push_back(hyp);
        } else {
          ParaVal edgeScore = 0.0;
          const char *prey = tmpBoundChar;
          vectorHyps::iterator previt = (*(-1 + it)).begin();
          vectorHyps::iterator prevEndit = (*(-1 + it)).end();
          for (; previt != prevEndit; ++previt) {

            if ((useTrans || useChain) && strcmp((*previt).y, prey) != 0) {
              // cerr <<"change: "<<(*previt).y<<"\t"<< prey<<endl;
              prey = (*previt).y;
              edgeScore = getCFeaturesScore(cNgramLists[i][k - 1], hyp.y, prey);
            }

            hyp.next = &(*previt);
            hyp.score = (*previt).score + nodeScore + edgeScore;
            if (useJoint) {
              hyp.score += getJFeaturesScore(hyp);
            }

            (*((k - 1) + it)).push_back(hyp);
          }
        }
      }
    }

    if ((*it).size() > beamSize) {
      // cerr <<"sort"<<endl;
      partial_sort((*it).begin(), (*it).begin() + beamSize, (*it).end(),
                   compHyp());
      (*it).resize(beamSize);
    }

    // cerr <<"show element"<<endl;
    // for(unsigned int u=0;u<(*it).size();u++){
    //	cerr<<"show x: "<<(*it)[u].x<<"\t"<<(void *)&((*it)[u].x[0])<<endl;
    //	cerr<<"show y: "<<(*it)[u].y<<"\t"<<(void *)&((*it)[u].y[0])<<endl;
    //}
    // cerr <<"sort by compPhonemes:"<<(*it).size()<<endl;
    if (useTrans || useChain) {
      stable_sort((*it).begin(), (*it).end(), compPhonemes());
    }
    // for(unsigned int u=0;u<(*it).size();u++){
    //	cerr<<"show x: "<<(*it)[u].x<<"\t"<<(void *)&((*it)[u].x[0])<<endl;
    //	cerr<<"show y: "<<(*it)[u].y<<"\t"<<(void *)&((*it)[u].y[0])<<endl;
    //}

    ++it;
  }

  hyp.pos = size;
  hyp.len = 1;
  hyp.x = tmpBoundChar;
  hyp.y = tmpBoundChar;

  if (useChain) {
    getCNgram(data, hyp.x, size, size + 2, cNgramLists[size][0]);
  }

  ParaVal edgeScore = 0.0;
  const char *prey = tmpBoundChar;
  vectorHyps::iterator previt = (*(-1 + it)).begin();
  vectorHyps::iterator prevEndit = (*(-1 + it)).end();
  for (; previt != prevEndit; ++previt) {

    if ((useTrans || useChain) && strcmp((*previt).y, prey) != 0) {
      prey = (*previt).y;
      edgeScore = getCFeaturesScore(cNgramLists[size][0], hyp.y, prey);
    }

    hyp.next = &(*previt);
    hyp.score = (*previt).score + edgeScore;
    if (useJoint) {
      hyp.score += getJFeaturesScore(hyp);
    }

    (*it).push_back(hyp);
  }

  if ((*it).size() > beamSize) {
    partial_sort((*it).begin(), (*it).begin() + beamSize, (*it).end(),
                 compHyp());
    (*it).resize(beamSize);
  }
}

void ssvm::getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
                          vector<vector<vector<string> > > &cNgramLists,
                          vector<vector<string> > &fjNgramLists) {
  string partfea1;
  string partfea2;

  if (useChain) {
    partfea1 = tmpTabChar + hyp->next->y + tmpTabChar + hyp->y;

    vector<string>::iterator it;
    vector<string>::iterator endit = cNgramLists[hyp->pos][hyp->len - 1].end();
    for (it = cNgramLists[hyp->pos][hyp->len - 1].begin(); it != endit; ++it) {
      x[*it + partfea1] += direction;
    }
  }

  if (useTrans) {
    x[hyp->next->y + tmpTabChar + hyp->y] += direction;
  }

  if (useJoint && jngram > 0) {
    partfea2 = tmpTabChar + hyp->x + tmpTabChar + hyp->y;
    x["0:0" + partfea2] += direction;

    const Hyp *p1, *p2;
    p1 = hyp->next;

    partfea1 = "";
    unsigned short i, j;
    for (i = 1; i < jngram; ++i) {

      p2 = p1;
      for (j = i; j < jngram; ++j) {

        partfea1 += string(p2->x) + tmpSeparateChar + p2->y + tmpTabChar;
        x[partfea1 + toString<int>(-j) + ":" + toString<int>(-i) + partfea2] +=
            direction;

        if ((p2 = p2->next) == NULL) {
          break;
        }
      }
      if ((p1 = p1->next) == NULL) {
        break;
      }
      partfea1 = "";
    }
  }

  bool addFJoint = false;
  hyp = hyp->next;
  while (hyp->next != NULL) {

    if (useContext && useChain) {
      partfea1 = tmpTabChar + hyp->y;
      partfea2 = tmpTabChar + hyp->next->y + tmpTabChar + hyp->y;

      vector<string>::iterator it;
      vector<string>::iterator endit =
          cNgramLists[hyp->pos][hyp->len - 1].end();
      for (it = cNgramLists[hyp->pos][hyp->len - 1].begin(); it != endit;
           ++it) {
        x[*it + partfea1] += direction;
        x[*it + partfea2] += direction;
      }
    } else {
      if (useContext) {
        partfea1 = tmpTabChar + hyp->y;
        vector<string>::iterator it;
        vector<string>::iterator endit =
            cNgramLists[hyp->pos][hyp->len - 1].end();
        for (it = cNgramLists[hyp->pos][hyp->len - 1].begin(); it != endit;
             ++it) {
          x[*it + partfea1] += direction;
        }
      }

      if (useChain) {
        partfea1 = tmpTabChar + hyp->next->y + tmpTabChar + hyp->y;
        vector<string>::iterator it;
        vector<string>::iterator endit =
            cNgramLists[hyp->pos][hyp->len - 1].end();
        for (it = cNgramLists[hyp->pos][hyp->len - 1].begin(); it != endit;
             ++it) {
          x[*it + partfea1] += direction;
        }
      }
    }

    if (useTrans) {
      x[hyp->next->y + tmpTabChar + hyp->y] += direction;
    }

    if (useJoint && jngram > 0) {
      partfea2 = tmpTabChar + hyp->x + tmpTabChar + hyp->y;
      x["0:0" + partfea2] += direction;

      const Hyp *p1, *p2;
      p1 = hyp->next;

      partfea1 = "";
      unsigned short i, j;
      for (i = 1; i < jngram; ++i) {

        p2 = p1;
        for (j = i; j < jngram; ++j) {

          partfea1 += string(p2->x) + tmpSeparateChar + p2->y + tmpTabChar;
          x[partfea1 + toString<int>(-j) + ":" + toString<int>(-i) +
            partfea2] += direction;

          if ((p2 = p2->next) == NULL) {
            break;
          }
        }
        if ((p1 = p1->next) == NULL) {
          break;
        }
        partfea1 = "";
      }
    }

    if (useFJoint) {
      if (addFJoint) {
        partfea1 = tmpTabChar + hyp->x + tmpTabChar + hyp->y;
        vector<string>::iterator it;
        vector<string>::iterator endit =
            fjNgramLists[hyp->pos + hyp->len - 1].end();
        for (it = fjNgramLists[hyp->pos + hyp->len - 1].begin(); it != endit;
             ++it) {
          x[*it + partfea1] += direction;
        }
      } else {
        addFJoint = true;
      }
    }
    hyp = hyp->next;
  }
}

void ssvm::getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
                              char direction, map<string, short> &x,
                              vector<vector<vector<string> > > &cNgramLists,
                              vector<vector<string> > &fjNgramLists) {

  string partfea1;
  string partfea2;
  unsigned short size = data.lenx;
  char **segx = yinfo.segx;
  char **segy = yinfo.segy;
  char *tmp;
  int k, j = 0;
  for (int i = 0; i < size; i += k, ++j) {

    k = 0;
    tmp = segx[j];
    do {
      ++tmp;
      while (*tmp != '\0' && *tmp != unknownChar && *tmp != joinChar) {
        ++tmp;
      }
      ++k;
    } while (*tmp != '\0');

    if ((useContext || useChain) && cNgramLists[i][k - 1].empty()) {
      getCNgram(data, segx[j], i, i + k + 1, cNgramLists[i][k - 1]);
    }

    if (useFJoint && (i + k) < size && fjNgramLists[i + k - 1].empty()) {
      getFJNgram(data, i + k, fjNgramLists[i + k - 1]);
    }

    if (useContext && useChain) {
      partfea1 = tmpTabChar + segy[j];
      if (j == 0) {
        partfea2 = tmpTabChar + tmpBoundChar + tmpTabChar + segy[j];
      } else {
        partfea2 = tmpTabChar + segy[j - 1] + tmpTabChar + segy[j];
      }

      vector<string>::iterator it;
      vector<string>::iterator endit = cNgramLists[i][k - 1].end();
      for (it = cNgramLists[i][k - 1].begin(); it != endit; ++it) {
        x[*it + partfea1] += direction;
        x[*it + partfea2] += direction;
      }
    } else {
      if (useContext) {
        partfea1 = tmpTabChar + segy[j];

        vector<string>::iterator it;
        vector<string>::iterator endit = cNgramLists[i][k - 1].end();
        for (it = cNgramLists[i][k - 1].begin(); it != endit; ++it) {
          x[*it + partfea1] += direction;
        }
      }

      if (useChain) {
        if (j == 0) {
          partfea2 = tmpTabChar + tmpBoundChar + tmpTabChar + segy[j];
        } else {
          partfea2 = tmpTabChar + segy[j - 1] + tmpTabChar + segy[j];
        }

        vector<string>::iterator it;
        vector<string>::iterator endit = cNgramLists[i][k - 1].end();
        for (it = cNgramLists[i][k - 1].begin(); it != endit; ++it) {
          x[*it + partfea2] += direction;
        }
      }
    }

    if (useTrans) {
      if (j == 0) {
        x[tmpBoundChar + tmpTabChar + segy[j]] += direction;
      } else {
        x[segy[j - 1] + tmpTabChar + segy[j]] += direction;
      }
    }

    if (useJoint && jngram > 0) {
      partfea2 = tmpTabChar + segx[j] + tmpTabChar + segy[j];
      x["0:0" + partfea2] += direction;

      partfea1 = "";
      unsigned short l, m;
      for (l = 1; l < jngram && (j - l) >= -1; ++l) {
        for (m = l; m < jngram && (j - m) >= -1; ++m) {
          if ((j - m) == -1) {
            partfea1 += string(tmpBoundChar) + tmpSeparateChar + tmpBoundChar +
                        tmpTabChar;
          } else {
            partfea1 += string(segx[j - m]) + tmpSeparateChar + segy[j - m] +
                        tmpTabChar;
          }

          x[partfea1 + toString<int>(-m) + ":" + toString<int>(-l) +
            partfea2] += direction;
        }
        partfea1 = "";
      }
    }

    if (useFJoint && (i + k) < size) {
      partfea1 = string(tmpTabChar) + segx[j] + tmpTabChar + segy[j];
      vector<string>::iterator it;
      vector<string>::iterator endit = fjNgramLists[i + k - 1].end();
      for (it = fjNgramLists[i + k - 1].begin(); it != endit; ++it) {
        x[*it + partfea1] += direction;
      }
    }
  }

  if (useChain) {
    partfea1 = tmpTabChar + segy[j - 1] + tmpTabChar + tmpBoundChar;

    vector<string>::iterator it;
    vector<string>::iterator endit = cNgramLists[size][0].end();
    for (it = cNgramLists[size][0].begin(); it != endit; ++it) {
      x[*it + partfea1] += direction;
    }
  }

  if (useTrans) {
    x[segy[j - 1] + tmpTabChar + tmpBoundChar] += direction;
  }

  if (useJoint && jngram > 0) {
    partfea2 = tmpTabChar + tmpBoundChar + tmpTabChar + tmpBoundChar;
    x["0:0" + partfea2] += direction;

    partfea1 = "";
    unsigned short l, m;
    for (l = 1; l < jngram && (j - l) >= -1; ++l) {
      for (m = l; m < jngram && (j - m) >= -1; ++m) {
        if ((j - m) == -1) {
          partfea1 += string(tmpBoundChar) + tmpSeparateChar + tmpBoundChar +
                      tmpTabChar;
        } else {
          partfea1 +=
              string(segx[j - m]) + tmpSeparateChar + segy[j - m] + tmpTabChar;
        }
        x[partfea1 + toString<int>(-m) + ":" + toString<int>(-l) + partfea2] +=
            direction;
      }
      partfea1 = "";
    }
  }
}

void ssvm::calcGrad(const sequenceData &data) {
  vector<vectorHyps> hypsTable(data.lenx + 2, vectorHyps());

  unsigned short ysize = data.yInfoVec.size();
  vector<vector<vector<string> > > cNgramLists;
  vector<vector<string> > fjNgramLists;

  if (useContext || useChain || useTrans) {
    cNgramLists.assign(data.lenx + 1, vector<vector<string> >(
                                          rules.maxlenx, vector<string>()));
  }

  if (useFJoint) {
    fjNgramLists.assign(data.lenx - 1, vector<string>());
  }

  nbestBeam(data, hypsTable, cNgramLists, fjNgramLists);

  vectorHyps *hyps = &(hypsTable.back());
  unsigned short hypNum = hyps->size();
  unsigned short trainNum = MIN(trainNbest, hypNum);
  vector<pair<Hyp *, pair<unsigned short, unsigned short> > > nbestHyp;
  nbestHyp.reserve(trainNum);
  vector<pair<Hyp *, pair<unsigned short, unsigned short> > >::iterator nit;
  vector<pair<Hyp *, pair<unsigned short, unsigned short> > >::iterator endNit;

  unsigned short miny, nbestHypSize = 0;
  float lastScore = -9e7;
  for (unsigned short i = 0; i < hypNum; i++) {
    Hyp *hyp = &((*hyps)[i]);

    unsigned short pwrong, pnum, minpnum = data.lenx, minpwrong = data.lenx,
                                 j = 0;
    float distance = 999.9;

    miny = 0;
    YInfoVec::const_iterator yit;
    YInfoVec::const_iterator endyit = data.yInfoVec.end();
    for (yit = data.yInfoVec.begin(); yit != endyit; ++yit) {
      if (lossFuncTyp == 0) {
        editDistance(data, *yit, hyp, pwrong, pnum);
      } else {
        editSegAndPhonemesDistance(*yit, hyp, data.lenx, pwrong, pnum);
      }

      if ((((float)pwrong) / pnum) <= distance) {
        minpwrong = pwrong;
        minpnum = pnum;
        miny = j;

        if (minpwrong == 0) {
          break;
        }
        distance = ((float)pwrong) / pnum;
      }
      ++j;
    }

    if (i == 0) {
      totalPWrong += minpwrong;
      totalPNum += minpnum;
    }

    if (lastScore >= hyp->score + data.lenx && nbestHypSize >= trainNum) {
      break;
    }

    hyp->score += minpwrong;

    if (nbestHypSize < trainNum) {
      pair<Hyp *, pair<unsigned short, unsigned short> > currHyp(
          hyp, pair<unsigned short, unsigned short>(minpwrong, miny));

      nit = nbestHyp.begin();
      endNit = nbestHyp.end();
      while (nit != endNit) {
        --endNit;
        if (endNit->first->score >= hyp->score) {
          ++endNit;
          nbestHyp.insert(endNit, currHyp);
          break;
        }
      }
      if (nit == endNit) {
        nbestHyp.insert(nit, currHyp);
      }

      lastScore = nbestHyp.back().first->score;

    } else {
      if (lastScore < hyp->score) {
        pair<Hyp *, pair<unsigned short, unsigned short> > currHyp(
            hyp, pair<unsigned short, unsigned short>(minpwrong, miny));

        nit = nbestHyp.begin();
        endNit = nbestHyp.end();
        --endNit;

        while (nit != endNit) {
          --endNit;
          if (endNit->first->score >= hyp->score) {
            ++endNit;
            nbestHyp.insert(endNit, currHyp);
            break;
          }
        }

        if (nit == endNit) {
          nbestHyp.insert(nit, currHyp);
        }
        nbestHyp.pop_back();

        lastScore = nbestHyp.back().first->score;
      }
    }
  }

  vector<map<string, short> > corrFeats(ysize, map<string, short>());
  endNit = nbestHyp.end();
  for (nit = nbestHyp.begin(); nit != endNit; ++nit) {

    if (nit->second.first == 0) {
      continue;
    }

    miny = nit->second.second;
    if (corrFeats[miny].empty()) {
      getAllCorrFeatures(data, data.yInfoVec[miny], 1, corrFeats[miny],
                         cNgramLists, fjNgramLists);
    }

    map<string, short> feastrs = corrFeats[miny];
    getAllFeatures(nit->first, -1, feastrs, cNgramLists, fjNgramLists);

    map<string, short>::iterator strit = feastrs.begin();
    map<string, short>::iterator endStrit = feastrs.end();
    while (strit != endStrit) {
      if (strit->second != 0) {
        (((hashSsvmFea *)feas)->get(strit->first))->g += strit->second;
      }
      ++strit;
    }
  }
}

void ssvm::train(const LocalOpt &lopt) {
  // set trainset and devset
  cerr << "Read training set:" << lopt.train << endl;
  {
    ifstream INPUTFILE;
    INPUTFILE.open(lopt.train);
    if (!INPUTFILE) {
      cerr << "Error:Unable to open file:" << lopt.train << endl;
      exit(EXIT_FAILURE);
    }

    int numOfLine = 0;
    while (!INPUTFILE.eof()) {
      ++numOfLine;
      string line;
      getline(INPUTFILE, line);
      if (line == "") {
        continue;
      }

      try {
        YInfo yinfo = trainset.regist(line);

        if (yinfo.segy != NULL) {
          for (unsigned short i = 0; i < yinfo.lenseg; ++i) {
            rules.regist(yinfo.segx[i], yinfo.segy[i]);
          }
        } else {
          cerr << "ERROR:Training set needs a correct label: " << lopt.train
               << endl;
          exit(EXIT_FAILURE);
        }
      }
      catch (const char *warn) {
        cerr << "Warning:Line " << numOfLine << ": " << warn
             << ":This line is ignored." << endl;
      }
    }
    INPUTFILE.close();
  }
  cerr << "Write pronunciation rule." << endl;

  // write pronunce rules
  rules.writePronunceRules(lopt.write);

  if (lopt.dev != NULL) {

    cerr << "Read development set:" << lopt.dev << endl;
    {
      ifstream INPUTFILE;
      INPUTFILE.open(lopt.dev);
      if (!INPUTFILE) {
        cerr << "Error:Unable to open file:" << lopt.dev << endl;
        exit(EXIT_FAILURE);
      }

      int numOfLine = 0;
      while (!INPUTFILE.eof()) {
        ++numOfLine;
        string line;
        getline(INPUTFILE, line);
        if (line == "") {
          continue;
        }

        try {
          YInfo yinfo = devset.regist(line);
          if (yinfo.segy == NULL) {
            cerr << "ERROR:Development set needs a correct label: " << lopt.dev
                 << endl;
            exit(EXIT_FAILURE);
          }
        }
        catch (const char *warn) {
          cerr << "Warning:Line " << numOfLine << ": " << warn
               << ":This line is ignored." << endl;
        }
      }
      INPUTFILE.close();
    }
  }

  cerr << "Start training..." << endl;
  iter = 0;
  unsigned int trialIter = 0;
  float prevScore = 3.402e+38;
  float tol = 2.0e-8;
  // float C =(lopt.C)/trainset.numData;
  float C = lopt.C;
  string writeModel;
  while (1) {
    ++iter;
    cerr << "Iteration " << iter << endl;
    totalPWrong = 0;
    totalPNum = 0;
    float modelScore = 0.0;
    unsigned int numTrain = 0;

    sequenceData *p;
    unsigned int i, j;
    unsigned int hashRow = trainset.hashRow;
    unsigned int hashCol = trainset.hashCol;
    sequenceData ***sequenceDataTable = trainset.sequenceDataTable;

    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = sequenceDataTable[i][j]; p != NULL; p = p->next) {
          calcGrad(*p);

          ++numTrain;
          if ((numTrain % 2000) == 0) {
            cerr << "Done: " << numTrain << endl;
          }
        }
      }
    }

    cerr << "Done: " << numTrain << endl;

    // update
    if (STRCMP(lopt.method, "ssvml1")) {
      ((hashSsvmFea *)feas)->updateWeightL1(C);
    } else {
      ((hashSsvmFea *)feas)->updateWeightL2(C);
    }

    modelScore = ((float)totalPWrong) / totalPNum;
    if (lossFuncTyp == 0) {
      cerr << "Phoneme error rate in train: " << modelScore << endl;
    } else {
      cerr << "Segmented phoneme error rate in train: " << modelScore << endl;
    }

    if (lopt.useAvg) {
      cerr << "Average weights with previous weights" << endl;
      ((hashSsvmFea *)feas)->averageWeight();
    }

    if (lopt.dev != NULL) {
      float wacc = 0.0;
      modelScore = 0.0;
      if (lopt.useAvg) {
        ((hashSsvmFea *)feas)->swapWeight();
      }

      evalDev(wacc, modelScore);
      cerr << "Phoneme error rate in dev: " << modelScore << endl;
      cerr << "Word accuracy in dev: " << wacc << endl;

      if (lopt.stopCond == 1) {
        modelScore =
            0.5 * (((float)totalPWrong) / totalPNum) + 0.5 * (modelScore);
        cerr << "Averaged phoneme error rate for train and dev: " << modelScore
             << endl;
      }

      if (modelScore > prevScore) {
        if (iter > lopt.minIter) {
          cerr << "Keep previous model because current model is less "
                  "performance than previous model" << endl;

          if (trialIter >= lopt.maxTrialIter) {
            break;
          }
          ++trialIter;

        } else {
          if (iter > 1) {
            if (remove(writeModel.c_str())) { // 0 is success
              cerr << "WARNING:Can not remove a previous model file:"
                   << writeModel << endl;
            }
          }

          writeModel = lopt.write;
          writeModel += "." + toString(iter);
          cerr << "Write current model: " << writeModel << endl;
          feas->writeFeatures(writeModel.c_str());

          prevScore = modelScore;
        }
      } else {
        if (iter > 1) {
          if (remove(writeModel.c_str())) { // 0 is success
            cerr << "WARNING:Can not remove a previous model file:"
                 << writeModel << endl;
          }
        }

        writeModel = lopt.write;
        writeModel += "." + toString(iter);
        cerr << "Write current model: " << writeModel << endl;
        feas->writeFeatures(writeModel.c_str());

        if (2.0 * ABS(modelScore - prevScore) <=
            tol * (ABS(modelScore) + ABS(prevScore))) {

          if (trialIter >= lopt.maxTrialIter && iter > lopt.minIter) {
            break;
          }
          ++trialIter;
        } else {
          trialIter = 0;
        }

        prevScore = modelScore;
      }

      if (lopt.iter != 0 && lopt.iter <= iter) {
        cerr << "Reach max iteration: " << iter << endl;
        break;
      }

      if (lopt.useAvg) {
        ((hashSsvmFea *)feas)->swapWeight();
      }

    } else {

      if (modelScore > prevScore) {
        cerr << "Keep previous model because current model is less performance "
                "than previous model" << endl;
        if (iter > lopt.minIter) {
          break;
        } else {
          if (iter > 1 && remove(writeModel.c_str())) { // 0 is success
            cerr << "WARNING:Can not remove a previous model file:"
                 << writeModel << endl;
          }

          writeModel = lopt.write;
          writeModel += "." + toString(iter);
          cerr << "Write current model: " << writeModel << endl;
          if (lopt.useAvg) {
            ((hashSsvmFea *)feas)->writeAvgFeatures(writeModel.c_str());
          } else {
            feas->writeFeatures(writeModel.c_str());
          }

          prevScore = modelScore;
        }

      } else {
        if (iter > 1 && remove(writeModel.c_str())) { // 0 is success
          cerr << "WARNING:Can not remove a previous model file:" << writeModel
               << endl;
        }

        writeModel = lopt.write;
        writeModel += "." + toString(iter);
        cerr << "Write current model: " << writeModel << endl;
        if (lopt.useAvg) {
          ((hashSsvmFea *)feas)->writeAvgFeatures(writeModel.c_str());
        } else {
          feas->writeFeatures(writeModel.c_str());
        }

        if (2.0 * ABS(modelScore - prevScore) <=
            tol * (ABS(modelScore) + ABS(prevScore))) {
          cerr << "Convergence of model" << endl;

          if (iter > lopt.minIter) {
            break;
          }
        }

        prevScore = modelScore;
      }

      if (lopt.iter != 0 && lopt.iter <= iter) {
        cerr << "Reach max iteration: " << iter << endl;
        break;
      }
    }
  }
}

void ssvm::readModel(const char *read) {
  ifstream READ;
  READ.open(read);
  if (!READ) {
    cerr << "Error:Unable to open file:" << read << endl;
    exit(EXIT_FAILURE);
  }

  int numOfLine = 0;
  while (!READ.eof()) {
    numOfLine++;
    string line;
    getline(READ, line);
    if (line.empty()) {
      continue;
    }

    string fea = "";
    size_t curr = 0, tab = 0, found;
    if ((found = line.find_first_of("\t", curr)) != string::npos) {
      fea += string(line, curr, found - curr);
      curr = found + 1;
      ++tab;
    } else {
      cerr << "Error:Wrong format:Line " << numOfLine << ": " << read << endl;
      exit(EXIT_FAILURE);
    }

    while ((found = line.find_first_of("\t", curr)) != string::npos) {
      fea += "\t" + string(line, curr, found - curr);
      curr = found + 1;
      ++tab;
    }

    if (tab > 1) {
      try {
        ((hashSsvmFea *)feas)->setWeight(
            fea, fromString<ParaVal>(string(line, curr, line.size() - curr)));
      }
      catch (const char *error) {
        cerr << "Error:" << error
             << " to parameter value (float or double) from string:Line "
             << numOfLine << ": " << read << endl;
        exit(EXIT_FAILURE);
      }
    } else {
      cerr << "Error:Wrong format:Line " << numOfLine << ": " << read << endl;
      exit(EXIT_FAILURE);
    }
  }
  READ.close();
}
