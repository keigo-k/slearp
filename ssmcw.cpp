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

ssmcw::ssmcw(const LocalOpt &lopt) : discriminativeModel(lopt) {
  trainNbest = lopt.trainNbest;
  b = lopt.b;
  invC = 1.0 / lopt.C;

  if (STRCMP(lopt.lossFuncTyp, (const char *)"ploss")) {
    lossFuncTyp = 0;
  } else {
    lossFuncTyp = 1;
  }

  unsigned int tmp1 = (unsigned int)sqrt((double)lopt.hashFeaTableSize);
  unsigned int tmp2 = 0;
  feas = new hashCwFea();
  ((hashCwFea *)feas)->initialize(tmp1, tmp1, tmp2, tmp2, lopt.initVar);
}

ssmcw::~ssmcw() { delete feas; }

void ssmcw::nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                      vector<vector<vector<string> > > &cNgramLists,
                      vector<vector<string> > &fjNgramLists,
                      unsigned short hypNum) {
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
    stable_sort((*it).begin(), (*it).end(), compPhonemes());
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

  hypNum = MIN((*it).size(), hypNum);
  partial_sort((*it).begin(), (*it).begin() + hypNum, (*it).end(), compHyp());
}

void ssmcw::getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
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

void ssmcw::getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
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

void ssmcw::update(const sequenceData &data, unsigned int totalNumTrain) {
  vector<vectorHyps> hypsTable(data.lenx + 2, vectorHyps());

  unsigned short ysize = data.yInfoVec.size();
  unsigned short hypNum = trainNbest + ysize;
  vector<vector<vector<string> > > cNgramLists;
  vector<vector<string> > fjNgramLists;

  if (useContext || useChain || useTrans) {
    cNgramLists.assign(data.lenx + 1, vector<vector<string> >(
                                          rules.maxlenx, vector<string>()));
  }

  if (useFJoint) {
    fjNgramLists.assign(data.lenx - 1, vector<string>());
  }

  nbestBeam(data, hypsTable, cNgramLists, fjNgramLists, hypNum);

  vectorHyps *hyps = &(hypsTable.back());
  hypNum = MIN(hypNum, hyps->size());

  vector<map<string, short> > corrFeats(ysize, map<string, short>());
  vector<double> x; // init value
  x.reserve(hypNum);

  vector<map<cwFea *, pair<short, double> > *> vecCwFeas; // feature info
  vecCwFeas.reserve(hypNum);

  vector<double> v(trainNbest, 0); // coefficient for first order
  vector<vector<double> > q(
      trainNbest, vector<double>(trainNbest, 0)); // matrix for second order

  map<cwFea *, double> cashForWeightedMargin; // \Sigma_i x_i v_ip

  bool marginErr = false;
  unsigned short k = 0;
  for (unsigned short i = 0; i < hypNum && k < trainNbest; i++) {
    Hyp *hyp = &((*hyps)[i]);

    unsigned short pwrong, pnum, miny = 0, minpnum = data.lenx,
                                 minpwrong = data.lenx, j = 0;
    float distance = 99999.9;

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

        if (minpwrong == 0) {
          break;
        }

        distance = ((float)pwrong) / pnum;
        miny = j;
      }
      ++j;
    }

    if (i == 0) {
      totalPWrong += minpwrong;
      totalPNum += minpnum;
    }

    if (minpwrong == 0) {
      continue;
    }

    double delta_score = 0;
    double var = 0;

    map<cwFea *, pair<short, double> > *cwFeas =
        new map<cwFea *, pair<short, double> >();
    vecCwFeas.push_back(cwFeas);

    // update
    {
      if (corrFeats[miny].empty()) {
        getAllCorrFeatures(data, data.yInfoVec[miny], 1, corrFeats[miny],
                           cNgramLists, fjNgramLists);
      }

      map<string, short> feastrs = corrFeats[miny];
      getAllFeatures(hyp, -1, feastrs, cNgramLists, fjNgramLists);

      map<string, short>::iterator strit = feastrs.begin();
      map<string, short>::iterator endStrit = feastrs.end();
      cwFea *cp;
      while (strit != endStrit) {
        if (strit->second != 0) {
          cp = ((hashCwFea *)feas)->get(strit->first);
          delta_score += cp->w * strit->second;
          double m = cp->v * strit->second * strit->second;
          (*cwFeas)[cp] =
              make_pair(strit->second, m); // o_inp, v_inp/part_of_phi
          var += m;
        }
        ++strit;
      }
    }

    // prepare init value in quadratic problem
    double part_of_phi = 0;
    if (var * b > 1) {
      part_of_phi = (var * b - 1) / var;
    }

    double m = delta_score - minpwrong - part_of_phi * var;
    if (m < 0) {
      m = -m / (trainNbest * (var + invC));
      if (m < 1) {
        x.push_back(1);
      } else {
        x.push_back(m);
      }
      marginErr = true;
    } else {
      m = 0;
      x.push_back(1);
    }

    v[k] = delta_score - minpwrong;
    q[k][k] = var + invC;

    map<cwFea *, pair<short, double> >::iterator it = cwFeas->begin();
    map<cwFea *, pair<short, double> >::iterator endit = cwFeas->end();

    if (part_of_phi > 0 && m > 0) {
      while (it != endit) {
        it->second.second *= part_of_phi;
        cashForWeightedMargin[it->first] +=
            m * it->second.second; //\hat{\alpha}*v_{inp}
        ++it;
      }
    } else {
      if (part_of_phi > 0) {
        while (it != endit) {
          it->second.second *= part_of_phi;
          ++it;
        }
      } else {
        while (it != endit) {
          it->second.second = 0;
          ++it;
        }
      }
    }
    ++k;
  }

  vector<map<cwFea *, pair<short, double> > *>::iterator hypit1 =
      vecCwFeas.begin();
  vector<map<cwFea *, pair<short, double> > *>::iterator hypEndit =
      vecCwFeas.end();

  if (!marginErr) { // no update
    while (hypit1 != hypEndit) {
      delete *hypit1;
      ++hypit1;
    }

    return;
  }

  // prepare quadratic problem
  hypNum = k;
  if (hypNum < trainNbest) {
    v.resize(hypNum);
  }

  for (unsigned short i = 0; i < hypNum; i++) {
    map<cwFea *, pair<short, double> >::iterator it = (*hypit1)->begin();
    map<cwFea *, pair<short, double> >::iterator endit = (*hypit1)->end();

    double tmp1 = 0.0;
    double tmp2 = 0.0;
    while (it != endit) {
      double v_np = it->second.second;
      double tmp3 = cashForWeightedMargin[it->first];
      double tmp4 = 1 / (1 + 2 * tmp3);
      double tmp5 = tmp4 * tmp4;
      tmp1 += v_np * (tmp4 + 2 * tmp3 * tmp5);
      tmp2 += 2 * (v_np * v_np * tmp5);
      ++it;
    }

    v[i] -= tmp1;
    q[i][i] += tmp2;

    unsigned short j = i + 1;
    unsigned int feaSize = (*hypit1)->size();
    vector<map<cwFea *, pair<short, double> > *>::iterator hypit2 = hypit1;
    ++hypit2;
    while (hypit2 != hypEndit) {

      vector<map<cwFea *, pair<short, double> > *>::iterator tmpHypit;
      if (feaSize < (*hypit2)->size()) {
        it = (*hypit1)->begin();
        endit = (*hypit1)->end();
        tmpHypit = hypit2;
      } else {
        it = (*hypit2)->begin();
        endit = (*hypit2)->end();
        tmpHypit = hypit1;
      }

      tmp1 = 0.0;
      tmp2 = 0.0;
      map<cwFea *, pair<short, double> >::iterator tmpit;
      map<cwFea *, pair<short, double> >::iterator tmpEndit =
          (*tmpHypit)->end();
      while (it != endit) {

        if ((tmpit = (*tmpHypit)->find(it->first)) != tmpEndit) {
          tmp1 += it->first->v * it->second.first * tmpit->second.first;

          double tmp3 = 1 / (1 + 2 * cashForWeightedMargin[it->first]);
          tmp2 += (it->second.second * (tmpit->second.second) * tmp3 * tmp3);
        }

        ++it;
      }

      tmp1 += 2 * tmp2;
      q[i][j] = tmp1;
      q[j][i] = tmp1;

      ++j;
      ++hypit2;
    }

    ++hypit1;
  }

  // solve quadratic problem
  // L2 ssmcw
  solver.solveWithoutLinearConstraint(q, v, x); // x is weights for each hyp

  // update mean and variance
  hypit1 = vecCwFeas.begin();
  if (1 == hypNum) {
    map<cwFea *, pair<short, double> >::iterator it = (*hypit1)->begin();
    map<cwFea *, pair<short, double> >::iterator endit = (*hypit1)->end();

    while (it != endit) {
      cwFea *cw = it->first;
      cw->w += x[0] * cw->v * it->second.first; // o_inp
      cw->v /= (1 + 2 * x[0] * it->second.second);
      ++it;
    }

    delete *hypit1;
  } else {
    cashForWeightedMargin.clear();
    for (unsigned short i = 0; i < hypNum; i++) {
      double alpha = x[i];
      if (alpha < 1.0E-13) {
        continue;
      }
      map<cwFea *, pair<short, double> >::iterator it = (*hypit1)->begin();
      map<cwFea *, pair<short, double> >::iterator endit = (*hypit1)->end();
      while (it != endit) {
        cwFea *cw = it->first;
        cw->w += alpha * (cw->v) * (it->second.first); // o_inp
        cashForWeightedMargin[cw] += alpha * (it->second.second);

        ++it;
      }

      delete *hypit1;
      ++hypit1;
    }

    map<cwFea *, double>::iterator it = cashForWeightedMargin.begin();
    map<cwFea *, double>::iterator endit = cashForWeightedMargin.end();
    while (it != endit) {
      cwFea *cw = it->first;
      cw->v /= (1 + 2 * it->second);
      ++it;
    }
  }
}

void ssmcw::train(const LocalOpt &lopt) {
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
  float tol = 2.0e-7;
  unsigned int totalNumTrain = 1;
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
          update(*p, totalNumTrain);

          ++numTrain;
          if ((numTrain % 2000) == 0) {
            cerr << "Learned: " << numTrain << endl;
          }
        }
      }
    }

    cerr << "Learned: " << numTrain << endl;

    modelScore = ((float)totalPWrong) / totalPNum;
    if (lossFuncTyp == 0) {
      cerr << "Phoneme error rate in train: " << modelScore << endl;
    } else {
      cerr << "Segment and phoneme error rate in train: " << modelScore << endl;
    }

    if (lopt.useAvg) {
      cerr << "Average weights with previous weights" << endl;
      ((hashCwFea *)feas)->averageWeight();
    }

    if (lopt.dev != NULL) {
      float wacc = 0.0;
      modelScore = 0.0;
      if (lopt.useAvg) {
        ((hashCwFea *)feas)->swapWeight();
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
        ((hashCwFea *)feas)->swapWeight();
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
            ((hashCwFea *)feas)->writeAvgFeatures(writeModel.c_str());
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
          ((hashCwFea *)feas)->writeAvgFeatures(writeModel.c_str());
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

void ssmcw::readModel(const char *read) {
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
        ((hashCwFea *)feas)->setWeight(
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

