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

sarow::sarow(const LocalOpt &lopt) : discriminativeModel(lopt) {
  useFullCovariance = lopt.useFullCovariance;
  r = lopt.r;

  if (r <= 0) {
    useOptimizeR = true;
    b = lopt.b;
  } else {
    useOptimizeR = false;
  }

  trainNbest = lopt.trainNbest;
  lamda = 1;

  if (STRCMP(lopt.lossFuncTyp, (const char *)"ploss")) {
    lossFuncTyp = 0;
  } else {
    lossFuncTyp = 1;
  }

  unsigned int tmp1 = (unsigned int)sqrt((double)lopt.hashFeaTableSize);
  unsigned int tmp2;
  if (useFullCovariance) {
    tmp2 = (unsigned int)sqrt((double)lopt.covarianceTableSize);
  } else {
    tmp2 = 0;
  }
  feas = new hashCwFea();
  ((hashCwFea *)feas)->initialize(tmp1, tmp1, tmp2, tmp2, lopt.initVar);
}

sarow::~sarow() { delete feas; }

/*
void sarow::constructGraph(const sequenceData &data, vector<vectorOnlineNode>
&graph){
        int size=data.lenx+1;
        vector<Pronunce>::iterator it;
        vector<Pronunce>::iterator endit;
        onlineNode node;

        for(int i=0; i<=size; ++i){

                if(i==0){
                        node.pos=0;
                        node.x=tmpBoundChar;
                        node.y=tmpBoundChar;
                        graph[0].push_back(node);

                        // setting for next node
                        node.cNgramList.reserve(500);
                }else if(i==size){
                        node.pos=size;
                        node.x=tmpBoundChar;
                        node.y=tmpBoundChar;
                        node.cNgramList.reserve(200);

                        graph[size].push_back(node);
                }else{
                        for(RuleLen j=1; j<=rules.maxlenx && i+j<=size; ++j){
                                PronunceRule *rule =
rules.refer(data.nosegx[i-1] , j);

                                node.pos=i;
                                if(rule!=NULL){
                                        node.x=rule->x;

                                        endit=rule->pronunceVec.end();
                                        for(it=rule->pronunceVec.begin();
it!=endit; ++it){
                                                node.y=it->y;
                                                graph[i+j-1].push_back(node);
                                        }
                                }else if(j==1){
                                        node.x=data.nosegx[i-1];
                                        node.y=tmpDelInsChar;
                                        graph[i+j-1].push_back(node);
                                }
                        }
                }

    }
}

void sarow::nbestViterbi(const sequenceData &data, vector<vectorOnlineNode>
&graph){
        vector<vectorOnlineNode>::iterator it=graph.begin();
        vector<vectorOnlineNode>::iterator endit=graph.end();

        // initialize
        Hyp hyp={&((*it)[0]), 0.0, NULL};
        (*it)[0].hyps.push_back(hyp);
        (*it)[0].corr=hyp;
        ++it;

        unsigned short internalNbest=15;
        unsigned short i=1;
        unsigned short corrNodeIndex=0;
        onlineNode *preCorrNode=NULL;
        int segIndex=0;
        for(;it!=endit;++it){
                vectorOnlineNode::iterator nodeit=(*it).begin();
                vectorOnlineNode::iterator nodeEndit=(*it).end();
                for(;nodeit!=nodeEndit;++nodeit){
                        hyp.node=&(*nodeit);

                        unsigned short leftPos=(*nodeit).pos-1;
                        ParaVal nodeScore=0;
                        char corrNodeFlag=0;

                        if((*nodeit).x==tmpBoundChar){
                                corrNodeFlag=1;
                        }else if(corrNodeIndex==leftPos
                                && STRCMP((*nodeit).y, data.segy[segIndex])
                && STRCMP((*nodeit).x, data.segx[segIndex])){
                                corrNodeFlag=1;
                corrNodeIndex=i;
                ++segIndex;
                        }

                        if(useContext || useChain){
                                getCNgram(data, (*nodeit).x, i-leftPos, leftPos,
i+1, (*nodeit).cNgramList);

                                if((*nodeit).x!=tmpBoundChar && useContext){
                                        nodeScore=getCFeaturesScore((*nodeit).cNgramList,
(*nodeit).y, NULL);
                                }
                        }

                        vectorOnlineNode::iterator
previt=(graph[leftPos]).begin();
                        vectorOnlineNode::iterator
prevEndit=(graph[leftPos]).end();
                        for(;previt!=prevEndit;++previt){

                                ParaVal edgeScore=0.0;
                                if(useTrans || useChain){
                                        edgeScore+=getCFeaturesScore((*nodeit).cNgramList,
(*nodeit).y, (*previt).y);
                                }

                                if(corrNodeFlag && (0==leftPos ||
preCorrNode==&(*previt))){
                                        hyp.next=&(previt->corr);
                                        hyp.score=hyp.next->score+nodeScore+edgeScore;
                    if(useJoint){
                        hyp.score+=getJFeaturesScore(hyp);
                    }

                                        (*nodeit).corr=hyp;
                                }

                                vector<Hyp>::iterator
hypit=(*previt).hyps.begin();
                                vector<Hyp>::iterator
hypEndit=(*previt).hyps.end();
                                for(;hypit!=hypEndit;++hypit){
                                        hyp.next=&(*hypit);
                                        hyp.score=(*hypit).score+nodeScore+edgeScore;
                                        if(useJoint){
                                                hyp.score+=getJFeaturesScore(hyp);
                                        }

                                        (*nodeit).hyps.push_back(hyp);
                                }
                        }

                        if(corrNodeFlag){
                                preCorrNode=&(*nodeit);
                        }

                        // sort
                        if((*nodeit).hyps.size() > internalNbest){
                                sort((*nodeit).hyps.begin(),
(*nodeit).hyps.end(), compHyp());
                                (*nodeit).hyps.resize(internalNbest);
                        }else if((*nodeit).x==tmpBoundChar){
                                sort((*nodeit).hyps.begin(),
(*nodeit).hyps.end(), compHyp());
                        }
                }
                i++;
        }

}
*/

void sarow::nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                      vector<vector<vector<string> > > &cNgramLists,
                      vector<vector<string> > &fjNgramLists,
                      unsigned short trainNum) {
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

  trainNum = MIN((*it).size(), trainNum);
  partial_sort((*it).begin(), (*it).begin() + trainNum, (*it).end(), compHyp());
}

void sarow::getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
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

void sarow::getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
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

void sarow::update(const sequenceData &data, unsigned int totalNumTrain) {
  /*
  vector<vectorOnlineNode> graph(data.lenx+2, vectorOnlineNode());
  constructGraph(data, graph);
  nbestViterbi(data, graph);
  */
  vector<vectorHyps> hypsTable(data.lenx + 2, vectorHyps());

  unsigned short ysize = data.yInfoVec.size();
  unsigned short trainNum = MAX(trainNbest, ysize + 1);
  vector<vector<vector<string> > > cNgramLists;
  vector<vector<string> > fjNgramLists;

  if (useContext || useChain || useTrans) {
    cNgramLists.assign(data.lenx + 1, vector<vector<string> >(
                                          rules.maxlenx, vector<string>()));
  }

  if (useFJoint) {
    fjNgramLists.assign(data.lenx - 1, vector<string>());
  }

  nbestBeam(data, hypsTable, cNgramLists, fjNgramLists, trainNum);

  vectorHyps *hyps = &(hypsTable.back());
  trainNum = MIN(trainNum, hyps->size());

  vector<map<string, short> > corrFeats(ysize, map<string, short>());

  for (unsigned short i = 0; i < trainNum; i++) {
    Hyp *hyp = &((*hyps)[i]);

    unsigned short pwrong, pnum, miny = 0, minpnum = data.lenx,
                                 minpwrong = data.lenx, j = 0;
    float distance = 999.9;

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

    float loss = 0;
    if (!useOptimizeR) {
      loss = distance;
    }
    // update
    map<cwFea *, short> cwFeas;

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
          cwFeas[cp] = strit->second;
          loss -= cp->w * strit->second;
        }
        ++strit;
      }
    }

    if (!useOptimizeR && loss <= 0) {
      continue;
    }

    float beta = 0;
    float tmpValue = 0.0;
    vector<float> sigmax(cwFeas.size(), 0.0);
    vector<float>::iterator sigmaxit1 = sigmax.begin();
    map<cwFea *, short>::iterator it1 = cwFeas.begin();
    map<cwFea *, short>::iterator endit = cwFeas.end();

    if (useFullCovariance) {
      map<cwFea *, short>::iterator it2;
      Covariance *cov;
      while (it1 != endit) {

        it2 = cwFeas.begin();
        while (it2 != endit) {
          if (it2 == it1) { // diagonal element
            *sigmaxit1 += it1->second * it1->first->v;
          } else {
            cov = ((hashCwFea *)feas)->getCovariance(it1->first, it2->first);
            *sigmaxit1 += it2->second * cov->cov;
          }
          ++it2;
        }

        if ((tmpValue = (*sigmaxit1) * (it1->second)) > 0) {
          beta += tmpValue;
        } else {
          *sigmaxit1 = 0;
        }

        ++it1;
        ++sigmaxit1;
      }

      beta = 1 / beta;
      float alpha = loss * beta;
      sigmaxit1 = sigmax.begin();
      vector<float>::iterator sigmaxit2;
      it1 = cwFeas.begin();
      while (it1 != endit) {
        // process diagonal element
        it1->first->w += *sigmaxit1 * alpha;
        it1->first->v -= beta * (*sigmaxit1) * (*sigmaxit1);
        if (it1->first->v < 0) {
          it1->first->v = 0;
        }

        sigmaxit2 = sigmaxit1;
        ++sigmaxit2; // skip diagonal element
        it2 = it1;
        ++it2; // skip diagonal element
        while (it2 != endit) {
          cov = ((hashCwFea *)feas)->getCovariance(it1->first, it2->first);
          cov->cov -= beta * (*sigmaxit1) * (*sigmaxit2);
          ++it2;
          ++sigmaxit2;
        }
        ++it1;
        ++sigmaxit1;
      }
    } else {
      while (it1 != endit) {
        *sigmaxit1 = it1->second * it1->first->v;
        beta += *sigmaxit1 * it1->second;
        ++it1;
        ++sigmaxit1;
      }

      if (useOptimizeR) {
        if (beta * b > 1) {
          r = beta / (beta * b - 1);
          loss /= (beta);

          loss += minpwrong;
          if (loss <= 0) {
            continue;
          }

          sigmaxit1 = sigmax.begin();
          it1 = cwFeas.begin();
          while (it1 != endit) {
            // process diagonal element
            it1->first->tdelta += it1->second;
            it1->first->w = it1->first->v * it1->first->tdelta;
            it1->first->v = (r * it1->first->v) /
                            (r + it1->second * it1->second * it1->first->v);

            ++it1;
            ++sigmaxit1;
          }
        } else {
          if (beta < 1.0e-8) {
            beta = 1.0e-8;
          }

          loss /= (beta);

          loss += minpwrong;
          if (loss <= 0) {
            continue;
          }

          sigmaxit1 = sigmax.begin();
          it1 = cwFeas.begin();
          while (it1 != endit) {
            // process diagonal element
            it1->first->tdelta += it1->second;
            it1->first->w = it1->first->v * it1->first->tdelta;

            ++it1;
            ++sigmaxit1;
          }
        }
      } else {
        beta += r;
        beta = 1 / beta;
        float alpha = loss * beta;
        sigmaxit1 = sigmax.begin();
        it1 = cwFeas.begin();
        while (it1 != endit) {
          // process diagonal element
          it1->first->w += (*sigmaxit1) * alpha;
          it1->first->v = (r * it1->first->v) /
                          (r + it1->second * it1->second * it1->first->v);

          ++it1;
          ++sigmaxit1;
        }
      }
    }
  }
}

void sarow::getLocalFeatures(Hyp *hyp, char direction, map<string, short> &x,
                             vector<string> &cNgramList,
                             vector<string> &fjNgramList) {
  /*
          string partfea1;
          string partfea2;

          if(tmpBoundChar==hyp->x){
                  if(useChain){
                          partfea1=tmpTabChar+hyp->next->y+tmpTabChar+hyp->y;
                          vector<string>::iterator it;
                          vector<string>::iterator endit=cNgramList.end();
                          for(it=cNgramList.begin();it!=endit;++it){
                                  x[*it+partfea1]+=direction;
                          }
                  }

          }else if(useContext && useChain){
                  partfea1=tmpTabChar+hyp->y;
                  partfea2=tmpTabChar+hyp->next->y+tmpTabChar+hyp->y;

                  vector<string>::iterator it;
                  vector<string>::iterator endit=cNgramList.end();
                  for(it=cNgramList.begin();it!=endit;++it){
                          x[*it+partfea1]+=direction;
                          x[*it+partfea2]+=direction;
                  }

          }else{
                  if(useContext){
                          partfea1=tmpTabChar+hyp->y;
                          vector<string>::iterator it;
                          vector<string>::iterator endit=cNgramList.end();
                          for(it=cNgramList.begin();it!=endit;++it){
                                  x[*it+partfea1]+=direction;
                          }
                  }

                  if(useChain){
                          partfea1=tmpTabChar+hyp->next->y+tmpTabChar+hyp->y;
                          vector<string>::iterator it;
                          vector<string>::iterator endit=cNgramList.end();
                          for(it=cNgramList.begin();it!=endit;++it){
                                  x[*it+partfea1]+=direction;
                          }
                  }
          }

          if(useTrans){
                  x[hyp->next->y+tmpTabChar+hyp->y]+=direction;
          }

          if(useJoint && jngram>0){
                  partfea2=tmpTabChar+hyp->x+tmpTabChar+hyp->y;
                  x["0:0"+partfea2]+=direction;

                  const Hyp *p1,*p2;
                  p1=hyp->next;

                  partfea1="";
                  unsigned short i,j;
                  for(i=1;i<jngram;++i){

                          p2=p1;
                          for(j=i;j<jngram;++j){

                                  partfea1+=string(p2->x)+tmpSeparateChar+p2->y+tmpTabChar;
                                  x[partfea1+toString<int>(-j)+":"+toString<int>(-i)+partfea2]+=direction;
                                  if((p2=p2->next)==NULL){
                                          break;
                                  }
                          }
                          if((p1=p1->next)==NULL){
                          break;
                          }
                          partfea1="";
                  }
          }
  */
}

/*
localUpdate means to update parameter on each part string segmented by phrase
decoder,
while update that is not localUpdate means to update parameter on a whole
string.
*/
void sarow::localUpdate(const sequenceData &data, unsigned int totalNumTrain) {
  /*
  vector<vectorOnlineNode> graph(data.lenx+2, vectorOnlineNode());
  constructGraph(data, graph);
  nbestViterbi(data, graph);
  */
  /*
          vector<vectorHyps> hypsTable(data.lenx+2, vectorHyps());

          vectorHyps correct;
          correct.reserve(data.lenseg+1);

          vector< vector<string> > cNgramLists;
          cNgramLists.reserve(data.lenx*2);
          nbestBeam(data, hypsTable, correct, cNgramLists);

          vectorHyps *hyps=&(hypsTable.back());
          Hyp *ref=&(correct.back());
          unsigned short ysize = data.yInfoVec.size();
          unsigned short trainNum=MIN(MAX(trainNbest, ysize), hyps->size());

          unsigned short changew=0;
          for(unsigned short i=0 ; i<trainNum; i++){
                  Hyp *hyp=&((*hyps)[i]);
                  unsigned short pwrong, pnum;

                  YInfoVec::iterator it;
          YInfoVec::iterator endit=p->yInfoVec.end();
          for(it=p->yInfoVec.begin();it!=endit;++it){

                  if(lossFuncTyp==0){
                          editDistance(data, *it, hyp, pwrong, pnum);
                  }else{
                          editSegAndPhonemesDistance(*it, hyp, data.lenx,
     pwrong, pnum);
                  }

                  if(i==0){
                          totalPWrong+=pwrong;
                          totalPNum+=pnum;
                  }

                  if(pwrong==0){
                          continue;
                  }
                  float distance=((float)pwrong)/pnum;
                  float loss=0;
                  if(!changew){
                          loss = distance - ref->score + hyp->score;
                  }else{
                          loss = distance;
                  }
                  if(changew || loss>0){
                          //update
                          vector< map<cwFea *, short> > localCwFeas;
                          localCwFeas.reserve(data.lenseg);
                          unsigned short rxl=0;
                          unsigned short hxl=0;
                          Hyp *tmpr=ref;
                          Hyp *tmph=hyp;
                          const char *tmp;
                          //string refstr;
                          //string hypstr;
                          do{
                                  //refstr="";
                                  //hypstr="";
                                  map<string, short> localFeatures;
                                  do{
                                          if(rxl < hxl){
                                                  getLocalFeatures(tmpr, 1,
     localFeatures, cNgramLists[tmpr->cNgramListIndex]);
                                  //		refstr+=tmpr->y;
                                                  tmp=tmpr->x;
                                                  while(*tmp!='\0'){
                                                          if(*tmp==joinChar ||
     *tmp==unknownChar){
                                                                  ++rxl;
                                                          }
                                                          ++tmp;
                                                  }
                                                  ++rxl;

                                                  tmpr=tmpr->next;
                                          }else{
                                                  getLocalFeatures(tmph, -1,
     localFeatures, cNgramLists[tmph->cNgramListIndex]);
                                  //		hypstr+=tmph->y;
                                                  tmp=tmph->x;
                                                  while(*tmp!='\0'){
                                                          if(*tmp==joinChar ||
     *tmp==unknownChar){
                                                                  ++hxl;
                                                          }
                                                          ++tmp;
                                                  }
                                                  ++hxl;

                                                  tmph=tmph->next;
                                          }
                                  }while(rxl != hxl);

                                  //if(refstr!=hypstr){
                                  localCwFeas.push_back(map<cwFea *, short>());
                                  map<cwFea *, short>
     *pLocalCwFea=&(localCwFeas.back());

                                  map<string, short>::iterator
     strit=localFeatures.begin();
                                  map<string, short>::iterator
     endStrit=localFeatures.end();
                                  if(changew){
                                          cwFea *cp;
                                          while(strit!=endStrit){
                                                  if(strit->second!=0){
                                                          cp=((hashCwFea
     *)feas)->get(strit->first);
                                                          (*pLocalCwFea)[cp]=strit->second;
                                                          loss-=cp->w*strit->second;
                                                  }
                                                  ++strit;
                                          }
                                  }else{
                                          while(strit!=endStrit){
                                                  if(strit->second!=0){
                                                          (*pLocalCwFea)[((hashCwFea
     *)feas)->get(strit->first)]=strit->second;
                                                  }
                                                  ++strit;
                                          }
                                  }
                                  //}
                          }while(tmpr->next!=NULL);

                          //if(tmph!=NULL){
                          //	cerr << "Error:Code error" << endl;
                          //	exit(EXIT_FAILURE);
                          //}

                          if((changew && loss<=0) || localCwFeas.empty()){
                                  continue;
                          }

                          vector< map<cwFea *, short> >::iterator
     localCwFeait=localCwFeas.begin();
                          vector< map<cwFea *, short> >::iterator
     endLocalCwFeait=localCwFeas.end();
                          while(localCwFeait!=endLocalCwFeait){
                                  float beta=r;
                                  float tmpValue=0.0;
                                  vector<float> sigmax((*localCwFeait).size(),
     0.0);
                                  vector<float>::iterator
     sigmaxit1=sigmax.begin();
                                  map<cwFea *, short>::iterator
     it1=(*localCwFeait).begin();
                                  map<cwFea *, short>::iterator
     endit=(*localCwFeait).end();
                                  if(useFullCovariance){
                                          map<cwFea *, short>::iterator it2;
                                          Covariance *cov;
                                          while(it1!=endit){

                                                  it2=(*localCwFeait).begin();
                                                  while(it2!=endit){
                                                          if(it2==it1){ //
     diagonal element
                                                                  (*sigmaxit1)+=(it1->second)*(it1->first->v);
                                                          }else{
                                                                  cov=((hashCwFea
     *)feas)->getCovariance(it1->first, it2->first);
                                                                  (*sigmaxit1)+=(it2->second)*(cov->cov);
                                                          }
                                                          ++it2;
                                                  }

                                                  if((tmpValue=(*sigmaxit1)*(it1->second))>0){
                                                          beta+=tmpValue;
                                                  }else{
                                                          *sigmaxit1=0;
                                                  }

                                                  ++it1;
                                                  ++sigmaxit1;
                                          }

                                          float alpha=loss/(beta-r+1);
                                          beta=1/beta;
                                          sigmaxit1=sigmax.begin();
                                          vector<float>::iterator sigmaxit2;
                                          it1=(*localCwFeait).begin();
                                          while(it1!=endit){
                                                  // process diagonal element
                                                  it1->first->w+=*sigmaxit1*alpha;
                                                  it1->first->v-=beta*(*sigmaxit1)*(*sigmaxit1);
                                                  if(it1->first->v<0){
                                                          it1->first->v=0;
                                                  }

                                                  sigmaxit2=sigmaxit1;
                                                  ++sigmaxit2; // skip diagonal
     element
                                                  it2=it1;
                                                  ++it2; // skip diagonal
     element
                                                  while(it2!=endit){
                                                          cov=((hashCwFea
     *)feas)->getCovariance(it1->first, it2->first);
                                                          cov->cov-=beta*(*sigmaxit1)*(*sigmaxit2);
                                                          ++it2;
                                                          ++sigmaxit2;
                                                  }
                                                  ++it1;
                                                  ++sigmaxit1;
                                          }
                                  }else{
                                          while(it1!=endit){
                                                  *sigmaxit1+=it1->second*it1->first->v;
                                                  beta+=(*sigmaxit1)*it1->second;
                                                  ++it1;
                                                  ++sigmaxit1;
                                          }

                                          beta=1/beta;
                                          float alpha=loss*beta;
                                          sigmaxit1=sigmax.begin();
                                          it1=(*localCwFeait).begin();
                                          while(it1!=endit){
                                                  // process diagonal element
                                                  it1->first->w+=(*sigmaxit1)*alpha;
                                                  //it1->first->v =
     1.0/((1.0/it1->first->v)+((it1->second*it1->second)/r));
                                                  it1->first->v =
     (r*it1->first->v)/(r+it1->second*it1->second*it1->first->v);
                                                  ++it1;
                                                  ++sigmaxit1;
                                          }
                                  }

                                  ++localCwFeait;
                          }

                          ++changew;
                  }
          }
  */
}

void sarow::train(const LocalOpt &lopt) {
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
          /*
                  localUpdate means to update parameter on each part string
             segmented by phrase decoder,
                  while update that is not localUpdate means to update parameter
             on a whole string.
          */
          if (lopt.useLocal) {
            localUpdate(*p, totalNumTrain);
          } else {
            update(*p, totalNumTrain);
          }

          //((hashCwFea *)feas)->oneStepRegularization(lamda);
          ++numTrain;
          if ((numTrain % 2000) == 0) {
            cerr << "Learned: " << numTrain << endl;
          }
        }
      }
    }

    // lamda*=0.5;
    cerr << "Learned: " << numTrain << endl;

    modelScore = ((float)totalPWrong) / totalPNum;
    if (lossFuncTyp == 0) {
      cerr << "Phoneme error rate in train: " << modelScore << endl;
    } else {
      cerr << "Segment and phoneme error rate in train: " << modelScore << endl;
    }

    // if(useFullCovariance){
    //	((hashCwFea *)feas)->writeCovariance();
    //}

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

void sarow::readModel(const char *read) {
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

