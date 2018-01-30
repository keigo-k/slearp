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

void *runThread(void *queue) {
  while (1) {
    Task *task = ((wqueue<Task *> *)queue)->remove();
    task->run();

    delete task;
  }
  return NULL;
};

discriminativeModel::discriminativeModel(const LocalOpt &lopt) {
  unsigned int tmp = (unsigned int)sqrt((double)lopt.hashPronunceRuleTableSize);
  rules.initialize(tmp, tmp);

  if (lopt.train) {
    tmp = (unsigned int)sqrt((double)lopt.hashTrainDataTableSize);
    trainset.initialize(tmp, tmp);
  }

  if (lopt.dev) {
    tmp = (unsigned int)sqrt((double)lopt.hashDevDataTableSize);
    devset.initialize(tmp, tmp);
  }

  tmpDelInsChar[0] = delInsChar;
  tmpDelInsChar[1] = '\0';
  tmpBoundChar[0] = interBound;
  tmpBoundChar[1] = '\0';
  tmpSeparateChar[0] = separateChar;
  tmpSeparateChar[1] = '\0';
  tmpTabChar = "\t";

  useContext = lopt.useContext;
  useTrans = lopt.useTrans;
  useChain = lopt.useChain;
  useJoint = lopt.useJoint;
  useFJoint = lopt.useFJoint;

  cngram = lopt.cngram;
  if ((cngram % 2) == 0) {
    cerr << "WARNING:Context and Chain ngram size must be odd:" << cngram
         << endl;
    cngram++;
    cerr << "The size is incremented:" << cngram << endl;
  }
  halfCNgram = cngram / 2;
  jngram = lopt.jngram;
  fjngram = lopt.fjngram;

  outputNbest = lopt.outputNbest;
  beamSize = lopt.beamSize;
}

inline unsigned short discriminativeModel::compOnePhoneme(const char *refp,
                                                          const char *&hypp) {
  while (*refp != '\0') {
    if (*refp != *hypp) {
      while (*hypp != '\0' && *hypp != unknownChar && *hypp != joinChar) {
        ++hypp;
      }
      return 1;
    }
    ++refp;
    ++hypp;
  }
  if (*hypp == '\0' || *hypp == unknownChar || *hypp == joinChar) {
    return 0;
  }
  while (*hypp != '\0' && *hypp != unknownChar && *hypp != joinChar) {
    ++hypp;
  }
  return 1;
}

void discriminativeModel::editDistance(const sequenceData &data,
                                       const YInfo &yinfo, Hyp *hyp,
                                       unsigned short &pwrong,
                                       unsigned short &pnum) {
  unsigned short i = 1, j = 0, min = 0;
  unsigned short lenref = yinfo.leny;
  unsigned short lenhyp;
  Hyp *p = hyp->next;
  const char *tmp;
  vector<vector<unsigned short> > sTable(lenref + 1, vector<unsigned short>());
  vector<const char *> hypsy;
  hypsy.reserve(data.lenx);
  for (; p->next != NULL; p = p->next) {
    tmp = p->y;
    if (!(*(tmp) == delInsChar && *(tmp + 1) == '\0')) {
      hypsy.push_back(tmp);
      do {
        ++tmp;

        if (*tmp == '\0' || *tmp == unknownChar || *tmp == joinChar) {
          sTable[0].push_back(j);
          ++j;
        }
      } while (*tmp != '\0');
    }
  }
  if (j == 0) {
    pwrong = lenref;
    pnum = lenref;
    return;
  }
  lenhyp = j;
  sTable[0].push_back(j);
  ++j;

  for (; i <= lenref; ++i) {
    sTable[i].assign(j, 0);
    sTable[i][0] = i;
  }

  vector<const char *>::iterator it;
  vector<const char *>::iterator endit = hypsy.begin();
  for (i = 1; i <= lenref; ++i) {

    j = 1;
    it = hypsy.end();
    do {
      --it;
      tmp = *it;
      do {
        min = sTable[i - 1][j - 1] + compOnePhoneme(yinfo.nosegy[i - 1], tmp);

        if (min > sTable[i - 1][j] + 1) {
          min = sTable[i - 1][j] + 1;
        }
        if (min > sTable[i][j - 1] + 1) {
          min = sTable[i][j - 1] + 1;
        }
        sTable[i][j] = min;

        ++j;
      } while (*tmp++ != '\0');
    } while (it != endit);
  }

  pwrong = sTable[lenref][lenhyp];
  pnum = MAX(lenref, lenhyp);
}

inline unsigned short
discriminativeModel::notSTRCMPIgnoreSpChar(const char *a, const char *b) {
  while (*a != '\0') {
    if (*a != *b && ((*a != unknownChar && *a != joinChar) ||
                     (*b != unknownChar && *b != joinChar))) {
      return 1;
    }
    ++a;
    ++b;
  }
  if (*b == '\0') {
    return 0;
  }
  return 1;
}

void discriminativeModel::editSegAndPhonemesDistance(const YInfo &yinfo,
                                                     Hyp *hyp,
                                                     unsigned short lenhyp,
                                                     unsigned short &pwrong,
                                                     unsigned short &pnum) {
  unsigned short i = 1, j = 1, min = 0;
  unsigned short lenref = yinfo.lenseg;
  Hyp *p = hyp->next;
  vector<vector<short> > sTable(
      lenref + 1, vector<short>(lenhyp + 1, 0)); // Now the lenhyp is lenx
                                                 // (>lenhyp) because the actual
                                                 // lenhyp is still unknown.

  for (; i <= lenref; ++i) {
    sTable[i][0] = i;
  }

  for (; p->next != NULL; p = p->next) {
    sTable[0][j] = j;
    ++j;
  }
  lenhyp = j - 1;

  for (i = 1; i <= lenref; ++i) {
    p = hyp->next;
    for (j = 1; j <= lenhyp; ++j) {
      min =
          sTable[i - 1][j - 1] + notSTRCMPIgnoreSpChar(yinfo.segy[i - 1], p->y);

      if (min > sTable[i - 1][j] + 1) {
        min = sTable[i - 1][j] + 1;
      }
      if (min > sTable[i][j - 1] + 1) {
        min = sTable[i][j - 1] + 1;
      }

      sTable[i][j] = min;

      p = p->next;
    }
  }

  pwrong = sTable[lenref][lenhyp];
  pnum = MAX(lenref, lenhyp);
}

void discriminativeModel::getCNgram(const sequenceData &data, const char *x,
                                    unsigned short leftPos,
                                    unsigned short rightPos,
                                    vector<string> &cNgramList) {

  unsigned short maxIndex = data.lenx + 1;
  unsigned short leftLimi = MAX(1 + leftPos - halfCNgram, 0);
  unsigned short rightLimi = MIN(rightPos + halfCNgram - 1, maxIndex);
  unsigned short posMiddle = leftPos - leftLimi + 1;
  unsigned short size = leftPos - leftLimi + rightLimi - rightPos + 3;

  vector<const char *> sources;
  sources.reserve(size);
  for (unsigned short i = leftLimi; i <= leftPos; ++i) {
    if (0 == i) {
      sources.push_back(tmpBoundChar);
    } else {
      sources.push_back(data.nosegx[i - 1]);
    }
  }

  string strx(x);
  strx.erase(remove_if(strx.begin(), strx.end(), specialCharJudge()),
             strx.end());
  sources.push_back(strx.c_str());

  for (unsigned short k = rightPos; k <= rightLimi; ++k) {
    if (k == maxIndex) {
      sources.push_back(tmpBoundChar);
    } else {
      sources.push_back(data.nosegx[k - 1]);
    }
  }

  cNgramList.reserve((size * (size + 1)) / 2);
  string feaStr = "";
  for (unsigned short i = 0; i < size; ++i) {
    for (unsigned short k = 0; k < cngram; ++k) {
      if (i + k >= size) {
        break;
      }
      feaStr += sources[i + k];
      cNgramList.push_back(feaStr + ":" + toString(i - posMiddle) + ":" +
                           toString(i + k - posMiddle));
    }
    feaStr = "";
  }
  //*/
  /*
          unsigned short graphMaxIndex=data.lenx+1;
          unsigned short middle1=leftPos+1;
          unsigned short middle2=rightPos-1;

          leftPos=MAX(1+leftPos-halfCNgram, 0);
          rightPos=MIN(rightPos+halfCNgram-1, graphMaxIndex);

          string feaStr;
          unsigned short i=leftPos;
          unsigned short lbias=middle1;

          while(i<=rightPos){
                  feaStr="";
                  unsigned short pos=i;
                  unsigned short rbias=lbias;
                  if(pos==0){
                          feaStr+=tmpBoundChar;
                          cNgramList.push_back(feaStr+":"+toString(i-lbias)+":"+toString(pos-rbias));
                          ++pos;
                  }
                  while(pos<rightPos){
                          if(middle1==(pos)){
                                  feaStr=feaStr+tmpSeparateChar + x +
     tmpSeparateChar;
                                  cNgramList.push_back(feaStr+":"+toString(i-lbias)+":"+toString(pos-rbias));
                                  pos+=lenx;
                                  rbias=middle2;
                          }else{
                                  feaStr+=data.nosegx[pos-1];
                                  cNgramList.push_back(feaStr+":"+toString(i-lbias)+":"+toString(pos-rbias));
                                  ++pos;
                          }
                  }

                  if(pos==graphMaxIndex){
                          feaStr+=tmpBoundChar;
                  }else if(middle1==(pos)){
                          feaStr=feaStr+tmpSeparateChar + x + tmpSeparateChar;
                  }else{
                          feaStr+=data.nosegx[pos-1];
                  }

                  cNgramList.push_back(feaStr+":"+toString(i-lbias)+":"+toString(pos-rbias));

                  if(middle1==i){
                          lbias=middle2;
                          i+=lenx;
                  }else{
                          ++i;
                  }
          }
  */
}

void discriminativeModel::recursiveGetFJNgram(const sequenceData &data,
                                              unsigned short nextPos,
                                              vector<string> &fjNgramList,
                                              vector<vector<bool> > &dupCheck,
                                              unsigned short currentNgram,
                                              unsigned short ngramLimit,
                                              int &start, int &end) {
  /*
          string partfea1;
          if(nextPos==data.lenx &&
     dupCheck[nextPos-startpos][currentNgram-2]==false){ //tmpBoundChar
                  dupCheck[][]=true;
                  partfea1=tmpBoundChar+tmpSeparateChar+tmpBoundChar+":"+toString(currentNgram);
                  return;
          }

          partfea1=":"+toString(currentNgram);
          ++currentNgram;

          int newstart=start int newend=end;

          vector<string>::iterator preit=fjNgramList.begin()+start;
          vector<string>::iterator preendit=preit+(end-start);
      vector<Pronunce>::iterator proit;
      vector<Pronunce>::iterator proendit;
      for(int k=1; k <= rules.maxlenx && (nextPos+k)<=data.lenx; k++){

                  PronunceRule *rule = rules.refer(data.nosegx[nextPos] , k);
          if(rule==NULL){
                          if(k==1){
                  rule=rules.registWithMemory(data.nosegx[nextPos],
     strlen(data.nosegx[nextPos])+1, tmpDelInsChar, 2);
              }else{
                  continue;
              }
          }

          proendit=rule->pronunceVec.end();
          for(proit=rule->pronunceVec.begin(); proit!=proendit; ++proit){
                          fjNgramList.push_back(string(rule->x)+tmpSeparateChar+proit->y+partfea1);
     // input only current point

                          vector<string>::iterator tmpit=preit;
                          for(; preit!=preendit; ++preit){
                                  fjNgramList.push_back(*preit+tmpTabChar+string(rule->x)+tmpSeparateChar+proit->y+partfea1);
                          }
                          ++end;
                  }

                  if(currentNgram!=ngramLimit){
                          recursiveGetFJNgram(data, nextPos+k, fjNgramList,
     dupCheck, currentNgram, ngramLimit, start, end);
                          start=fjNgramList.size();
                          end=start;
                  }
          }
  */
}

void discriminativeModel::getFJNgram(const sequenceData &data,
                                     unsigned short nextPos,
                                     vector<string> &fjNgramList) {

  if (fjngram <= 1) {
    return;
  }

  int k = 1, start = 0, end = 0;
  unsigned short currentNgram = 2;
  unsigned short size = data.lenx;
  unsigned short ngramLimit =
      (fjngram < size - nextPos + 1)
          ? fjngram
          : size - nextPos + 1; // have to include last bound char

  if (ngramLimit == 1) {
    fjNgramList.push_back(string(tmpBoundChar) + tmpSeparateChar +
                          tmpBoundChar + ":1");
    return;
  }

  fjNgramList.reserve((ngramLimit * (ngramLimit + 1)));

  string partfea1;
  vector<Pronunce>::iterator proit;
  vector<Pronunce>::iterator proendit;

  if (ngramLimit <= 3) {

    vector<Pronunce>::iterator proit2;
    vector<Pronunce>::iterator proendit2;
    for (; k <= rules.maxlenx && (nextPos + k) <= size; k++) {

      PronunceRule *rule = rules.refer(data.nosegx[nextPos], k);
      if (rule == NULL) {
        if (k == 1) {
          rule = rules.registWithMemory(data.nosegx[nextPos],
                                        strlen(data.nosegx[nextPos]) + 1,
                                        tmpDelInsChar, 2);
        } else {
          continue;
        }
      }

      if (currentNgram != ngramLimit) {
        int l = nextPos + k;
        if (l == size) {
          fjNgramList.push_back(string(tmpBoundChar) + tmpSeparateChar +
                                tmpBoundChar + ":2");
          end = start + 1;
        } else {

          for (int m = 1; m <= rules.maxlenx && (l + m) <= size; m++) {

            PronunceRule *rule2 = rules.refer(data.nosegx[l], m);
            if (rule2 == NULL) {
              if (m == 1) {
                rule2 = rules.registWithMemory(data.nosegx[l],
                                               strlen(data.nosegx[l]) + 1,
                                               tmpDelInsChar, 2);
              } else {
                continue;
              }
            }

            proendit2 = rule2->pronunceVec.end();
            for (proit2 = rule2->pronunceVec.begin(); proit2 != proendit2;
                 ++proit2) {
              fjNgramList.push_back(string(rule2->x) + tmpSeparateChar +
                                    proit2->y + ":2");
            }
          }

          end = fjNgramList.size();
        }

        proendit = rule->pronunceVec.end();
        for (proit = rule->pronunceVec.begin(); proit != proendit; ++proit) {
          partfea1 = string(rule->x) + tmpSeparateChar + proit->y + ":1";
          fjNgramList.push_back(partfea1);
          for (l = start; l < end; ++l) {
            fjNgramList.push_back(partfea1 + tmpTabChar + fjNgramList[l]);
          }
        }
        start = fjNgramList.size();
      } else {
        proendit = rule->pronunceVec.end();
        for (proit = rule->pronunceVec.begin(); proit != proendit; ++proit) {
          fjNgramList.push_back(string(rule->x) + tmpSeparateChar + proit->y +
                                ":1");
        }
      }
    }

  } else {
    /*
                    vector< vector<bool> > dupCheck(ngramLimit-1,
       vector<bool>(fjngram-2, false));
            for(; k <= rules.maxlenx && (nextPos+k)<=size; k++){

                            PronunceRule *rule =
       rules.refer(data.nosegx[nextPos] , k);
                if(rule==NULL){
                                    if(k==1){
                            rule=rules.registWithMemory(data.nosegx[nextPos],
       strlen(data.nosegx[nextPos])+1, tmpDelInsChar, 2);
                        }else{
                            continue;
                        }
                    }

                            recursiveGetFJNgram(data, nextPos+k, fjNgramList,
       dupCheck, currentNgram, ngramLimit);
                            end=fjNgramList.size();

                    proendit=rule->pronunceVec.end();
                    for(proit=rule->pronunceVec.begin(); proit!=proendit;
       ++proit){
                                    partfea1=string(rule->x)+tmpSeparateChar+proit->y+":1";
                                    fjNgramList.push_back(partfea1);
                                    for(int l=start; l<end; ++l){
                                            fjNgramList.push_back(partfea1+tmpTabChar+fjNgramList[l]);
                                    }
                            }
                            start=fjNgramList.size();
                    }
    */
  }
}

float discriminativeModel::getCFeaturesScore(vector<string> &cNgramList,
                                             const char *curr,
                                             const char *prev) {

  float val = 0.0;
  if (prev != NULL) { // chain feature and trans feature
    string partfea = tmpTabChar + prev + tmpTabChar + curr;
    if (useChain) {
      vector<string>::iterator it;
      vector<string>::iterator endit = cNgramList.end();
      for (it = cNgramList.begin(); it != endit; ++it) {
        val += feas->getParaVal(*it + partfea);
      }
    }

    // trans feature
    if (useTrans) {
      partfea = prev + tmpTabChar + curr;
      val += feas->getParaVal(partfea);
    }
  } else { // context feature
    string partfea = tmpTabChar + curr;

    vector<string>::iterator it;
    vector<string>::iterator endit = cNgramList.end();
    for (it = cNgramList.begin(); it != endit; ++it) {
      val += feas->getParaVal(*it + partfea);
    }
  }

  return val;
}

float discriminativeModel::getJFeaturesScore(const Hyp &hyp) {
  float val = 0.0;
  string partfea1 = "";
  string partfea2 = tmpTabChar + hyp.x + tmpTabChar + hyp.y;
  const Hyp *p1, *p2;
  unsigned short i, j;
  if (jngram <= 0) {
    return val;
  }
  val += feas->getParaVal("0:0" + partfea2);

  /*
  if((p1=hyp.next)==NULL){
          return val;
  }
  */
  p1 = hyp.next;
  for (i = 1; i < jngram; ++i) {

    p2 = p1;
    for (j = i; j < jngram; ++j) {

      partfea1 += string(p2->x) + tmpSeparateChar + p2->y + tmpTabChar;
      val += feas->getParaVal(partfea1 + toString<int>(-j) + ":" +
                              toString<int>(-i) + partfea2);

      if ((p2 = p2->next) == NULL) {
        break;
      }
    }
    partfea1 = "";
    if ((p1 = p1->next) == NULL) {
      break;
    }
  }

  return val;
}

float discriminativeModel::getFJFeaturesScore(vector<string> &fjNgramList,
                                              const Hyp &hyp) {
  float val = 0.0;
  string partfea1 = tmpTabChar + hyp.x + tmpTabChar + hyp.y;

  vector<string>::iterator it;
  vector<string>::iterator endit = fjNgramList.end();
  for (it = fjNgramList.begin(); it != endit; ++it) {
    val += feas->getParaVal(*it + partfea1);
  }
  return val;
}

/*
void discriminativeModel::constructEvalGraph(const sequenceData &data,
vector<vectorNode> &graph){
        int size=data.lenx+1;
        //graph.resize(size+1); // include start and end symbol
        vector<Pronunce>::iterator it;
        vector<Pronunce>::iterator endit;
        //Node node={0,tmpBoundChar,tmpBoundChar,vector<Hyp>()};
        Node node;

        for(int i=0; i<=size; ++i){

                if(i==0){
                        node.pos=0;
                        node.x=tmpBoundChar;
                        node.y=tmpBoundChar;

                        graph[0].push_back(node);
                }else if(i==size){
                        node.pos=size;
                        node.x=tmpBoundChar;
                        node.y=tmpBoundChar;

                        graph[size].push_back(node);
                }else{
                        for(RuleLen j=1; j<=rules.maxlenx && i+j<=size; ++j){
                                PronunceRule *rule =
rules.refer(data.nosegx[i-1] , j);
                                if(rule!=NULL){
                                        node.pos=i;
                                        node.x=rule->x;

                                        endit=rule->pronunceVec.end();
                                        for(it=rule->pronunceVec.begin();
it!=endit; ++it){
                                                node.y=it->y;
                                                graph[i+j-1].push_back(node);
                                        }
                                }else if(j==1){
                                        node.pos=i;
                                        node.x=data.nosegx[i-1];
                                        node.y=tmpDelInsChar;
                                        graph[i+j-1].push_back(node);
                                }
                        }
                }

    }

}

void discriminativeModel::nbestViterbiEval(const sequenceData &data,
vector<vectorNode> &graph){
        vector<vectorNode>::iterator it=graph.begin();
        vector<vectorNode>::iterator endit=graph.end();

        // initialize
        Hyp hyp={&((*it)[0]), 0.0, NULL};
        (*it)[0].hyps.push_back(hyp);
        ++it;

        unsigned short internalNbest=15;
        int i=1;
        for(;it!=endit;++it){
                vectorNode::iterator nodeit=(*it).begin();
                vectorNode::iterator nodeEndit=(*it).end();
                for(;nodeit!=nodeEndit;++nodeit){
                        hyp.node=&(*nodeit);

                        vector<string> cNgramList;
                        unsigned short leftPos=(*nodeit).pos-1;
                        getCNgram(data, (*nodeit).x, i-leftPos, leftPos, i+1,
cNgramList);

                        ParaVal nodeScore=0;
                        if((*nodeit).x!=tmpBoundChar && useContext){
                                nodeScore=getCFeaturesScore(cNgramList,
(*nodeit).y, NULL);
                        }
                        vectorNode::iterator previt=(graph[leftPos]).begin();
                        vectorNode::iterator prevEndit=(graph[leftPos]).end();
                        for(;previt!=prevEndit;++previt){

                                ParaVal edgeScore=0.0;
                                if(useTrans || useChain){
                                        edgeScore+=getCFeaturesScore(cNgramList,
(*nodeit).y, (*previt).y);
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

void discriminativeModel::nbestBeamEval(const sequenceData &data,
                                        vector<vectorHyps> &hypsTable) {

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

      PronunceRule *rule = rules.refer(data.nosegx[i], k);
      if (rule == NULL) {
        if (k == 1) {
          rule = rules.registWithMemory(
              data.nosegx[i], strlen(data.nosegx[i]) + 1, tmpDelInsChar, 2);
        } else {
          continue;
        }
      }

      vector<string> cNgramList;
      hyp.x = rule->x;
      getCNgram(data, hyp.x, i, i + k + 1, cNgramList);

      vector<string> fjNgramList;
      if (useFJoint && (i + k) < size) {
        getFJNgram(data, i + k, fjNgramList);
      }

      proendit = rule->pronunceVec.end();
      for (proit = rule->pronunceVec.begin(); proit != proendit; ++proit) {
        hyp.y = proit->y;
        ParaVal nodeScore = 0.0;
        if (useContext) {
          nodeScore = getCFeaturesScore(cNgramList, hyp.y, NULL);
        }

        if (useFJoint && (i + k) < size) {
          nodeScore += getFJFeaturesScore(fjNgramList, hyp);
        }

        if (i == 0) {
          ParaVal edgeScore = 0.0;
          if (useTrans || useChain) {
            edgeScore = getCFeaturesScore(cNgramList, hyp.y, tmpBoundChar);
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
              prey = (*previt).y;
              edgeScore = getCFeaturesScore(cNgramList, hyp.y, prey);
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
      partial_sort((*it).begin(), (*it).begin() + beamSize, (*it).end(),
                   compHyp());
      (*it).resize(beamSize);
    }
    stable_sort((*it).begin(), (*it).end(), compPhonemes());
    ++it;
  }

  hyp.x = tmpBoundChar;
  hyp.y = tmpBoundChar;

  vector<string> cNgramList;
  getCNgram(data, hyp.x, size, size + 2, cNgramList);

  ParaVal edgeScore = 0.0;
  const char *prey = tmpBoundChar;
  vectorHyps::iterator previt = (*(-1 + it)).begin();
  vectorHyps::iterator prevEndit = (*(-1 + it)).end();
  for (; previt != prevEndit; ++previt) {

    if ((useTrans || useChain) && strcmp((*previt).y, prey) != 0) {
      edgeScore = getCFeaturesScore(cNgramList, hyp.y, (*previt).y);
      prey = (*previt).y;
    }

    hyp.next = &(*previt);
    hyp.score = (*previt).score + edgeScore;
    if (useJoint) {
      hyp.score += getJFeaturesScore(hyp);
    }

    (*it).push_back(hyp);
  }

  unsigned short outputNum = MIN((*it).size(), outputNbest);
  partial_sort((*it).begin(), (*it).begin() + outputNum, (*it).end(),
               compHyp());
}

discriminativeModel::~discriminativeModel(void) {}

void discriminativeModel::evalDev(float &wacc, float &perr) {
  // set testset
  unsigned int corr = 0;
  unsigned int total = 0;
  unsigned int totalPWrong = 0;
  unsigned int totalPNum = 0;

  sequenceData *p;
  unsigned int i, j;
  unsigned int hashRow = devset.hashRow;
  unsigned int hashCol = devset.hashCol;
  sequenceData ***sequenceDataTable = devset.sequenceDataTable;

  for (i = 0; i < hashRow; ++i) {
    for (j = 0; j < hashCol; ++j) {
      for (p = sequenceDataTable[i][j]; p != NULL; p = p->next) {

        /*
                vector<vectorNode> graph(p->lenx+2, vectorNode());
                constructEvalGraph(*p, graph);
                nbestViterbiEval(*p, graph);
        */
        vector<vectorHyps> hypsTable(p->lenx + 2, vectorHyps());
        nbestBeamEval(*p, hypsTable);

        // Hyp *hyp=&(graph.back().front().hyps.front());
        Hyp *hyp = &(hypsTable.back().front());

        YInfoVec::iterator it;
        YInfoVec::iterator endit = p->yInfoVec.end();
        for (it = p->yInfoVec.begin(); it != endit; ++it) {
          unsigned short pwrong, pnum;
          editDistance(*p, *it, hyp, pwrong, pnum);

          totalPWrong += pwrong;
          totalPNum += pnum;

          if (pwrong == 0) {
            ++corr;
          }

          ++total;
        }
      }
    }
  }

  wacc = ((float)corr) / total;
  perr = ((float)totalPWrong) / totalPNum;
}

void discriminativeModel::eval(const LocalOpt &lopt) {
  // set testset
  unsigned int corr = 0;
  unsigned int total = 0;
  unsigned int totalPWrong = 0;
  unsigned int totalPNum = 0;
  bool writePerfomance = false;

  ifstream INPUTFILE;
  INPUTFILE.open(lopt.eval);
  if (!INPUTFILE) {
    cerr << endl << "Error:Unable to open file:" << lopt.eval << endl;
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

    sequenceData data;
    try {
      hashSequenceData::mkSequenceData(line, data);
      /*
      vector<vectorNode> graph(data.lenx+2, vectorNode());
      constructEvalGraph(data, graph);
      nbestViterbiEval(data, graph);
      */
      vector<vectorHyps> hypsTable(data.lenx + 2, vectorHyps());
      nbestBeamEval(data, hypsTable);

      Hyp *hyp;
      YInfoVec::iterator yit = data.yInfoVec.begin();

      if (yit->nosegy != NULL) {
        hyp = &(hypsTable.back().front());
        unsigned short pwrong, pnum;
        editDistance(data, *yit, hyp, pwrong, pnum);
        totalPWrong += pwrong;
        totalPNum += pnum;

        if (pwrong == 0) {
          ++corr;
        }

        writePerfomance = true;
      } else {
        vectorHyps *vh = &(hypsTable.back());
        vectorHyps::iterator it;
        vectorHyps::iterator endit = vh->end();
        int i = 0;
        for (it = vh->begin(); it != endit && i < outputNbest; ++it) {
          string hypstrx = "";
          string hypstry = "";
          hyp = (*it).next;
          while (hyp->next != NULL) {
            // cout << hyp->x << "/" << hyp->y << endl;
            hypstrx = hyp->x + string(tmpSeparateChar) + hypstrx;
            hypstry = hyp->y + string(tmpSeparateChar) + hypstry;
            hyp = hyp->next;
          }
          cout << hypstrx << string(tmpTabChar) << hypstry << endl;
          ++i;
        }
      }

      ++total;
    }
    catch (const char *warn) {
      cerr << "Warning:Line " << numOfLine << ": " << warn
           << ":This line is ignored." << endl;
    }
  }
  INPUTFILE.close();

  if (writePerfomance) {
    cerr << "Phoneme error rate in test: " << ((float)totalPWrong) / totalPNum
         << endl;
    cerr << "Word accuracy in test: " << ((float)corr) / total << endl;
  } else {
    cerr << "Finish: " << lopt.eval << endl;
  }
}

void discriminativeModel::readPronunceRule(const char *read) {
  ifstream READ;

  READ.open(read);
  if (!READ) {
    cerr << endl << "Error:Unable to open file:" << read << endl;
    exit(EXIT_FAILURE);
  }
  int numOfLine = 0;
  while (!READ.eof()) {
    numOfLine++;
    string line;
    getline(READ, line);
    if (line == "") {
      continue;
    }

    vector<string> res;
    size_t curr = 0, found;
    while ((found = line.find_first_of(tmpTabChar, curr)) != string::npos) {
      if (!res.empty()) {
        cerr << endl << "Error:Wrong format:Line " << numOfLine << ": " << read
             << endl;
        exit(EXIT_FAILURE);
      }
      res.push_back(string(line, curr, found - curr));
      curr = found + 1;
    }
    if (res.empty()) {
      cerr << endl << "Error:Wrong format:Line " << numOfLine << ": " << read
           << endl;
      exit(EXIT_FAILURE);
    }

    res.push_back(string(line, curr, line.size() - curr));

    if (res[0].size() > 0 && res[1].size() > 0) {
      rules.registWithMemory(res[0].c_str(), res[0].size() + 1, res[1].c_str(),
                             res[1].size() + 1);
    } else {
      cerr << endl << "Error:Wrong format:Line " << numOfLine << ": " << read
           << endl;
      exit(EXIT_FAILURE);
    }
  }
  READ.close();
}

