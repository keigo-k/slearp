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
#include <string>
#include <map>
#include <list>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include "hashFea.h"
#include "hashPronunceRule.h"
#include "utility.h"
#include "sequenceData.h"
#include "QPsolver.h"
#ifndef _INCLUDE_STRCMP_
#define _INCLUDE_STRCMP_
#define STRCMP(a, b) (*(a) == *(b) && !strcmp(a, b))
#endif

typedef struct hyp {
  const char *x;
  const char *y;
  short pos;
  short len;

  float score;
  struct hyp *next;
} Hyp;

class CrfceNode {
public:
  unsigned short pos;
  const char *x;
  const char *y;
  ParaVal fProb;
  ParaVal bProb;
  ParaVal nodeProb;
  vector<ParaVal> edgeNodeProb;
  unsigned int cNgramListIndex;
};

typedef vector<CrfceNode> vectorCrfceNode;
typedef vector<Hyp> vectorHyps;

class negativeSample : public sequenceData {
public:
  negativeSample(void) { nosegx = NULL; }

  vector<vectorCrfceNode> graph;
};

class negativeSamplePVec : public vector<negativeSample *> {
public:
  negativeSamplePVec(void) {}
  ~negativeSamplePVec(void) {
    vector<negativeSample *>::iterator it;
    vector<negativeSample *>::iterator endit = (*this).end();
    for (it = (*this).begin(); it != endit; ++it) {
      delete (*it);
    }
  }
};

// For pthread
class Task {
public:
  virtual ~Task() {};
  virtual void run() {};
};

class exitTask : public Task {
public:
  void run() { pthread_exit(this); }
};

template <typename T> class wqueue;

void *runThread(void *queue);

template <typename T> class wqueue {
  list<T> m_queue;
  pthread_mutex_t m_mutex;
  pthread_cond_t m_condv;
  pthread_cond_t m_condf;
  unsigned short numOfThread;
  unsigned short wait_worker;
  vector<pthread_t> ths;

public:
  wqueue(unsigned short arg_numOfThread) {
    pthread_mutex_init(&m_mutex, NULL);
    pthread_cond_init(&m_condv, NULL);
    pthread_cond_init(&m_condf, NULL);
    wait_worker = 0;
    numOfThread = arg_numOfThread;

    pthread_t th;
    ths.assign(numOfThread, th);

    for (unsigned short i = 0; i < numOfThread; ++i) {
      pthread_create(&(ths[i]), NULL, runThread, (void *)this);
    }
  }

  ~wqueue() {
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_condv);
    pthread_cond_destroy(&m_condf);
  }

  void add(T item) {
    pthread_mutex_lock(&m_mutex);
    m_queue.push_back(item);
    pthread_cond_signal(&m_condv);
    pthread_mutex_unlock(&m_mutex);
  }

  T remove() {
    pthread_mutex_lock(&m_mutex);
    ++wait_worker;
    while (m_queue.empty()) {
      if (wait_worker == numOfThread) {
        pthread_cond_signal(&m_condf);
      }
      pthread_cond_wait(&m_condv, &m_mutex);
    }
    --wait_worker;
    T item = m_queue.front();
    m_queue.pop_front();
    pthread_mutex_unlock(&m_mutex);
    return item;
  }

  void wait_allwork() {
    pthread_mutex_lock(&m_mutex);
    while (!m_queue.empty() || wait_worker != numOfThread) {
      pthread_cond_wait(&m_condf, &m_mutex);
    }
    pthread_mutex_unlock(&m_mutex);
  }

  void destroy() {
    for (unsigned short i = 0; i < numOfThread; ++i) {
      add(new exitTask());
    }

    void *task = NULL;
    for (unsigned short i = 0; i < numOfThread; ++i) {
      pthread_join(ths[i], &task);
      delete (exitTask *)(task);
    }
  }
};

class compHyp {
public:
  inline bool operator()(const Hyp &a, const Hyp &b) const {
    return a.score > b.score;
  }
};

class compPhonemes {
public:
  inline bool operator()(const Hyp &a, const Hyp &b) const {
    return !(strcmp(a.y, b.y));
  }
};

class specialCharJudge {
public:
  inline bool operator()(char c) const {
    return c == joinChar || c == unknownChar;
  }
};

class discriminativeModel {
protected:
  hashFea *feas;
  hashPronunceRule rules;
  hashSequenceData trainset;
  hashSequenceData devset;
  unsigned short iter;
  unsigned int updateIter;
  bool unSupFlag;

  char tmpDelInsChar[2];
  char tmpBoundChar[2];
  char tmpSeparateChar[2];
  string tmpTabChar;

  bool useContext;
  bool useTrans;
  bool useChain;
  bool useJoint;
  bool useFJoint;

  unsigned short cngram;
  unsigned short halfCNgram;
  unsigned short jngram;
  unsigned short fjngram;

  unsigned short beamSize;
  unsigned short outputNbest;

  inline unsigned short compOnePhoneme(const char *refp, const char *&hypp);
  void editDistance(const sequenceData &data, const YInfo &yinfo, Hyp *hyp,
                    unsigned short &pwrong, unsigned short &pnum);
  inline unsigned short notSTRCMPIgnoreSpChar(const char *a, const char *b);
  void editSegAndPhonemesDistance(const YInfo &yinfo, Hyp *hyp,
                                  unsigned short lenhyp, unsigned short &pwrong,
                                  unsigned short &pnum);
  void getCNgram(const sequenceData &data, const char *x,
                 unsigned short leftPos, unsigned short rightPos,
                 vector<string> &cNgramList);
  void recursiveGetFJNgram(const sequenceData &data, unsigned short nextPos,
                           vector<string> &fjNgramList,
                           vector<vector<bool> > &dupCheck,
                           unsigned short currentNgram,
                           unsigned short ngramLimit, int &start, int &end);
  void getFJNgram(const sequenceData &data, unsigned short nextPos,
                  vector<string> &fjNgramList);
  float getCFeaturesScore(vector<string> &cNgramList, const char *curr,
                          const char *prev);
  float getJFeaturesScore(const Hyp &hyp);
  float getFJFeaturesScore(vector<string> &fjNgramList, const Hyp &hyp);

  void nbestBeamEval(const sequenceData &data, vector<vectorHyps> &hypsTable);
  // void constructEvalGraph(const sequenceData &data, vector<vectorNode>
  // &graph);
  // void nbestViterbiEval(const sequenceData &data, vector<vectorNode> &graph);

public:
  discriminativeModel(const LocalOpt &lopt);
  virtual ~discriminativeModel(void) = 0;

  virtual void train(const LocalOpt &lopt) = 0;
  void evalDev(float &wacc, float &perr);
  void eval(const LocalOpt &lopt);
  void readPronunceRule(const char *read);
  virtual void readModel(const char *read) = 0;
};

class crfce : public discriminativeModel {
protected:
  pthread_mutex_t addGrad_mutex;
  pthread_mutex_t addUnSupGrad_mutex;
  pthread_mutex_t modelObject_mutex;
  unsigned short numGrad;
  float modelObject;

  inline float logadd(float x, float y);
  void constructGraph(const sequenceData &data, vector<vectorCrfceNode> &graph);
  float calcAlpha(const sequenceData &data, vector<vectorCrfceNode> &graph,
                  vector<vector<string> > &cNgramLists);
  float calcBetaAndGrad(const sequenceData &data,
                        vector<vectorCrfceNode> &graph, float z,
                        vector<vector<string> > &cNgramLists);
  void calcBetaAndGradForCE(vector<vectorCrfceNode> &graph, float z,
                            vector<vector<string> > &cNgramLists);
  void calcBetaAndGradInCorrForCE(vector<vectorCrfceNode> &graph, float zCorr,
                                  float z,
                                  vector<vector<string> > &cNgramLists);

public:
  crfce(const LocalOpt &lopt);
  ~crfce(void);

  void crf(const sequenceData &data);
  void ce(const sequenceData &data);

  void train(const LocalOpt &lopt);
  void readModel(const char *read);
  float operator()(vector<float> &vec);
};

class crfceTask : public Task {
public:
  crfce *instance;
  sequenceData *data;
  crfceTask(crfce *arg_instance, sequenceData *arg_data)
      : instance(arg_instance), data(arg_data) {}

  void run() {
    if (!data->yInfoVec.empty()) {
      instance->crf(*data);
    } else {
      instance->ce(*data);
    }
  }
};

class sarow : public discriminativeModel {
protected:
  char lossFuncTyp;

  bool useFullCovariance;
  bool useOptimizeR;

  float lamda;
  float r;
  float b;
  unsigned short trainNbest;
  unsigned int totalPWrong;
  unsigned int totalPNum;
  // void constructGraph(const sequenceData &data, vector<vectorOnlineNode>
  // &graph);
  // void nbestViterbi(const sequenceData &data, vector<vectorOnlineNode>
  // &graph);
  void nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                 vector<vector<vector<string> > > &cNgramLists,
                 vector<vector<string> > &fjNgramLists,
                 unsigned short trainNum);
  void getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
                      vector<vector<vector<string> > > &cNgramLists,
                      vector<vector<string> > &fjNgramLists);
  void getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
                          char direction, map<string, short> &x,
                          vector<vector<vector<string> > > &cNgramLists,
                          vector<vector<string> > &fjNgramLists);
  void update(const sequenceData &data, unsigned int numTrain);
  void getLocalFeatures(Hyp *hyp, char direction, map<string, short> &x,
                        vector<string> &cNgramList,
                        vector<string> &fjNgramList);
  void localUpdate(const sequenceData &data, unsigned int totalNumTrain);

public:
  sarow(const LocalOpt &lopt);
  ~sarow(void);

  void train(const LocalOpt &lopt);
  void readModel(const char *read);
  float operator()(vector<float> &vec);
};

class ssvm : public discriminativeModel {
protected:
  char lossFuncTyp;

  unsigned short trainNbest;
  unsigned int totalPWrong;
  unsigned int totalPNum;
  void nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                 vector<vector<vector<string> > > &cNgramLists,
                 vector<vector<string> > &fjNgramLists);
  void getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
                      vector<vector<vector<string> > > &cNgramLists,
                      vector<vector<string> > &fjNgramLists);
  void getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
                          char direction, map<string, short> &x,
                          vector<vector<vector<string> > > &cNgramLists,
                          vector<vector<string> > &fjNgramLists);
  void calcGrad(const sequenceData &data);

public:
  ssvm(const LocalOpt &lopt);
  ~ssvm(void);

  void train(const LocalOpt &lopt);
  void readModel(const char *read);
  float operator()(vector<float> &vec);
};

class ssmcw : public discriminativeModel {
protected:
  char lossFuncTyp;
  float b;
  float invC;
  unsigned short trainNbest;
  unsigned int totalPWrong;
  unsigned int totalPNum;

  QPsolver solver;

  void nbestBeam(const sequenceData &data, vector<vectorHyps> &hypsTable,
                 vector<vector<vector<string> > > &cNgramLists,
                 vector<vector<string> > &fjNgramLists, unsigned short hypNum);
  void getAllFeatures(Hyp *hyp, char direction, map<string, short> &x,
                      vector<vector<vector<string> > > &cNgramLists,
                      vector<vector<string> > &fjNgramLists);
  void getAllCorrFeatures(const sequenceData &data, const YInfo &yinfo,
                          char direction, map<string, short> &x,
                          vector<vector<vector<string> > > &cNgramLists,
                          vector<vector<string> > &fjNgramLists);
  void update(const sequenceData &data, unsigned int numTrain);

public:
  ssmcw(const LocalOpt &lopt);
  ~ssmcw(void);

  void train(const LocalOpt &lopt);
  void readModel(const char *read);
  float operator()(vector<float> &vec);
};

