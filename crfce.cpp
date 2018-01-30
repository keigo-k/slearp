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

crfce::crfce(const LocalOpt &lopt) : discriminativeModel(lopt) {
  unSupFlag = 0;
  pthread_mutex_init(&modelObject_mutex, NULL);
  pthread_mutex_init(&addGrad_mutex, NULL);
  pthread_mutex_init(&addUnSupGrad_mutex, NULL);
  numGrad = lopt.numGrad;
  unsigned int tmp = (unsigned int)sqrt((double)lopt.hashFeaTableSize);
  feas = new hashCrfceFea();
  ((hashCrfceFea *)feas)->initialize(tmp, tmp, numGrad);
}

crfce::~crfce() {
  delete feas;
  pthread_mutex_destroy(&modelObject_mutex);
  pthread_mutex_destroy(&addUnSupGrad_mutex);
  pthread_mutex_destroy(&addGrad_mutex);
}

inline float crfce::logadd(float x, float y) {
  if (y > x) {
    float tmp = x;
    x = y;
    y = tmp;
  }

  float diff = y - x;
  if (diff < -14.5) {
    return x;
  }
  return x + log(1.0 + exp(diff));
}

void crfce::constructGraph(const sequenceData &data,
                           vector<vectorCrfceNode> &graph) {
  int size = data.lenx + 1;
  vector<Pronunce>::iterator it;
  vector<Pronunce>::iterator endit;
  CrfceNode node;

  for (int i = 0; i <= size; ++i) {

    if (i == 0) {
      node.pos = 0;
      node.x = tmpBoundChar;
      node.y = tmpBoundChar;
      node.fProb = 0.0;
      node.bProb = -9999;
      node.nodeProb = 0.0;
      graph[0].push_back(node);

      // setting for next node
      node.fProb = -9999;
      node.edgeNodeProb.reserve(5);
      // node.cNgramList.reserve(500);
    } else if (i == size) {
      node.pos = size;
      node.x = tmpBoundChar;
      node.y = tmpBoundChar;
      node.bProb = 0.0;
      // node.cNgramList.reserve(200);

      graph[size].push_back(node);
    } else {
      for (RuleLen j = 1; j <= rules.maxlenx && i + j <= size; ++j) {
        PronunceRule *rule = rules.refer(data.nosegx[i - 1], j);

        node.pos = i;
        if (rule != NULL) {
          node.x = rule->x;

          endit = rule->pronunceVec.end();
          for (it = rule->pronunceVec.begin(); it != endit; ++it) {
            node.y = it->y;
            graph[i + j - 1].push_back(node);
          }
        } else if (j == 1) {
          node.x = data.nosegx[i - 1];
          node.y = tmpDelInsChar;
          graph[i + j - 1].push_back(node);
        }
      }
    }
  }
}

float crfce::calcAlpha(const sequenceData &data, vector<vectorCrfceNode> &graph,
                       vector<vector<string> > &cNgramLists) {
  vector<vectorCrfceNode>::iterator it = graph.begin();
  vector<vectorCrfceNode>::iterator endit = graph.end();
  ++it;
  int i = 1; // current reached position
  for (; it != endit; ++it) {
    vectorCrfceNode::iterator nodeit = (*it).begin();
    vectorCrfceNode::iterator nodeEndit = (*it).end();

    int k = 0;
    ParaVal nodeProb = 0.0;
    for (; nodeit != nodeEndit; ++nodeit) {
      ParaVal alpha = -9999;

      unsigned short leftPos = (*nodeit).pos - 1;
      int tmpk = i - leftPos;
      if (k != tmpk) {
        k = tmpk;
        cNgramLists.push_back(vector<string>());
        getCNgram(data, (*nodeit).x, leftPos, i + 1, cNgramLists.back());
      }

      (*nodeit).cNgramListIndex = cNgramLists.size() - 1;

      vectorCrfceNode::iterator previt = (graph[leftPos]).begin();
      vectorCrfceNode::iterator prevEndit = (graph[leftPos]).end();
      for (; previt != prevEndit; ++previt) {
        if ((*previt).fProb < -5000) {
          continue;
        }

        ParaVal edgeProb = 0.0;
        if (useTrans || useChain) {
          edgeProb =
              getCFeaturesScore(cNgramLists.back(), (*nodeit).y, (*previt).y);
          (*nodeit).edgeNodeProb.push_back(edgeProb);
        }

        // if(useJoint){
        //
        //}
        alpha = logadd(alpha, (*previt).fProb + edgeProb);
      }

      if ((*nodeit).x == tmpBoundChar) {
        (*nodeit).fProb = alpha;
        return alpha;
      }

      if (useContext) {
        nodeProb = getCFeaturesScore(cNgramLists.back(), (*nodeit).y, NULL);
      }
      (*nodeit).fProb = alpha + nodeProb;
      (*nodeit).nodeProb = nodeProb;
    }

    ++i;
  }
  return 0.0;
}

float crfce::calcBetaAndGrad(const sequenceData &data,
                             vector<vectorCrfceNode> &graph, float z,
                             vector<vector<string> > &cNgramLists) {

  vector<vectorCrfceNode>::iterator endit = graph.begin();
  float zCorr = 0.0;

  YInfoVec::const_iterator yit;
  YInfoVec::const_iterator endyit = data.yInfoVec.end();
  for (yit = data.yInfoVec.begin(); yit != endyit; ++yit) {

    vector<vectorCrfceNode>::iterator it = graph.end();
    --it; // skip -1

    unsigned short i = data.lenx + 1;
    unsigned short corrNodeIndex = i;
    int segIndex = (*yit).lenseg - 1;
    for (; it != endit; --it) {
      vectorCrfceNode::iterator nodeit = (*it).begin();
      vectorCrfceNode::iterator nodeEndit = (*it).end();

      for (; nodeit != nodeEndit; ++nodeit) {
        if ((*nodeit).bProb < -5000) {
          continue;
        }

        char corrNodeFlag = 0;
        unsigned short leftPos = (*nodeit).pos - 1;
        if ((*nodeit).x == tmpBoundChar) {
          corrNodeFlag = 1;
          corrNodeIndex = leftPos;
        } else if (corrNodeIndex == i &&
                   STRCMP((*nodeit).y, (*yit).segy[segIndex]) &&
                   STRCMP((*nodeit).x, (*yit).segx[segIndex])) {
          corrNodeFlag = 1;
          corrNodeIndex = leftPos;
          --segIndex;
        }

        // update gradiant
        ParaVal nodeProb = (*nodeit).bProb;
        ParaVal prob;
        string partfea;
        if ((*nodeit).x != tmpBoundChar && useContext) {
          nodeProb += (*nodeit).nodeProb;
          prob = exp((*nodeit).fProb + (*nodeit).bProb - z);
          // cerr << "Node prob: "<<prob<<endl;
          partfea = tmpTabChar + (*nodeit).y;
          if (corrNodeFlag == 0) {
            // cerr << "node x: "<<(*nodeit).x<< " y: " << (*nodeit).y<< " wrong
            // prob: " << -prob <<endl;
            if (prob > 1.0E-15) {
              vector<string>::iterator feait;
              vector<string>::iterator feaEndit =
                  cNgramLists[(*nodeit).cNgramListIndex].end();

              pthread_mutex_lock(&addGrad_mutex);
              for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                   feait != feaEndit; ++feait) {
                ((hashCrfceFea *)feas)->addGrad(*feait + partfea, -prob);
              }
              pthread_mutex_unlock(&addGrad_mutex);
            }
          } else {
            // cerr << "node x: "<<(*nodeit).x<< " y: " << (*nodeit).y<< " corr
            // prob: " << 1-prob <<endl;
            zCorr += (*nodeit).nodeProb;
            vector<string>::iterator feait;
            vector<string>::iterator feaEndit =
                cNgramLists[(*nodeit).cNgramListIndex].end();
            pthread_mutex_lock(&addGrad_mutex);
            for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                 feait != feaEndit; ++feait) {
              ((hashCrfceFea *)feas)->addGrad(*feait + partfea, 1 - prob);
            }
            pthread_mutex_unlock(&addGrad_mutex);
          }
        }

        unsigned short prevIndex = 0;
        vectorCrfceNode::iterator previt = (graph[leftPos]).begin();
        vectorCrfceNode::iterator prevEndit = (graph[leftPos]).end();
        for (; previt != prevEndit; ++previt) {
          if ((*previt).fProb < -5000) {
            continue;
          }

          ParaVal edgeProb = 0.0;
          if (useTrans || useChain) {
            edgeProb = (*nodeit).edgeNodeProb[prevIndex];
            ++prevIndex;
          }

          // if(useJoint){
          //
          //}

          (*previt).bProb = logadd((*previt).bProb, nodeProb + edgeProb);
          if (useTrans || useChain) {
            // cerr << "fProb: " <<(*previt).fProb<< " nodeProb:"<<nodeProb<<"
            // edgeProb:" << edgeProb <<" z:" << z <<endl;
            prob = exp((*previt).fProb + nodeProb + edgeProb - z);
            if (corrNodeFlag == 1 &&
                (segIndex < 0 ||
                 (STRCMP((*previt).y, (*yit).segy[segIndex]) &&
                  STRCMP((*previt).x, (*yit).segx[segIndex])))) {
              zCorr += edgeProb;
              // cerr << "prev x: "<<(*previt).x<< " y: " << (*previt).y<< "
              // corr prob: " << 1-prob <<endl;
              if (useChain) {
                partfea = tmpTabChar + (*previt).y + tmpTabChar + (*nodeit).y;
                vector<string>::iterator feait;
                vector<string>::iterator feaEndit =
                    cNgramLists[(*nodeit).cNgramListIndex].end();
                pthread_mutex_lock(&addGrad_mutex);
                for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                     feait != feaEndit; ++feait) {
                  ((hashCrfceFea *)feas)->addGrad(*feait + partfea, 1 - prob);
                }
                pthread_mutex_unlock(&addGrad_mutex);
              }

              if (useTrans) {
                partfea = (*previt).y + tmpTabChar + (*nodeit).y;
                pthread_mutex_lock(&addGrad_mutex);
                ((hashCrfceFea *)feas)->addGrad(partfea, 1 - prob);
                pthread_mutex_unlock(&addGrad_mutex);
              }
            } else {
              if (prob > 1.0E-15) {
                // cerr << "prev x: "<<(*previt).x<< " y: " << (*previt).y<< "
                // wrong prob: " << -prob <<endl;
                if (useChain) {
                  partfea = tmpTabChar + (*previt).y + tmpTabChar + (*nodeit).y;
                  vector<string>::iterator feait;
                  vector<string>::iterator feaEndit =
                      cNgramLists[(*nodeit).cNgramListIndex].end();
                  pthread_mutex_lock(&addGrad_mutex);
                  for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                       feait != feaEndit; ++feait) {
                    ((hashCrfceFea *)feas)->addGrad(*feait + partfea, -prob);
                  }
                  pthread_mutex_unlock(&addGrad_mutex);
                }
                if (useTrans) {
                  partfea = (*previt).y + tmpTabChar + (*nodeit).y;
                  pthread_mutex_lock(&addGrad_mutex);
                  ((hashCrfceFea *)feas)->addGrad(partfea, -prob);
                  pthread_mutex_unlock(&addGrad_mutex);
                }
              }
            }
          }
        }
      }
      --i;
    }
  }

  return zCorr;
}

void crfce::calcBetaAndGradForCE(vector<vectorCrfceNode> &graph, float z,
                                 vector<vector<string> > &cNgramLists) {
  vector<vectorCrfceNode>::iterator it = graph.end();
  --it; // skip -1

  vector<vectorCrfceNode>::iterator endit = graph.begin();

  for (; it != endit; --it) {
    vectorCrfceNode::iterator nodeit = (*it).begin();
    vectorCrfceNode::iterator nodeEndit = (*it).end();

    for (; nodeit != nodeEndit; ++nodeit) {
      if ((*nodeit).bProb < -5000) {
        continue;
      }

      // update gradiant
      ParaVal nodeProb = (*nodeit).bProb;
      ParaVal prob;
      string partfea;
      if ((*nodeit).x != tmpBoundChar && useContext) {
        nodeProb += (*nodeit).nodeProb;
        prob = exp((*nodeit).fProb + (*nodeit).bProb - z);
        if (prob > 1.0E-15) {
          partfea = tmpTabChar + (*nodeit).y;
          vector<string>::iterator feait;
          vector<string>::iterator feaEndit =
              cNgramLists[(*nodeit).cNgramListIndex].end();
          pthread_mutex_lock(&addUnSupGrad_mutex);
          for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
               feait != feaEndit; ++feait) {
            ((hashCrfceFea *)feas)->addUnSupGrad(*feait + partfea, -prob);
          }
          pthread_mutex_unlock(&addUnSupGrad_mutex);
        }
      }

      unsigned short prevIndex = 0;
      unsigned short leftPos = (*nodeit).pos - 1;
      vectorCrfceNode::iterator previt = (graph[leftPos]).begin();
      vectorCrfceNode::iterator prevEndit = (graph[leftPos]).end();
      for (; previt != prevEndit; ++previt) {
        if ((*previt).fProb < -5000) {
          continue;
        }

        ParaVal edgeProb = 0.0;
        if (useTrans || useChain) {
          edgeProb = (*nodeit).edgeNodeProb[prevIndex];
          ++prevIndex;
        }

        // if(useJoint){
        //	edgeProb+=getJFeaturesScore(hyp);
        //}

        (*previt).bProb = logadd((*previt).bProb, nodeProb + edgeProb);
        if (useTrans || useChain) {
          prob = exp((*previt).fProb + nodeProb + edgeProb - z);
          if (prob > 1.0E-15) {
            if (useChain) {
              partfea = tmpTabChar + (*previt).y + tmpTabChar + (*nodeit).y;
              vector<string>::iterator feait;
              vector<string>::iterator feaEndit =
                  cNgramLists[(*nodeit).cNgramListIndex].end();
              pthread_mutex_lock(&addUnSupGrad_mutex);
              for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                   feait != feaEndit; ++feait) {
                ((hashCrfceFea *)feas)->addUnSupGrad(*feait + partfea, -prob);
              }
              pthread_mutex_unlock(&addUnSupGrad_mutex);
            }
            if (useTrans) {
              partfea = (*previt).y + tmpTabChar + (*nodeit).y;
              pthread_mutex_lock(&addUnSupGrad_mutex);
              ((hashCrfceFea *)feas)->addUnSupGrad(partfea, -prob);
              pthread_mutex_unlock(&addUnSupGrad_mutex);
            }
          }
        }
      }
    }
  }
}

void crfce::calcBetaAndGradInCorrForCE(vector<vectorCrfceNode> &graph,
                                       float zCorr, float z,
                                       vector<vector<string> > &cNgramLists) {
  vector<vectorCrfceNode>::iterator it = graph.end();
  --it; // skip -1

  vector<vectorCrfceNode>::iterator endit = graph.begin();
  for (; it != endit; --it) {
    vectorCrfceNode::iterator nodeit = (*it).begin();
    vectorCrfceNode::iterator nodeEndit = (*it).end();

    for (; nodeit != nodeEndit; ++nodeit) {
      if ((*nodeit).bProb < -5000) {
        continue;
      }

      // update gradiant
      ParaVal nodeProb = (*nodeit).bProb;
      ParaVal prob;
      string partfea;
      if ((*nodeit).x != tmpBoundChar && useContext) {
        nodeProb += (*nodeit).nodeProb;
        prob = exp((*nodeit).fProb + (*nodeit).bProb - zCorr) -
               exp((*nodeit).fProb + (*nodeit).bProb - z);
        if (prob > 1.0E-15) {
          partfea = tmpTabChar + (*nodeit).y;
          vector<string>::iterator feait;
          vector<string>::iterator feaEndit =
              cNgramLists[(*nodeit).cNgramListIndex].end();
          pthread_mutex_lock(&addUnSupGrad_mutex);
          for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
               feait != feaEndit; ++feait) {
            ((hashCrfceFea *)feas)->addUnSupGrad(*feait + partfea, prob);
          }
          pthread_mutex_unlock(&addUnSupGrad_mutex);
        }
      }

      unsigned short prevIndex = 0;
      unsigned short leftPos = (*nodeit).pos - 1;
      vectorCrfceNode::iterator previt = (graph[leftPos]).begin();
      vectorCrfceNode::iterator prevEndit = (graph[leftPos]).end();
      for (; previt != prevEndit; ++previt) {
        if ((*previt).fProb < -5000) {
          continue;
        }

        ParaVal edgeProb = 0.0;
        if (useTrans || useChain) {
          edgeProb = (*nodeit).edgeNodeProb[prevIndex];
          ++prevIndex;
        }

        // if(useJoint){
        //	edgeProb+=getJFeaturesScore(hyp);
        //}

        (*previt).bProb = logadd((*previt).bProb, nodeProb + edgeProb);
        if (useTrans || useChain) {
          prob = exp((*previt).fProb + nodeProb + edgeProb - zCorr) -
                 exp((*previt).fProb + nodeProb + edgeProb - z);
          if (prob > 1.0E-15) {
            if (useChain) {
              partfea = tmpTabChar + (*previt).y + tmpTabChar + (*nodeit).y;
              vector<string>::iterator feait;
              vector<string>::iterator feaEndit =
                  cNgramLists[(*nodeit).cNgramListIndex].end();
              pthread_mutex_lock(&addUnSupGrad_mutex);
              for (feait = cNgramLists[(*nodeit).cNgramListIndex].begin();
                   feait != feaEndit; ++feait) {
                ((hashCrfceFea *)feas)->addUnSupGrad(*feait + partfea, prob);
              }
              pthread_mutex_unlock(&addUnSupGrad_mutex);
            }

            if (useTrans) {
              partfea = (*previt).y + tmpTabChar + (*nodeit).y;
              pthread_mutex_lock(&addUnSupGrad_mutex);
              ((hashCrfceFea *)feas)->addUnSupGrad(partfea, prob);
              pthread_mutex_unlock(&addUnSupGrad_mutex);
            }
          }
        }
      }
    }
  }
}

void crfce::crf(const sequenceData &data) {
  vector<vectorCrfceNode> graph(data.lenx + 2, vectorCrfceNode());
  constructGraph(data, graph);
  vector<vector<string> > cNgramLists;
  float zAll = calcAlpha(data, graph, cNgramLists);
  float zCorr = calcBetaAndGrad(data, graph, zAll, cNgramLists);
  // cerr << "crf zCorr: "<<zCorr<<" zAll: "<<(zAll *
  // data.yInfoVec.size())<<endl;
  pthread_mutex_lock(&modelObject_mutex);
  modelObject += zCorr - zAll * data.yInfoVec.size();
  pthread_mutex_unlock(&modelObject_mutex);
}

void crfce::ce(const sequenceData &data) {
  char *p = data.nosegx[data.lenx - 1];
  while (*p != '\0') {
    ++p;
  }
  unsigned short size = ABS(p - data.nosegx[0]) + 1;

  // vector<negativeSample> negatives;
  negativeSamplePVec negatives;
  for (int i = 1; i < data.lenx; ++i) {
    if (!(STRCMP(data.nosegx[i - 1], data.nosegx[i]))) {
      char **nosegx;

      if ((nosegx = (char **)malloc(sizeof(char *) * data.lenx +
                                    sizeof(char) * (size))) == NULL) {
        cerr << "ERROR:Can not get memory in malloc.\nYou must need more "
                "memory.\n";
        exit(EXIT_FAILURE);
      }

      nosegx[0] = (char *)(nosegx + data.lenx);
      for (int j = 1; j < data.lenx; ++j) {
        nosegx[j] = nosegx[0] + (data.nosegx[j] - data.nosegx[0]);
      }

      memcpy((void *)nosegx[0], (void *)data.nosegx[0], size);

      p = data.nosegx[i];
      char *tmp = nosegx[i - 1];
      while (*p != '\0') {
        *tmp = *p;
        ++tmp;
        ++p;
      }
      *tmp = '\0';
      ++tmp;

      p = data.nosegx[i - 1];
      while (*p != '\0') {
        *tmp = *p;
        ++tmp;
        ++p;
      }

      negativeSample *negative = new negativeSample();
      negative->nosegx = nosegx;
      negative->lenx = data.lenx;
      negatives.push_back(negative);
    }
  }

  if (negatives.empty()) {
    // cerr << "x: ";
    // for(int i=1; i<data.lenx; ++i){
    //	cerr << data.nosegx[i-1];
    //}
    // cerr << endl;
    return;
  }

  vector<vectorCrfceNode> cgraph(data.lenx + 2, vectorCrfceNode());
  constructGraph(data, cgraph);
  vector<vector<string> > cNgramLists;
  float zCorr = calcAlpha(data, cgraph, cNgramLists);

  float zAll = -9999;
  negativeSamplePVec::iterator it;
  negativeSamplePVec::iterator endit = negatives.end();
  for (it = negatives.begin(); it != endit; ++it) {
    (*it)->graph.assign(data.lenx + 2, vectorCrfceNode());
    constructGraph((*(*it)), (*it)->graph);

    zAll = logadd(zAll, calcAlpha((*(*it)), (*it)->graph, cNgramLists));
  }

  zAll = logadd(zAll, zCorr);
  if (zAll <= zCorr) {
    return;
  }

  for (it = negatives.begin(); it != endit; ++it) {
    calcBetaAndGradForCE((*it)->graph, zAll, cNgramLists);
  }

  calcBetaAndGradInCorrForCE(cgraph, zCorr, zAll, cNgramLists);
  // cerr << "ce zCorr: "<<zCorr<<" zAll: "<<zAll<<endl;
  pthread_mutex_lock(&modelObject_mutex);
  modelObject += zCorr - zAll;
  pthread_mutex_unlock(&modelObject_mutex);
}

void crfce::train(const LocalOpt &lopt) {
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
          unSupFlag = 1;
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

  // prepare queue and worker
  wqueue<Task *> queue(lopt.numOfThread);

  cerr << "Start training..." << endl;
  iter = 0;
  unsigned int trialIter = 0;
  float prevScore = 3.402e+37;
  float prevObject = -3.402e+37;
  float tol = 2.0e-6;
  updateIter = 0;

  string writeModel;

  OptimizePara optimize;
  vector<OptimizePara::Range> ranges;
  vector<float> hypara;
  OptimizePara::Range range = { 1.0, 0.0 };
  if (unSupFlag) {
    hypara.push_back(lopt.unSupWeight);
    hypara.push_back(lopt.learningRate / (1.0 * trainset.numData));
    hypara.push_back(lopt.C / (1.0 * trainset.numData));
    ranges.push_back(range);
    range.upper = 5.0;
    ranges.push_back(range);
    range.upper = 10;
    ranges.push_back(range);
  } else {
    hypara.push_back(lopt.learningRate / (1.0 * trainset.numData));
    hypara.push_back(lopt.C / (1.0 * trainset.numData));
    range.upper = 5.0;
    ranges.push_back(range);
    range.upper = 10;
    ranges.push_back(range);
  }

  while (1) {
    ++iter;
    cerr << "Iteration " << iter << endl;

    modelObject = 0.0;
    float modelScore = 0.0;
    // unsigned int numTrain=1;

    sequenceData *p;
    unsigned int i, j;
    unsigned int hashRow = trainset.hashRow;
    unsigned int hashCol = trainset.hashCol;
    sequenceData ***sequenceDataTable = trainset.sequenceDataTable;

    for (i = 0; i < hashRow; ++i) {
      for (j = 0; j < hashCol; ++j) {
        for (p = sequenceDataTable[i][j]; p != NULL; p = p->next) {

          queue.add(new crfceTask(this, p));

          /*
          if(p->yInfoVec.empty()){
                  // unsupervised training
                  modelObject+=ce(*p);
}else{
                  modelObject+=crf(*p);
}

++numTrain;
if((numTrain%2000)==0){
cerr << "Learned: " << numTrain << endl;
}
          */
        }
      }
    }

    queue.wait_allwork();

    if (modelObject < prevObject) {
      if (unSupFlag) {
        hypara[1] *= 0.75;
      } else {
        hypara[0] *= 0.75;
      }
    }

    cerr << "Object value: " << modelObject << endl;

    if (lopt.dev != NULL) {
      float wacc = 0.0;
      modelScore = 0.0;
      prevObject = modelObject;

      if (lopt.useOpt) {
        cerr << "Optimize parameters by powell method" << endl;
        ++updateIter;
        modelScore =
            optimize.powell(*this, hypara, ranges, 0.4 / log(1 + updateIter));

        if (updateIter == 1) {
          if (unSupFlag) {
            cerr << "Choice supWeight: " << (1 - hypara[0])
                 << " unSupWeight: " << hypara[0]
                 << " learningRate/numOfTrain: " << hypara[1] << endl;
            ((hashCrfceFea *)feas)
                ->updateWeight(1 - hypara[0], hypara[0], hypara[1], hypara[2]);
            ((hashCrfceFea *)feas)
                ->nextSettingInIter1(1 - hypara[0], hypara[0]);
          } else {
            cerr << "Choice learningRate/numOfTrain: " << hypara[0] << endl;
            ((hashCrfceFea *)feas)->updateWeightForSup(hypara[0], hypara[1]);
            ((hashCrfceFea *)feas)->nextSettingInIter1ForSup();
          }
        } else {
          unsigned short storeSize = MIN(updateIter - 1, numGrad);
          if (unSupFlag) {
            cerr << "supWeight: " << (1 - hypara[0])
                 << " unSupWeight: " << hypara[0]
                 << " learningRate/numOfTrain: " << hypara[1]
                 << " C/numOfTrain: " << hypara[2] << endl;
            ((hashCrfceFea *)feas)->updateWeightByLBGFS(
                1 - hypara[0], hypara[0], hypara[1], hypara[2], storeSize);
            ((hashCrfceFea *)feas)
                ->nextSetting(1 - hypara[0], hypara[0], hypara[2], storeSize);
          } else {
            cerr << "Choice learningRate/numOfTrain: " << hypara[0]
                 << " C/numOfTrain: " << hypara[1] << endl;
            ((hashCrfceFea *)feas)
                ->updateWeightForSupByLBGFS(hypara[0], hypara[1], storeSize);
            ((hashCrfceFea *)feas)->nextSettingForSup(hypara[1], storeSize);
          }
        }
      } else {
        // not optimize
        ++updateIter;
        if (updateIter == 1) {
          if (unSupFlag) {
            cerr << "supWeight: " << (1 - hypara[0])
                 << " unSupWeight: " << hypara[0]
                 << " learningRate/numOfTrain: " << hypara[1]
                 << " C/numOfTrain: " << hypara[2] << endl;
            ((hashCrfceFea *)feas)
                ->updateWeight(1 - hypara[0], hypara[0], hypara[1], hypara[2]);
            ((hashCrfceFea *)feas)
                ->nextSettingInIter1(1 - hypara[0], hypara[0]);
          } else {
            cerr << "learningRate/numOfTrain: " << hypara[0]
                 << " C/numOfTrain: " << hypara[1] << endl;
            ((hashCrfceFea *)feas)->updateWeightForSup(hypara[0], hypara[1]);
            ((hashCrfceFea *)feas)->nextSettingInIter1ForSup();
          }
        } else {
          unsigned short storeSize = MIN(updateIter - 1, numGrad);
          if (unSupFlag) {
            cerr << "supWeight: " << (1 - hypara[0])
                 << " unSupWeight: " << hypara[0]
                 << " learningRate/numOfTrain: " << hypara[1]
                 << " C/numOfTrain: " << hypara[2] << endl;
            ((hashCrfceFea *)feas)->updateWeightByLBGFS(
                1 - hypara[0], hypara[0], hypara[1], hypara[2], storeSize);
            ((hashCrfceFea *)feas)
                ->nextSetting(1 - hypara[0], hypara[0], hypara[2], storeSize);
          } else {
            cerr << "learningRate/numOfTrain: " << hypara[0]
                 << " C/numOfTrain: " << hypara[1] << endl;
            ((hashCrfceFea *)feas)
                ->updateWeightForSupByLBGFS(hypara[0], hypara[1], storeSize);
            ((hashCrfceFea *)feas)->nextSettingForSup(hypara[1], storeSize);
          }
        }

        evalDev(wacc, modelScore);
        cerr << "Phoneme error rate in dev: " << modelScore << endl;
        cerr << "Word accuracy in dev: " << wacc << endl;
      }

      if (modelScore > prevScore) {
        cerr << "Bad modelScore:" << modelScore << " prevScore:" << modelScore
             << " trialIter:" << trialIter
             << " lopt.maxTrialIter:" << lopt.maxTrialIter << endl;
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
          ((hashCrfceFea *)feas)->writeFeatures(writeModel.c_str());

          prevScore = modelScore;
        }
      } else {
        cerr << "Good modelScore:" << modelScore << " prevScore:" << modelScore
             << " trialIter:" << trialIter
             << " lopt.maxTrialIter:" << lopt.maxTrialIter << endl;
        if (iter > 1) {
          if (remove(writeModel.c_str())) { // 0 is success
            cerr << "WARNING:Can not remove a previous model file:"
                 << writeModel << endl;
          }
        }

        writeModel = lopt.write;
        writeModel += "." + toString(iter);
        cerr << "Write current model: " << writeModel << endl;
        ((hashCrfceFea *)feas)->writeFeatures(writeModel.c_str());

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

    } else { // stop condtion with trainset
      modelScore = modelObject;
      prevScore = prevObject;

      if (modelScore < prevScore) {
        cerr << "Keep previous model because current model is less performance "
                "than previous model" << endl;
        if (iter > lopt.minIter) {
          break;
        } else {

          ++updateIter;
          if (updateIter == 1) {
            if (unSupFlag) {
              ((hashCrfceFea *)feas)->updateWeight(1 - hypara[0], hypara[0],
                                                   hypara[1], hypara[2]);
              ((hashCrfceFea *)feas)
                  ->nextSettingInIter1(1 - hypara[0], hypara[0]);
              hypara.push_back(lopt.learningRate / (1.0 * trainset.numData));
            } else {
              ((hashCrfceFea *)feas)->updateWeightForSup(hypara[0], hypara[1]);
              ((hashCrfceFea *)feas)->nextSettingInIter1ForSup();
            }
          } else {

            unsigned short storeSize = MIN(updateIter - 1, numGrad);
            if (unSupFlag) {
              ((hashCrfceFea *)feas)->updateWeightByLBGFS(
                  1 - hypara[0], hypara[0], hypara[1], hypara[2], storeSize);
              ((hashCrfceFea *)feas)
                  ->nextSetting(1 - hypara[0], hypara[0], hypara[2], storeSize);
            } else {
              ((hashCrfceFea *)feas)
                  ->updateWeightForSupByLBGFS(hypara[0], hypara[1], storeSize);
              ((hashCrfceFea *)feas)->nextSettingForSup(hypara[1], storeSize);
            }
          }

          if (iter > 1 && remove(writeModel.c_str())) { // 0 is success
            cerr << "WARNING:Can not remove a previous model file:"
                 << writeModel << endl;
          }

          writeModel = lopt.write;
          writeModel += "." + toString(iter);
          cerr << "Write current model: " << writeModel << endl;

          ((hashCrfceFea *)feas)->writeFeatures(writeModel.c_str());

          prevScore = modelScore;
        }

      } else {

        ++updateIter;
        if (updateIter == 1) {
          if (unSupFlag) {
            ((hashCrfceFea *)feas)
                ->updateWeight(1 - hypara[0], hypara[0], hypara[1], hypara[2]);
            ((hashCrfceFea *)feas)
                ->nextSettingInIter1(1 - hypara[0], hypara[0]);
            hypara.push_back(lopt.learningRate / (1.0 * trainset.numData));
          } else {
            ((hashCrfceFea *)feas)->updateWeightForSup(hypara[0], hypara[1]);
            ((hashCrfceFea *)feas)->nextSettingInIter1ForSup();
          }
        } else {
          unsigned short storeSize = MIN(updateIter - 1, numGrad);
          if (unSupFlag) {
            ((hashCrfceFea *)feas)->updateWeightByLBGFS(
                1 - hypara[0], hypara[0], hypara[1], hypara[2], storeSize);
            ((hashCrfceFea *)feas)
                ->nextSetting(1 - hypara[0], hypara[0], hypara[2], storeSize);
          } else {
            ((hashCrfceFea *)feas)
                ->updateWeightForSupByLBGFS(hypara[0], hypara[1], storeSize);
            ((hashCrfceFea *)feas)->nextSettingForSup(hypara[1], storeSize);
          }
        }

        if (iter > 1 && remove(writeModel.c_str())) { // 0 is success
          cerr << "WARNING:Can not remove a previous model file:" << writeModel
               << endl;
        }

        writeModel = lopt.write;
        writeModel += "." + toString(iter);
        cerr << "Write current model: " << writeModel << endl;
        ((hashCrfceFea *)feas)->writeFeatures(writeModel.c_str());

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

  queue.destroy();
}

void crfce::readModel(const char *read) {
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

float crfce::operator()(vector<float> &vec) {
  if (unSupFlag) {
    if (updateIter == 1) {
      cerr << "supWeight: " << (1 - vec[0]) << " unSupWeight: " << vec[0]
           << " learningRate/numOfTrain: " << vec[1] << endl;
      ((hashCrfceFea *)feas)->updateWeight(1 - vec[0], vec[0], vec[1], vec[2]);
    } else {
      cerr << "supWeight: " << (1 - vec[0]) << " unSupWeight: " << vec[0]
           << " learningRate/numOfTrain: " << vec[1]
           << " C/numOfTrain: " << vec[2] << endl;
      ((hashCrfceFea *)feas)->updateWeightByLBGFS(
          1 - vec[0], vec[0], vec[1], vec[2], MIN(updateIter - 1, numGrad));
    }
  } else {
    if (updateIter == 1) {
      cerr << "learningRate/numOfTrain: " << vec[0] << endl;
      ((hashCrfceFea *)feas)->updateWeightForSup(vec[0], vec[1]);
    } else {
      cerr << "learningRate/numOfTrain: " << vec[0]
           << " C/numOfTrain: " << vec[1] << endl;
      ((hashCrfceFea *)feas)->updateWeightForSupByLBGFS(
          vec[0], vec[1], MIN(updateIter - 1, numGrad));
    }
  }

  float modelScore = 0.0;
  float wacc = 0.0;
  evalDev(wacc, modelScore);
  cerr << "Phoneme error rate in dev: " << modelScore << endl;
  cerr << "Word accuracy in dev: " << wacc << endl;

  return modelScore;
}
