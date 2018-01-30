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
#include "utility.h"
#include "parseOption.h"
#include <iostream>

char unknownChar = ' ';
char separateChar = '|';
char joinChar = ':';
char delInsChar = '_';
char escapeChar = '\\';

const char interBound = 0x0b;
const char interUnknown = 0x1c;
const char interSeparate = 0x1d;
const char interJoin = 0x1e;
const char interDelIns = 0x1f;

void usage() {

  const char *DOC = "\n\
Slearp (structured learning and prediction) is the structured learning and predict toolkit for tasks such as g2p conversion, based on discriminative leaning.\n\
Copyright (C) 2013, 2014 Keigo Kubo\n\
version 0.96\n\
\n\
usage: ./slearp -t <string> [-d <string>] [-e <string>]\n\
            [-r <string>] [-rr <string>] [-w <string>]\n\
            [-m <crf, sarow, or snarow>] [-useContext <true or false>]\n\
            [-useTrans <true or false>] [-useChain <true or false>]\n\
            [-useJoint <true or false>] [-cn <int>] [-jn <int>]\n\
            [-numOfThread <int>] [-learningRate <float>] [-C <float>]\n\
            [-g <unsigned short>] [ -lossFunc <ploss or sploss>]\n\
            [-regWeight <float>] [-bWeight <float>] [-tn <unsigned short>]\n\
            [-beamSize <unsigned short>] [-on <unsigned short>]\n\
            [-iteration <int>] [-minIter <int>] [-maxTrialIter <unsigned short>]\n\
            [-stopCond <0 or 1>] [-hashFeaTableSize <int>]\n\
            [-hashPronunceRuleTableSize <int>] [-uc <char>]\n\
            [-sc <char>] [-jc <char>] [-dc <char>] [-es <char>]\n\
            [-h] [--help]\n\
\n\
options:\n\
\n\
    -t <string>\n\
        Input training data file which the cosegmentation (string alignment) has already been performed.\n\
\n\
    -d <string>\n\
        Input development data file which the cosegmentation (string alignment) has already been performed.\n\
\n\
    -e <string>\n\
        Input evaluation (test) data file which the cosegmentation (string alignment) has not been performed, and only source sequences or both source sequences and target sequences are included.\n\
\n\
    -r <string>\n\
        Read a model learned by slearp.\n\
\n\
    -rr <string>\n\
        Read conversion rules that includes conversion rules that define the possible conversion to target characters from source characters.\n\
\n\
    -w <string>\n\
        Write a model learned by slearp.\n\
\n\
    -m <crf, sarow, snarow, or ssmcw>\n\
        Select a model learning method. The crf, sarow, snarow, and ssmcw means CRF, structured AROW, structured NAROW, and structured SMCW respectively. (defalut ssmcw)\n\
\n\
    -useContext <true or false>\n\
        Use context features. (defalut true)\n\
\n\
    -useTrans <true or false>\n\
        Use transition features. This feature is only 2-gram for target sequence side in the current implementation.  (defalut false)\n\
\n\
    -useChain <true or false>\n\
        Use chain features. (defalut true)\n\
\n\
    -useJoint <true or false>\n\
        Use joint n-gram features. But this feature does not work in the CRF. (defalut true)\n\
\n\
    -cn <int>\n\
        N-gram size for context and chain features. This size consists of succeeding and preceding context window sizes and a current segment (e.g. 11=5+5+1). (defalut 11)\n\
\n\
    -jn <int>\n\
        Joint n-gram feature size. (defalut 6)\n\
\n\
    -numOfThread <int>\n\
        The number of thread for calculating expectation values in CRF. (defalut 5)\n\
\n\
    -learningRate <float>\n\
        Learning rate for CRF. (defalut 1.0)\n\
\n\
    -C <float>\n\
        Set a hyperparameter C for reguralization in CRF and structured SMCW. This value divided by the number of training data is used in the reguralization in CRF. (defalut 1000)\n\
\n\
    -g <unsigned short>\n\
        The number of gradient stored for limited BFGS to optimize CRF model. (defalut 5)\n\
\n\
    -lossFunc <ploss or sploss>\n\
        Set a loss value type. The ploss use a prediction error in character level as loss value. The sploss use a prediction error in segment level as loss value. (defalut ploss)\n\
\n\
    -regWeight <float>\n\
        Set a hyperparameter r for reguralization in structured AROW. (defalut 500)\n\
\n\
    -bWeight <float>\n\
        Set a hyperparameter b for reguralization in structured NAROW and structured SMCW. (defalut 0.01)\n\
\n\
    -tn <unsigned short>\n\
        Set n for training n-best in structured AROW and structured NAROW. (defalut 5)\n\
\n\
    -beamSize <unsigned short>\n\
        Set beam size for decoding in training and prediction step. The beam size means the number to leave hypotheses in each position of character in target sequence. (defalut 50)\n\
\n\
    -on <unsigned short>\n\
        Set n for n-best output in prediction step. (defalut 1)\n\
\n\
    -iteration <int>\n\
        Set the number of iterations, if you want to stop a learning in a particular iteration.\n\
\n\
    -minIter <int>\n\
        Set the number of minimum iterations. A training is not stopped until it reaches minimum iterations. (defalut 3)\n\
\n\
    -maxTrialIter <unsigned short>\n\
        If slearp is given a development data file, after a degradation of a performance in development data, slearp performs a trial iteration until the number set with this option. When a performance in development data was improved in the trial iteration, the trial iteration is reset to 0. (default 4)\n\
\n\
    -stopCond <0 or 1>\n\
        If slearp is given a development data file, This option works. The stop condition 0 determines to stop a learning based on a degradation of performance in development data. The stop condition 1 determines to stop a learning based on a degradation of average performance between training data and development data. (default 0)\n\
\n\
    -hashFeaTableSize <int>\n\
        Size of hash table to store features. (default 33333333)\n\
\n\
    -hashPronunceRuleTableSize <int>\n\
        Size of hash table to store conversion rules. (default 5929)\n\
\n\
    -uc <char>\n\
        Set unknown character. (default ' ')\n\
\n\
    -sc <char>\n\
        Set separate character. (default '|')\n\
\n\
    -jc <char>\n\
        Set join character. (default ':')\n\
\n\
    -dc <char>\n\
        Set deletion character. (default '_')\n\
\n\
    -es <char>\n\
        Set escape character to avoid regarding non-special character as special character such as deletion character '_'. (default '\\')\n\
\n\
    -h\n\
        Display how to use.\n\
\n\
    --help\n\
        Display how to use.\n\
\n\
";
  cerr << DOC << endl;
}

int main(int argc, char **argv) {
  LocalOpt lopt = { NULL, // train
                    NULL, // dev
                    NULL, // eval
                    NULL, // read
                    NULL, // readRule
                    NULL, // write
                    (char *)"ssmcw", // method
                    true, // useContext
                    false, // useTrans
                    true, // useChain
                    true, // useJoint
                    false, // useFJoint
                    11, // cngram
                    6, // jngram
                    2, // fjngram
                    false, // useOpt
                    5, // numOfThread
                    1.0, // learningRate
                    1000, // hyperparmeter C in CRF and SSCW
                    0.1, // unSupWeight
                    5, // numGrad
                    (const char *)"ploss", // lossFuncTyp
                    false, // useFullCovariance
                    500, // hyperparameter r in SAROW
                    0.01, // hyperparameter b in SNAROW and SSMCW
                    1, // initVar
                    5, // trainNbest
                    false, // useLocal
                    false, // useAvg
                    50, // beamSize
                    1, // outputNbest
                    0, // iter
                    3, // minIter
                    0, // stopCond
                    4, // maxTrialIter
                    33333333, // hashFeaTableSize
                    33333333, // covarianceTableSize
                    777777, // train table size
                    33333, // dev table size
                    5929, // hashPronunceRuleTableSize
  };

  {
    char help = 0;
    parseOption popt(17);

    OptInfo optinfo((char *)"-t", (char *)"string", &lopt.train);
    popt.optionVec.push_back(optinfo);
    popt.optionVec.push_back(
        optinfo.set((char *)"-d", (char *)"string", &lopt.dev));
    popt.optionVec.push_back(
        optinfo.set((char *)"-e", (char *)"string", &lopt.eval));
    popt.optionVec.push_back(
        optinfo.set((char *)"-r", (char *)"string", &lopt.read));
    popt.optionVec.push_back(
        optinfo.set((char *)"-rr", (char *)"string", &lopt.readRule));
    popt.optionVec.push_back(
        optinfo.set((char *)"-w", (char *)"const char *", &lopt.write));
    popt.optionVec.push_back(
        optinfo.set((char *)"-m", (char *)"string", &lopt.method));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useContext", (char *)"bool", &lopt.useContext));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useTrans", (char *)"bool", &lopt.useTrans));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useChain", (char *)"bool", &lopt.useChain));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useJoint", (char *)"bool", &lopt.useJoint));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useFJoint", (char *)"bool", &lopt.useFJoint));
    popt.optionVec.push_back(
        optinfo.set((char *)"-cn", (char *)"unsigned short", &lopt.cngram));
    popt.optionVec.push_back(
        optinfo.set((char *)"-jn", (char *)"unsigned short", &lopt.jngram));
    popt.optionVec.push_back(
        optinfo.set((char *)"-fjn", (char *)"unsigned short", &lopt.fjngram));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useOpt", (char *)"bool", &lopt.useOpt));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-numOfThread", (char *)"unsigned short", &lopt.numOfThread));
    popt.optionVec.push_back(optinfo.set((char *)"-learningRate",
                                         (char *)"float", &lopt.learningRate));
    popt.optionVec.push_back(
        optinfo.set((char *)"-C", (char *)"float", &lopt.C));
    popt.optionVec.push_back(
        optinfo.set((char *)"-uw", (char *)"float", &lopt.unSupWeight));
    popt.optionVec.push_back(
        optinfo.set((char *)"-g", (char *)"unsigned short", &lopt.numGrad));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-lossFunc", (char *)"const char *", &lopt.lossFuncTyp));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-useFullCovariance", (char *)"bool", &lopt.useFullCovariance));
    popt.optionVec.push_back(
        optinfo.set((char *)"-regWeight", (char *)"float", &lopt.r));
    popt.optionVec.push_back(
        optinfo.set((char *)"-bWeight", (char *)"float", &lopt.b));
    popt.optionVec.push_back(
        optinfo.set((char *)"-initVar", (char *)"float", &lopt.initVar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-tn", (char *)"unsigned short", &lopt.trainNbest));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useLocal", (char *)"bool", &lopt.useLocal));
    popt.optionVec.push_back(
        optinfo.set((char *)"-useAvg", (char *)"bool", &lopt.useAvg));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-beamSize", (char *)"unsigned short", &lopt.beamSize));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-on", (char *)"unsigned short", &lopt.outputNbest));
    popt.optionVec.push_back(
        optinfo.set((char *)"-iteration", (char *)"unsigned int", &lopt.iter));
    popt.optionVec.push_back(
        optinfo.set((char *)"-minIter", (char *)"unsigned int", &lopt.minIter));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-stopCond", (char *)"unsigned short", &lopt.stopCond));
    popt.optionVec.push_back(optinfo.set(
        (char *)"-maxTrialIter", (char *)"unsigned short", &lopt.maxTrialIter));
    popt.optionVec.push_back(optinfo.set((char *)"-hashFeaTableSize",
                                         (char *)"unsigned int",
                                         &lopt.hashFeaTableSize));
    popt.optionVec.push_back(optinfo.set((char *)"-covarianceTableSize",
                                         (char *)"unsigned int",
                                         &lopt.covarianceTableSize));
    popt.optionVec.push_back(optinfo.set((char *)"-hashPronunceRuleTableSize",
                                         (char *)"unsigned int",
                                         &lopt.hashPronunceRuleTableSize));
    popt.optionVec.push_back(
        optinfo.set((char *)"-uc", (char *)"char", &unknownChar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-sc", (char *)"char", &separateChar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-jc", (char *)"char", &joinChar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-dc", (char *)"char", &delInsChar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-es", (char *)"char", &escapeChar));
    popt.optionVec.push_back(
        optinfo.set((char *)"-h", (char *)"boolean", &help));
    popt.optionVec.push_back(
        optinfo.set((char *)"--help", (char *)"boolean", &help));

    cerr << "set options are showed in the follows." << endl;
    popt.parseArgv(argv);

    if (help) {
      usage();
      return 1;
    } else if (lopt.read == NULL && lopt.train == NULL) {
      cerr << "Please input trainingset file or model file with -t or -r"
           << endl << endl;

      return 1;
    }
  }

  discriminativeModel *model;
  if (STRCMP(lopt.method, "crfce") || STRCMP(lopt.method, "crf")) {
    lopt.useJoint = false;
    lopt.useFJoint = false;
    model = new crfce(lopt);
    if (lopt.unSupWeight > 1) {
      cerr << "ERROR:Unsupervized weight must be less than 1." << endl;
      exit(EXIT_FAILURE);
    }
  } else if (STRCMP(lopt.method, "sarow") || STRCMP(lopt.method, "snarow")) {
    model = new sarow(lopt);
  } else if (STRCMP(lopt.method, "ssvml1") || STRCMP(lopt.method, "ssvml2")) {
    model = new ssvm(lopt);
  } else if (STRCMP(lopt.method, "ssmcw")) {
    model = new ssmcw(lopt);
  } else {
    cerr << "ERROR:Unknown method:" << lopt.method << endl;
    exit(EXIT_FAILURE);
  }

  if (lopt.readRule != NULL) {
    cerr << "Start to read conversion rule." << endl;
    model->readPronunceRule(lopt.readRule);
  }

  if (lopt.read != NULL) {
    if (lopt.readRule == NULL) {
      cerr << "ERROR:Lack of conversion rule file" << endl;
      cerr << "Please set conversion rule file with -rr option." << endl;
      exit(EXIT_FAILURE);
    }
    cerr << "Start to read model." << endl;
    model->readModel(lopt.read);
  }

  if (lopt.train != NULL) {
    cerr << "Start to train model." << endl;
    if (lopt.write == NULL) {
      string w = lopt.train;
      w += (string) "." + lopt.method;

      if (STRCMP(lopt.method, "crfce") || STRCMP(lopt.method, "crf")) {
        if (lopt.useOpt) {
          w += ".useOpt";
        } else {
          w += ".learningRate" + toString(lopt.learningRate);
          w += ".C" + toString(lopt.C);
          if (STRCMP(lopt.method, "crfce")) {
            w += ".unSupWeight" + toString(lopt.unSupWeight);
          }
          w += ".numGrad" + toString(lopt.numGrad);
        }
      } else if (STRCMP(lopt.method, "sarow") ||
                 STRCMP(lopt.method, "snarow")) {
        w += "." + toString(lopt.lossFuncTyp);
        if (lopt.useFullCovariance) {
          w += ".useFullCov";
        }
        if (lopt.r > 0) {
          w += ".regWeight" + toString(lopt.r);
        } else {
          w += ".useOptR.b" + toString(lopt.b);
        }
        w += ".initVar" + toString(lopt.initVar);
        w += ".trainNbest" + toString(lopt.trainNbest);
        w += ".beam" + toString(lopt.beamSize);
        if (lopt.useLocal) {
          w += ".useLocal";
        }
        if (lopt.useAvg) {
          w += ".useAvg";
        }
      } else if (STRCMP(lopt.method, "ssvml1") ||
                 STRCMP(lopt.method, "ssvml2")) {
        w += "." + toString(lopt.method);
        w += ".C" + toString(lopt.C);
        w += "." + toString(lopt.lossFuncTyp);
        w += ".trainNbest" + toString(lopt.trainNbest);
        w += ".beam" + toString(lopt.beamSize);
        if (lopt.useAvg) {
          w += ".useAvg";
        }
      } else if (STRCMP(lopt.method, "ssmcw")) {
        w += "." + toString(lopt.lossFuncTyp);
        w += ".C" + toString(lopt.C);
        w += ".b" + toString(lopt.b);
        w += ".initVar" + toString(lopt.initVar);
        w += ".trainNbest" + toString(lopt.trainNbest);
        w += ".beam" + toString(lopt.beamSize);
        if (lopt.useAvg) {
          w += ".useAvg";
        }
      }

      if (lopt.useContext || lopt.useChain) {
        if (lopt.useContext) {
          w += ".Context";
        }
        if (lopt.useChain) {
          w += ".Chain";
        }
        w += "." + toString(lopt.cngram) + "cngram";
      }
      if (lopt.useTrans) {
        w += ".Trans";
      }
      if (lopt.useJoint) {
        w += ".Joint." + toString(lopt.jngram) + "jngram";
      }
      if (lopt.useFJoint) {
        w += ".FJoint." + toString(lopt.fjngram) + "fjngram";
      }
      w += ".sc" + toString(lopt.stopCond);
      lopt.write = w.c_str();
      model->train(lopt);
    } else {
      model->train(lopt);
    }
    cerr << "Finish to train model." << endl;
  }

  if (lopt.eval != NULL) {
    cerr << "Start to evaluate model." << endl;
    model->eval(lopt);
  }
}

