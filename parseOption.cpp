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

#include "parseOption.h"

parseOption::parseOption(short numOption) { optionVec.reserve(numOption); }

void parseOption::parseArgv(char **argv) {
  int i, error = 0;
  short size = optionVec.size();

  while (*(++argv)) {
    for (i = 0; i < size || !(error = 1); i++) {
      if (STRCMP(optionVec[i].optname, *argv)) {
        OptInfo optinfo = optionVec[i];

        if (STRCMP(optinfo.type, "int")) {
          if (*(++argv) != NULL) {
            *((int *)optinfo.address) = atoi(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "unsigned int")) {
          if (*(++argv) != NULL) {
            *((unsigned int *)optinfo.address) = atoi(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "short")) {
          if (*(++argv) != NULL) {
            *((short *)optinfo.address) = atoi(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "unsigned short")) {
          if (*(++argv) != NULL) {
            *((unsigned short *)optinfo.address) = atoi(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "float")) {
          if (*(++argv) != NULL) {
            *((float *)optinfo.address) = atof(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "double")) {
          if (*(++argv) != NULL) {
            *((double *)optinfo.address) = atof(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "const char *")) {
          if (*(++argv) != NULL) {
            *((const char **)optinfo.address) = *argv;
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "string")) {
          if (*(++argv) != NULL) {
            *((char **)optinfo.address) = *argv;
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "char")) {
          if (*(++argv) != NULL) {
            if (strlen(*argv) == 1) {
              *((char *)optinfo.address) = **argv;
              cerr << optinfo.optname << ": " << *argv << endl;
            } else {
              cerr << "ERROR:" << optinfo.optname
                   << " option is set char type. \"" << *argv
                   << "\" is string type.\n";
              exit(EXIT_FAILURE);
            }
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "unsigned char")) {
          if (*(++argv) != NULL) {
            *((unsigned char *)optinfo.address) = atoi(*argv);
            cerr << optinfo.optname << ": " << *argv << endl;
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "bool")) {
          if (*(++argv) != NULL) {
            if (**argv == 't') {
              *((char *)optinfo.address) = true;
              cerr << optinfo.optname << ": true" << endl;
            } else if (**argv == 'f') {
              *((char *)optinfo.address) = false;
              cerr << optinfo.optname << ": false" << endl;
            } else {
              cerr << "ERROR:" << optinfo.optname
                   << " option is set true or false.\n";
              exit(EXIT_FAILURE);
            }
          } else {
            cerr << "ERROR:Need to set argument after " << optinfo.optname
                 << " option.\n";
            exit(EXIT_FAILURE);
          }
        } else if (STRCMP(optinfo.type, "boolean")) {
          *((char *)optinfo.address) = 1;
          cerr << optinfo.optname << ": true" << endl;
        } else {
          cerr << "ERROR:Unknown option type:" << optinfo.type << endl;
          exit(EXIT_FAILURE);
        }

        break;
      }
    }
    if (error == 1) {
      cerr << "ERROR:Unknown option:" << *argv << endl;
      exit(EXIT_FAILURE);
    }
  }
}
