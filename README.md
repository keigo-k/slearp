# What is slearp.

Slearp (structured learning and prediction) is the structured learning and predict toolkit 
for tasks such as g2p conversion, based on discriminative leaning. Currently, slearp is 
implemented to be convenient for a learning of g2p conversion model. However, slearp can 
be applied in another structured learning problems such as morphological analysis, with 
some devisals. 


Given a training dataset which the cosegmentation (string alignment) has already been 
performed, slearp learns a model (e.g. g2p conversion model). Next, given a source sequence 
(e.g. grapheme sequence), slearp predicts a appropriate target sequence (e.g. phoneme 
sequence) with the segmentation of the source and target sequence, using the model learned 
by slearp.


So far, slearp supports L2 norm conditional random fields (CRF), Structured Adaptive Regularization
of Weight Vectors (Structured AROW), Structured Narrow Adaptive Regularization of Weights 
(Structured NAROW), and Structured Soft Margin Confidence Weighted Learning (Structured SMCW) 
as discriminative leaning. Structured SMCW get higher performance than other approches.
The license of slearp is GNU GPL.


# Publications

Show important information such as parameter settings for training methods in sleap. Also, if you don't mind, please cite the following papers when you write a paper using slearp.

For structured AROW:

``Structured Adaptive Regularization of Weight Vectors for a Robust Grapheme-to-Phoneme Conversion Model,'' Keigo Kubo, Sakriani Sakti, Graham Neubig, Tomoki Toda, and Satoshi Nakamura, 2014, IEICE Journal.

``Grapheme-to-phoneme conversion based on adaptive regularization of weight vectors,'' Keigo Kubo, Sakriani Sakti, Graham Neubig, Tomoki Toda, and Satoshi Nakamura, 2013, in Proc. INTERSPEECH, pages 1946-1950.

For structured NAROW:

``Narrow Adaptive Regularization of Weights for Grapheme-to-Phoneme Conversion,'' Keigo Kubo, Sakriani Sakti, Graham Neubig, Tomoki Toda, and Satoshi Nakamura, 2014, in Proc. ICASSP.

For structured SMCW:

``Structured Soft Margin Confidence Weighted Learning for Grapheme-to-Phoneme Conversion,'' Keigo Kubo, Sakriani Sakti, Graham Neubig, Tomoki Toda, and Satoshi Nakamura, 2014, in Proc. INTERSPEECH.


# Install

Go as follow:

```
  $ git clone git@github.com:keigo-k/slearp.git
  $ cd slearp
  $ make
  $ cp slearp <directory included in PATH>
```

# Usage

Example of g2p conversion:

1. Change directory to g2p_example

```
$ cd g2p_example/
```

2. Learn g2p model

```
$ ../slearp -t train.align -d dev.align -cn 5 -jn 2 -w g2p.model
```

train.align and dev.align include training data and development data with alignment respectively, and is formatted as below:

```
S|U|B|S|I|D|I|A|R|I|E|S|'|	S|AH|B|S|IH|D|IY|EH|R|IY|_|Z|_|
P|A:I|N|L|E|S:S|L|Y|	P|EY|N|L|AH|S|L|IY|
S|I|F|U|E|N|T|E|S|	S|IY|F|W|EH|N|T|EH|S|
```

Left side in training and development data file is a source sequence, and the right side is a target sequence. A source sequence and a target sequence are separated by '\t'. The signs '|', ':', and '_' is separate character, join character, and deletion character respectively. Separate character means to segment the string in the point. Join character means not to segment the string in the point. Deletion character means not to exist the corresponding characters in the target side for the characters in the source side. (e.g. "'" -> "_"). 

mpaligner (http://en.sourceforge.jp/projects/mpaligner/releases/ or https://github.com/keigo-k/mpaligner) which is the string alignment tool implementing the many-to-many alignment based on generative model can output the above format. If you only have training and development data without alignment, you can use mpaligner to obtain the training and development data with the alignment.

"-cn 5" and "-jn 2" set n-gram size for context and chain features and n-gram size for joint n-gram feature respectively. The model learned by the above command is output as g2p.model.

3. Predict phoneme sequences with g2p model learned by the above step, or evaluate the g2p model.

```
# Predict phoneme sequences
$ ../slearp -e test.unlabeled.txt -r g2p.model.6 -rr g2p.model.rule -cn 5 -jn 2 > g2p.result
# Evaluate the g2p model
$ ../slearp -e test.labeled.txt -r g2p.model.6 -rr g2p.model.rule -cn 5 -jn 2
```

g2p.model.6 was produced in the step 2 and named as \<model name\>.\<the number of iterations\>. g2p.model.rule was also produced in the step 2 and includes conversion rules that define the possible conversion to target characters from source characters. Phoneme sequences with the alignement predicted by slearp is output to g2p.result. Note that a settings for feature set such as "-cn 5" and  "-jn 2" must be the same as the learning step, and the performance in the g2p model is poor due to the short n-gram settings and the small number of the training data.

Also, this package includes g2p_example/cmudict_task which be a dataset employed as g2p task on our paper.

The description of the options:

```
usage: ./slearp -t <string> [-d <string>] [-e <string>]
			[-r <string>] [-rr <string>] [-w <string>]
			[-m <crf, sarow, or snarow>] [-useContext <true or false>]
			[-useTrans <true or false>] [-useChain <true or false>]
			[-useJoint <true or false>] [-cn <int>] [-jn <int>] 
			[-numOfThread <int>] [-learningRate <float>] [-C <float>] 
			[-g <unsigned short>] [ -lossFunc <ploss or sploss>]
			[-regWeight <float>] [-bWeight <float>] [-tn <unsigned short>]
			[-beamSize <unsigned short>] [-on <unsigned short>] 
			[-iteration <int>] [-minIter <int>] [-maxTrialIter <unsigned short>]
			[-stopCond <0 or 1>] [-hashFeaTableSize <int>]
			[-hashPronunceRuleTableSize <int>] [-uc <char>]
			[-sc <char>] [-jc <char>] [-dc <char>] [-es <char>]
			[-h] [--help]

options:

	-t <string>
		Input training data file which the cosegmentation (string alignment) has already been performed.

	-d <string>
		Input development data file which the cosegmentation (string alignment) has already been performed.

	-e <string>
		Input evaluation (test) data file which the cosegmentation (string alignment) has not been performed, and only source sequences or both source sequences and target sequences are included.

	-r <string>
		Read a model learned by slearp.

	-rr <string>
		Read conversion rules that includes conversion rules that define the possible conversion to target characters from source characters.

	-w <string>
		Write a model learned by slearp.

	-m <crf, sarow, snarow, or ssmcw>
		Select a model learning method. The crf, sarow, snarow, and ssmcw means CRF, structured AROW, structured NAROW, structured SMCW respectively. (defalut ssmcw)

	-useContext <true or false>
		Use context features. (defalut true)

	-useTrans <true or false> 
		Use transition features. This feature is only 2-gram for target sequence side in the current implementation.  (defalut false)

	-useChain <true or false> 
		Use chain features. (defalut true)

	-useJoint <true or false> 
		Use joint n-gram features. But this feature does not work in the CRF. (defalut true)

	-cn <int>
		N-gram size for context and chain features. This size consists of succeeding and preceding context window sizes and a current segment (e.g. 11=5+5+1). (defalut 11)

	-jn <int>
		Joint n-gram feature size. (defalut 6)

	-numOfThread <int>
		The number of thread for calculating expectation values in CRF. (defalut 5)

	-learningRate <float>
		Learning rate for CRF. (defalut 1.0)

	-C <float>
		Set a hyperparameter C for reguralization in CRF and structured SMCW. This value divided by the number of training data is used in the reguralization in CRF (defalut 1000)

	-g <unsigned short>
		The number of gradient stored for limited BFGS to optimize CRF model. (defalut 5)

	-lossFunc <ploss or sploss>
		Set a loss value type. The ploss use a prediction error in character level as loss value. The sploss use a prediction error in segment level as loss value. (defalut ploss)

	-regWeight <float>
		Set a hyperparameter r for reguralization in structured AROW. (defalut 500)

	-bWeight <float>
		Set a hyperparameter b for reguralization in structured NAROW and structured SMCW. (defalut 0.01)

	-tn <unsigned short>
		Set n for training n-best in structured AROW and structured NAROW. (defalut 5)

	-beamSize <unsigned short>
		Set beam size for decoding in training and prediction step. The beam size means the number to leave hypotheses in each position of character in target sequence. (defalut 50)

	-on <unsigned short>
		Set n for n-best output in prediction step. (defalut 1)

	-iteration <int>
		Set the number of iterations, if you want to stop a learning in a particular iteration.

	-minIter <int>
		Set the number of minimum iterations. A training is not stopped until it reaches minimum iterations. (defalut 3)

	-maxTrialIter <unsigned short>
		If slearp is given a development data file, after a degradation of a performance in development data, slearp performs a trial iteration until the number set with this option. When a performance in development data was improved in the trial iteration, the trial iteration is reset to 0. (default 4)	

	-stopCond <0 or 1>
		If slearp is given a development data file, this option works. The stop condition 0 determines to stop a learning based on a degradation of performance in development data. The stop condition 1 determines to stop a learning based on a degradation of average performance between training data and development data. (default 0)

	-hashFeaTableSize <int>
		Size of hash table to store features. (default 33333333)

	-hashPronunceRuleTableSize <int>
		Size of hash table to store conversion rules. (default 5929)
	
	-uc <char>
		Set unknown character. (default ' ')

	-sc <char>
		Set separate character. (default '|')

	-jc <char>
		Set join character. (default ':')

	-dc <char>
		Set deletion character. (default '_')

	-es <char>
		Set escape character to avoid regarding non-special character as special character such as deletion character '_'. (default '\')

	-h
		Display how to use.

	--help
		Display how to use.
```

# Acknowledge
The origin data of training, development, and test data files of g2p conversion and cmudict_task in g2p_example directory is the CMU Pronouncing Dictionary (http://www.speech.cs.cmu.edu/cgi-bin/cmudict).

# Release information

2014/09/12: version 0.96 Added structured SMCW. Also Included g2p_example/cmudict_task which be a dataset employed as g2p task on our paper.

2013/03/05: version 0.95 prototype 

# Bug report

If you find a bug, please send for it to a following email.

E-mail : keigokubo[at]gmail.com  <- Please replace [at] with @ 
