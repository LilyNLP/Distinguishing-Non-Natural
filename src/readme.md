This is the source code to apply our proposed model to defense bert-base against TextFooler attack on the MR dataset, as an example to illustrate how our method is realized.

Due to the size limit of code submission, we cannot include the file counter-fitted-vectors.txt, you can download it from https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view and then put it in './TextFooler/'. We also do not include the folder USE, you can download it from https://tfhub.dev/google/universal-sentence-encoder-large/3 and then put it in './TextFooler/'.

You can then run the run_MR_pipeline.py to see how the pipeline works and get the defense performance. Comments in run_MR_pipeline.py explain the function of each step in the pipeline.

We will publish the complete version of source code containing the realization of our model on other datasets (SST2, IDMB, MNLI) and using other Pre-trained Language Models(RoBERTa, ELECTRA) on github soon.
