/***************************************************************************
 *   Copyright (C) 2021 by Ben F McLean                                    *
 *   drbenmclean@gmail.com                                                 *
 *                                                                         *
 ***************************************************************************/
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <thread>
#include "cmath"

//see for parallel:
//https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050913X00043/1-s2.0-S1877050913003414/main.pdf
//batch_size=1: stochastic gradient descent - freq weight updates but noisy error estimate (current implementation)
//batch_size=training_count: batch gradient descent - accurate error estimate but less freq weight updates
//1<batch_size<training_count: minibatch gradient descent - balances between noise of error and freq of weight updates

//developed with reference to https://kaifabi.github.io/2021/01/14/micro-mlp.html

//***************
//Hyperparameters
//***************
int n_epochs = 1050;
int n_gradient_accumulation = 1; //1 for stochastic gradient descent
//int n_training_data = 20000;
int batch_size = 500; //number of training data per batch (allows parallelisation one day?)
double learning_rate;
//ideas for learning rate optimisation:
//https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b
//for annealing the learning rate based on current accuracy:
double base_learning_rate = 0.001; //learning rate if accuracy is 0%
double final_learning_rate = 0.0001; //learning rate if accuracy is 100%
int updates = 100; //provide updates every x epochs
double alpha = 0.0001; //hyperparameter for controlling linear weight normalisation
double beta = 0.0001; //hyperparameter for controlling quadratic weight normalisation
double weight_decay = 0.0001; //https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab //not working????

//*******
//Dropout
//*******
//#define NODEWISE_DROPOUT
double input_layer_node_dropout_rate = 0.0; //percent dropout - not implemented yet
double node_dropout_rate = 50.0; //percent dropout
//https://github.com/BVLC/caffe/blob/df412ac0da3e2e7eb194f0c16842fd126496d90d/src/caffe/layers/dropout_layer.cpp
//https://stats.stackexchange.com/questions/207481/dropout-backpropagation-implementation
//https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
//https://agustinus.kristia.de/techblog/2016/06/25/dropout/
//https://github.com/wiseodd/hipsternet/blob/master/hipsternet/neuralnet.py
//https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
//https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
//https://towardsdatascience.com/12-main-dropout-methods-mathematical-and-visual-explanation-58cdc2112293

//********************
//Batch Normalisation:
//********************
double BN_alpha = 0.5;
double BN_epsilon = 0.00001; //small constant
//https://agustinus.kristia.de/techblog/2016/07/04/batchnorm/
//https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm
//https://kevinzakka.github.io/2016/09/14/batch_normalization/

double max_loss = 0.0;

//**************
//Loss functions
//**************
//https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
//(see especially Huber Loss)

//Activation functions
//********************
//default is ReLU, else
//#define ELU
//double elu_alpha = 1.0;
//(ELU *may* benefit from higher learning rate, eg 0.01-0.001, not sure)

//#define TN_PARALLEL
#define NUM_THREADS 4
#define CATEGORICAL

using namespace std;
#include "TN_AI.h"

using namespace std;

int main(/*int argc, char* argv[]*/){
	
	//read commandline params
	//TN_Commandline commandline(argc,argv);
	clock_t start, end, current;
	start = clock();
	
	//**************
	//set up cluster
	//**************
	
	cout << "build cluster..." << endl;
	TN_AI_Cluster mycluster1(2,64,64,64,1);
	
	//********************
	//set up weights array
	//********************

	//Either random "he" initialisation, or read in a pre-calculated W and b file...
	mycluster1.he_initialization();
	//mycluster1.input_W("W_file.bin");
	//mycluster1.input_b("b_file.bin");
	
	//******************
	//load training data
	//******************
	
	TN_AI_Dataset chess_training_data(2,1);
	chess_training_data.read_file("chessboard_training_data.csv");
	//chess_training_data.normalise();
	chess_training_data.normalise_per_feature();
	//chess_training_data.add_noise_to_inputs(0.0,0.02);
	//chess_training_data.output_to_screen();
	
	//********************
	//load validation data
	//********************
	
	TN_AI_Dataset chess_validation_data(2,1);
	chess_validation_data.read_file("chessboard_test_data.csv");
	//chess_validation_data.copy_data_range(chess_training_data); //so that normalisation will be identical between datasets - needed?
	//chess_validation_data.normalise();
	chess_validation_data.normalise_per_feature();
	//chess_validation_data.output_to_screen();
	
	//****************************
	//assess pre-training accuracy
	//****************************
	
	double pre_training_accuracy = mycluster1.check_accuracy(chess_validation_data);
	cout << "pre-training accuracy = " << pre_training_accuracy << "%" << endl;
	
	//***********
	//do training
	//***********
	
	cout << "do training..." << endl << endl;
	
	//prep randomiser
	vector<int> shuffleindex(chess_training_data.size());
	for(unsigned int i=0;i<shuffleindex.size();i++){
		shuffleindex[i] = i;
	}
	
	//double average_loss;
	stringstream running_output;
	running_output << "epoch,valid_accuracy,train_accuracy,loss" << endl;

	for (int i=0; i<n_epochs; ++i) {
		
		double loss=0.0;
		
		//*******************************
		//adjust learning rate each epoch
		//*******************************
		
		//accuracy-based learning rate
		//double accuracy = mycluster1.check_accuracy(chess_validation_data);
		//learning_rate = base_learning_rate - (accuracy/100.0)*(base_learning_rate - final_learning_rate);
		
		//linear learning rate
		//learning_rate = base_learning_rate - ((i+1)/n_epochs)*(base_learning_rate - final_learning_rate);
		
		//exponentially decaying cosine learning rate:
		double maximum = 0.01;
		double wavelength = 100.0;
		double minimum = 0.0;
		double lambda = 3.0;
		learning_rate = ((cos(2*M_PI*i/wavelength)+1)*maximum/2+minimum) * exp(-lambda*i/n_epochs);
		
		//linearly decaying cosine learning rate:
		/*double maximum = 0.01;
		double wavelength = 100.0;
		double minimum = 0.0;
		learning_rate = ((cos(2*M_PI*i/wavelength)+1)*maximum/2+minimum) * (double(n_epochs-i)/double(n_epochs));
		cout << learning_rate << endl;*/
		
		//shuffle the shuffleindex
		random_shuffle(shuffleindex.begin(),shuffleindex.end());


		for (int a=0; a<chess_training_data.size(); ++a) {
			
			mycluster1.set_input(chess_training_data.get_input_datapoint(shuffleindex[a]));
			mycluster1.set_output(chess_training_data.get_output_datapoint(shuffleindex[a]));
#ifdef NODEWISE_DROPOUT
			mycluster1.build_node_dropout_map();
#endif
			//mycluster1.add_noise(0.0, 0.00005 * (double(n_epochs-i)/double(n_epochs))); //linear decay of random noise fn
			mycluster1.feedforward();
			mycluster1.feedbackward();

			if(n_epochs % n_gradient_accumulation == 0 || i == n_epochs-1){ mycluster1.gradient_descent(); }
			
			loss += mycluster1.compute_loss();
        }
		
		if(i==0){
			max_loss = loss;
			loss = 100.0;
		}else{
			loss = loss/max_loss*100.0;
		}
		double valid_accuracy = mycluster1.check_accuracy(chess_validation_data);
		double train_accuracy = mycluster1.check_accuracy(chess_training_data);
		running_output << i << "," << valid_accuracy << "," << train_accuracy << "," << loss << endl;
		
		if (i ==0 || (i+1) % updates == 0 || i == n_epochs-1) {
			
			cout << double(i+1)/double(n_epochs)*100.0 << "% complete:" << endl;
			cout << "epoch = " << i+1 << endl;
			cout << "accuracy = " << valid_accuracy << endl;
			mycluster1.W_stats();
			current = clock();
			int seconds = ((current - start)/CLOCKS_PER_SEC);
			cout << "time = " << seconds << " seconds" << endl;
			cout << "time left = " << int((double(seconds)/double(i+1))*double(n_epochs+1) - seconds) << " seconds" << endl << endl;
			
			mycluster1.output_W("W_file.bin");
			mycluster1.output_b("b_file.bin");
			
			mycluster1.output_essential_data(running_output.str(),"essential_data_0noise.csv");
		}
		
		//average_loss = loss/double(training_data.size());
		
		
        if (i % 10 == 0) {
            //comp_stats(data_valid);
        }

        if (i % 50 == 0) {
			//comp_prediction_landscape();
			
        }
		//cout << "." ;
		
    }
	cout << endl;
	
	end = clock();
	double seconds = ((end - start)/CLOCKS_PER_SEC);
	cout << "run took " << seconds << " seconds" << endl << endl;
	
  	return EXIT_SUCCESS;
	
}
