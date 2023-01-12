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
#include <cstdio> //for stdin stdout
#include <vector>
#include <cassert>
#include <random>
#include <algorithm>
#include <thread>
#include "cmath"

//test sync

//see for parallel"
//https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050913X00043/1-s2.0-S1877050913003414/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAsaCXVzLWVhc3QtMSJIMEYCIQCW7Jn%2FtfNPCB3%2FyQ%2FLJeS8wmwA8St68w81iDc%2Br6BuUgIhAJWEEJyrhaIr6e%2FN3MwR8vSaMrUKgiG%2FnuBWaWulRb4EKvoDCFQQBBoMMDU5MDAzNTQ2ODY1IgwgCpBh%2FwobWCbaqvgq1wNFvh%2BE9%2FGF1La2fU4Eka9YShiE2eD%2BtgLwBoycr7nqz9pFfvBOEt6YjBnQ0Vfj69INvUknUzK0%2FwcPkFEWc%2F7Iec0%2BXaDQhjgDnu03a5nJQk8Cjp99FgcXf9VR885Cowq7OSGRP1N6s11Fz9K28btdfsvk5EZgM0WjyDU3%2BxxIgGl430PPTqkdHtYVatYSdSi50vjQEfxKjWPJx2V2r75Vxts9Y6jeegO2JK1TRWcektbcdi9mYyY15Ua9N0DknEtpW84aUml8rlBZjZEahvmLVdqlhrYo%2BDwAnIblocPSvpxDos7pYgOjnbjk8e5iw5ue4US32jMUYmebbD6LKdYDWG2GBNEuBIy%2B0B0%2FvgLY2oPj8M5%2FfS7yBvohBCP6ZYxycQtkG7pPASSaa2AaeJPitYvJcRistR9hXiwE0gSVLGbO5WHjoBUXEjOoFtimpGVM201IvHS6TY6AfUCqiGzWmggTGqQYFgjZQO24aY0MmveavbQxn6QKrTsEfa59xj%2F1eElGsfY4hzGyb52QFM0Vp9gUM%2BRGaEg%2BQZbv5NOPbujS4d%2BvbbYgH3KQ3f%2FFdB0nBM62Q%2BuqVWqx5h8Te%2Fu4%2BVXHURztUFE%2Fpu06%2Bm1M57pJ4dgQfCYw67mckAY6pAGsK%2FPfWp%2B1tucyqHAIMNiCb6BIl8Zi6oilaxP4ztpRe6fPQMZ8q5WjU6gONY7IOOFpDLpxIsl4gC50twY%2B0rBxdxK7qr4myaXEjBl9EsJYZMUXmD80AFMC6zvRS6ZXmp0uWocl56TrYIR%2B6JoHTaG3VH8mB3A5mj3ymeRILXXL3pJ0hQh0TEAVARUB4xbu3x50qxy6FUJsc74lfYucI07LkJ02sA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220212T040508Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYTJXA3BIU%2F20220212%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d1777d8bba56eee6ed1219366eac30651242f46065f3968ed4bfd0ed5d74da55&hash=0f29dee58ca988a850a1796f645ce40a8ad195876c7062a2d8408e8dcd3d6255&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050913003414&tid=spdf-aa980085-d40c-4ac4-9c96-3cf547f1cd9c&sid=a9f582a88a8a154b5e9b0902aba66e82c265gxrqa&type=client&ua=4c0004530602065a0200&rr=6dc2efd8febd5a98
//batch_size=1: stochastic gradient descent - freq weight updates but noisy error estimate (current implementation)
//batch_size=training_count: batch gradient descent - accurate error estimate but less freq weight updates
//1<batch_size<training_count: minibatch gradient descent - balances between noise of error and freq of weight updates

//developed with reference to https://kaifabi.github.io/2021/01/14/micro-mlp.html

//********************
//training  parameters
//********************
int n_epochs = 1050;
int n_gradient_accumulation = 1; //1 for stochastic gradient descent
int n_training_data = 20000;
int batch_size = 500; //number of training data per batch (allows parallelisation)
int n_validations = 0;
double learning_rate = 0.0;
//for annealing the learning rate based on current accuracy:
double base_learning_rate = 0.001; //learning rate if accuracy is 0%
double final_learning_rate = 0.0001; //learning rate if accuracy is 100%
int updates = 1; //provide updates every x epochs
double alpha = 0.0; //hyperparameter for controlling linear weight normalisation
double beta = 0.0; //hyperparameter for controlling quadratic weight normalisation

//*******
//Dropout
//*******
#define NODEWISE_DROPOUT
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

//#define TN_PARALLEL //havent found any fn that benefits...
#define NUM_THREADS 4

using namespace std;
#define CATEGORICAL
#include "TN_AI.h"
//#include "matplotlibcpp.h"
//namespace plt = matplotlibcpp;

//https://www.kaggle.com/c/digit-recognizer/


void load_training_data(string filename, vector<vector<double> > &training_data, vector<vector<int> > &training_answers){
	
	std::ifstream in;
	in.open(filename.c_str(),std::ios::in);
	if(in.bad()){
		std::cerr << "cannot open file " << filename << std::endl;
		exit(8);
	}
	
	training_data.resize(42000);
	training_answers.resize(42000);
	
	string line;
	string item;
	int answer;
	double pixel;
	
	getline(in,line,'\n'); //get one line of column headers
	
	for(int i=0;i<42000;i++){ //42000 training samples
		
		getline(in,line,'\n');
		stringstream ss(line);
		getline(ss, item, ',');
		stringstream ss_item(item);
		ss_item >> answer;
		training_answers[i].resize(10,0);
		for(int val=0;val<10;val++){
			if(val==answer){training_answers[i][val] = 1;}
		}
		//cout << "answer = " << answer << endl;
		
		for(int j=0;j<784;j++){ //28x28, or 784 pixels in total
			getline(ss, item, ',');
			stringstream ss_pixel(item);
			ss_pixel >> pixel;
			training_data[i].push_back(pixel);
			//cout << "pixel = " << item << endl;
		}
	}
};


void load_testing_data(string filename, vector<vector<double> > &training_data, vector<vector<int> > &training_answers){
	
	std::ifstream in;
	in.open(filename.c_str(),std::ios::in);
	if(in.bad()){
		std::cerr << "cannot open file " << filename << std::endl;
		exit(8);
	}
	
	training_data.resize(2000);
	training_answers.resize(2000);
	
	string line;
	string item;
	int answer;
	double pixel;
	
	getline(in,line,'\n'); //get one line of column headers
	
	for(int i=0;i<2000;i++){ //42000 training samples
		
		getline(in,line,'\n');
		stringstream ss(line);
		getline(ss, item, ',');
		stringstream ss_item(item);
		ss_item >> answer;
		training_answers[i].resize(10,0);
		for(int val=0;val<10;val++){
			if(val==answer){training_answers[i][val] = 1;}
		}
		//cout << "answer = " << answer << endl;
		
		for(int j=0;j<784;j++){ //28x28, or 784 pixels in total
			getline(ss, item, ',');
			stringstream ss_pixel(item);
			ss_pixel >> pixel;
			training_data[i].push_back(pixel);
			//cout << "pixel = " << item << endl;
		}
	}
};

int main(/*int argc, char* argv[]*/){
	
	//read commandline params
	//TN_Commandline commandline(argc,argv);
	clock_t start, end, current;
	start = clock();
	
	//**************
	//set up cluster
	//**************
	
	cout << "build cluster..." << endl;
	
	TN_AI_Cluster mycluster1(784,300,200,200,10);
	mycluster1.he_initialization();
	
	
	mt19937 gen(clock()); //produces random numbers via the Mersenne twister algorithm
	uniform_real_distribution<double> rand_data(0.0,40.0);
	
	//**********************
	//generate training data
	//**********************
	vector<vector<double> > training_data;
	vector<vector<int> > training_output;
	
	cout << "input training data..." << endl;
	load_training_data("train2.csv",training_data,training_output);
	n_validations = training_data.size();
	cout << "n_validations = " << n_validations << endl;
	//multi-celled output //could be int if categorical... can it be double if not?
	
	//this normalisation sets up a transformation that can also be used on the validation data.
	cout << "normalise training data..." << endl;
	double xmin,xmax;
	cout << "get data range" << endl;
	get_data_range(training_data,xmin,xmax);
	cout << "normalise" << endl;
	normalise(training_data,xmin,xmax);
	
	
	//************************
	//read in validation data
	//************************
	cout << "load validation data..." << endl;
	vector<vector<double> > validation_data;
	vector<vector<int> > validation_output;
	
	load_testing_data("test2.csv",validation_data,validation_output);
	n_validations = validation_data.size();
	cout << "n_validations = " << n_validations << endl;
	//multi-celled output //could be int if categorical... can it be double if not?
	
	//this normalisation sets up a transformation that can also be used on the validation data.
	cout << "normalise training data..." << endl;
	cout << "get data range" << endl;
	get_data_range(validation_data,xmin,xmax);
	cout << "normalise" << endl;
	normalise(validation_data,xmin,xmax);
	
	/*
	vector<vector<double> > validation_data;
	vector<vector<int> > validation_output;
	validation_data = training_data;
	validation_output = training_output;
	*/
	
	cout << "calculate pre-training accuracy..." << endl;
	double pre_training_accuracy = mycluster1.check_accuracy(validation_data,validation_output);
	cout << "pre-accuracy = " << pre_training_accuracy << "%" << endl;
	
	//***********
	//do training
	//***********
	cout << "do training..." << endl << endl;
	
	stringstream running_output;
	running_output << "epoch,valid_accuracy,train_accuracy,loss" << endl;
	
	//prep randomiser
	vector<int> shuffleindex(training_data.size());
	for(unsigned int i=0;i<shuffleindex.size();i++){
		shuffleindex[i] = i;
	}
	
	for (int i=0; i<n_epochs; ++i) {
		
		double loss=0.0;
		
		//adjust learning rate each epoch
		//double accuracy = average(mycluster1.check_accuracy(1000,validation_data,validation_output));
		//double accuracy = mycluster1.check_strict_accuracy(validation_data,validation_output);
		//learning_rate = base_learning_rate - (accuracy/100.0)*(base_learning_rate - final_learning_rate);
		//exponentially decaying cosine learning rate:
		double maximum = 0.01;
		double wavelength = 100.0;
		double minimum = 0.0;
		double lambda = 3.0;
		learning_rate = ((cos(2*M_PI*i/wavelength)+1)*maximum/2+minimum) * exp(-lambda*i/n_epochs);



		//shuffle the shuffleindex
		random_shuffle(shuffleindex.begin(),shuffleindex.end());


		for (unsigned int a=0; a<training_data.size(); ++a) {
			
			mycluster1.set_input(training_data[shuffleindex[a]]);
			mycluster1.set_output(training_output[shuffleindex[a]]);
			
#ifdef NODEWISE_DROPOUT
			mycluster1.build_node_dropout_map_gaussian();
#endif		
			
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
		
		double valid_accuracy = mycluster1.check_accuracy(validation_data,validation_output);
		double train_accuracy = mycluster1.check_accuracy(training_data,training_output);
		running_output << i << "," << valid_accuracy << "," << train_accuracy << "," << loss << endl;

		if ((i+1) % updates == 0 || i == n_epochs-1) {
			double accuracy = mycluster1.check_accuracy(validation_data,validation_output);
			cout << double(i+1)/double(n_epochs)*100.0 << "% complete:" << endl;
			cout << "epoch = " << i+1 << endl;
			cout << "accuracy = " << accuracy << endl;
			current = clock();
			int seconds = ((current - start)/CLOCKS_PER_SEC);
			cout << "time = " << seconds << " seconds" << endl;
			cout << "time left = " << int((double(seconds)/double(i))*double(n_epochs) - seconds) << " seconds" << endl << endl;
			mycluster1.output_essential_data(running_output.str(),"essential_data_gaussdropout_big.csv");
		}
    }
	cout << endl;

	
	end = clock();
	double seconds = ((end - start)/CLOCKS_PER_SEC);
	cout << "run took " << seconds << " seconds" << endl << endl;
	
	
	
  	return EXIT_SUCCESS;
	
}
