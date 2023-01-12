/**************************
Copyright Ben F McLean 2023
ben@finlaytech.com.au
**************************/

#include <vector>

#ifndef TN_AI_COMPUTES
#define TN_AI_COMPUTES

void compute_delta_initial(vector<double>& delta,
							const vector<double>& z,
							const vector<double>& x,
							const vector<double>& y) {
	for (unsigned int i=0; i<delta.size(); ++i) {
#if defined(CATEGORICAL)
		delta[i] = sigmoid_prime(z[i]) * (x[i] - y[i]); //more robust to outliers
		//delta[i] = sigmoid_prime(z[i]) * pow(x[i] - y[i],2); //less robust to outliers?
		//https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/
#elif defined(CONTINUOUS)
		delta[i] = relu_prime(z[i]) * (x[i] - y[i]);
#else
		cout << "in compute_delta_initial(), CATEGORICAL or CONTINUOUS not defined" << endl;
#endif
	}
};

void compute_delta(const vector<vector<double>>& W,
				const vector<double>& z,
				const vector<double>& delta_old,
				vector<double>& delta) {
	
	int W_size = W.size();
	int W_inner_size = W[0].size(); //size is same for all W[i]...
#ifdef FGL_PARALLEL
#pragma omp parallel for
#endif
	for (int j=0; j<W_inner_size; ++j) {
		double tmp = 0.0;
		for (int i=0; i<W_size; ++i) {
			tmp += W[i][j] * delta_old[i];
		}
		delta[j] = relu_prime(z[j]) * tmp;
	}
}

void compute_gradients(vector<vector<double>>& dW, 
						vector<double>& db, 
						const vector<double>& x,
						const vector<double>& delta) {
	int dW_size = dW.size();
	int dW_inner_size = dW[0].size(); //size is same for all W[i]...
#ifdef FGL_PARALLEL
#pragma omp parallel for
#endif
	for (int i=0; i<dW_size; ++i) {
		for (int j=0; j<dW_inner_size; ++j) {
			dW[i][j] = x[j] * delta[i];
		}
		db[i] = delta[i];
	}
};

void descent(vector<vector<double>>& W,vector<double>& b,
	const vector<vector<double>>& dW,const vector<double>& db/*,double L1_norm,double L2_norm*/){
//where the computed gradients are used to update the current weights and biases...
	int W_size = W.size();
	int W_inner_size = W[0].size(); //size is same for all W[i]...

#ifdef FGL_PARALLEL
#pragma omp parallel for
#endif
	for (int i=0; i<W_size; ++i) {
		for (int j=0; j<W_inner_size; ++j) {
#ifdef DEEP_DEBUG			
			cout << "descent" << endl;
			cout << "*******" << endl;
			cout << "W[layer].size() = " << W.size() << endl;
			cout << "W[layer][0].size() = " << W[0].size() << endl;
			cout << "W[layer][i].size() = " << W[i].size() << endl;
			cout << "bW[layer].size() = " << b.size() << endl;
			cout << "dW[layer].size() = " << dW.size() << endl;
			cout << "dbW[layer].size() = " << db.size() << endl;
			cout << "i = " << i << endl;
			cout << "j = " << j << endl << endl;
#endif
			W[i][j] -= learning_rate * dW[i][j]; //works well
			//W[i][j] -= learning_rate * dW[i][j] + 2.0*weight_decay*W[i][j]; //https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab
			//W[i][j] -= learning_rate * dW[i][j] + alpha*L1_norm + beta*L2_norm; //not working
		}
		b[i] -= learning_rate * db[i];
	}
};

//https://machinelearningmastery.com/vector-norms-machine-learning/
//for weight decay, see https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/
//https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab illustrates weight regularisation is conducted in "descent"
void compute_L1_norms(const vector<vector<vector<double>>>& W,vector<double>& L1_norms) {
							
	assert(W.size() == L1_norms.size());
	
	for(unsigned int layer=0; layer<W.size(); ++layer) {
		
		double sum = 0.0;
		for(unsigned int node=0; node<W[layer].size(); node++){
			for (unsigned int weight=0; weight<W[layer][node].size(); weight++) {
				sum += abs(W[layer][node][weight]);
			}
		}
		L1_norms[layer] = sum;
	}
};

//https://machinelearningmastery.com/vector-norms-machine-learning/
//for weight decay, see https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/
void compute_L2_norms(const vector<vector<vector<double>>>& W,vector<double>& L2_norms) {
							
	assert(W.size() == L2_norms.size());
	
	for(unsigned int layer=0; layer<W.size(); ++layer) {
		double sum = 0.0;
		for(unsigned int node=0; node<W[layer].size(); node++){
			for (unsigned int weight=0; weight<W[layer][node].size(); weight++) {
				sum += pow(W[layer][node][weight],2);
			}
		}
		L2_norms[layer] = sqrt(sum);
	}
};

//see build_node_dropout_map in class
vector<double>& node_dropout(vector<double>& z, const vector<double>& map){
	for (unsigned int i=0;i<z.size();i++) {
		z[i] = z[i]*map[i];
		//cout << "z[i] = " << z[i] << ", map[i] = " << map[i] << endl;
	}
    return z;
};

#endif //#TN_AI_COMPUTES