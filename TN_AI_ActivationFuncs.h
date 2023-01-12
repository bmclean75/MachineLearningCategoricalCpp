/**************************
Copyright Ben F McLean 2023
   ben@finlaytech.com.au
**************************/


//************************
//ML Activation Functions
//************************

#ifndef TN_AI_ACTIVATIONFUNCS
#define TN_AI_ACTIVATIONFUNCS


//*************************
//Sigmoid and relu fuctions
//*************************

//https://arxiv.org/abs/1511.07289
//elu for a single node
double elu(const double z,const double alpha) {
	double x;
	if (z > 0.0) {
		x = z;
	} else {
		x = alpha * (exp(z)-1.0);
	}
	return x;
};

//elu for a layer of nodes
//(note could incorp relu in this... any faster? If so, same below with sigmoids and primes...)
vector<double> elu(const vector<double>& z,const double alpha){
	vector<double> x(z.size());
	for(unsigned int node=0;node<z.size();node++){
		x[node] = elu(z[node],alpha);
	}
	return x;
};

//elu derivative:
//elu_prime for a single node
double elu_prime(const double z, const double alpha) {
	double x;
	if (z > 0.0) {
		x = 1.0;
	} else {
		x = elu(z,alpha) + alpha;
	}
	return x;
};

//elu_prime for a layer of nodes
vector<double> elu_prime(const vector<double>& z,const double alpha){
	vector<double> x(z.size());
	for(unsigned int node=0;node<z.size();node++){
		x[node] = elu_prime(z[node],alpha);
	}
	return x;
};

//****
//ReLU
//****

//relu for a single node
double relu(const double z) {
	double x;
	if (z > 0.0) {
		x = z;
	} else {
		x = 0.0;
	}
	return x;
};

//relu for a layer of nodes
vector<double> relu(const vector<double>& z){
	vector<double> x(z.size());
	for(unsigned int node=0;node<z.size();node++){
		x[node] = relu(z[node]);
	}
	return x;
};


//references version
void relu(const vector<double>& z, vector<double>& x){
	for(unsigned int node=0;node<z.size();node++){
		if (z[node] > 0.0) {
			x[node] = z[node];
		} else {
			x[node] = 0.0;
		}
		//cout << x[0] << endl;
	}
};

//relu_prime for a single node
double relu_prime(const double z) {
	double x;
	if (z >= 0.0) {
		x = 1.0;
	} else {
		x = 0.0;
	}
	return x;
};

//relu_prime for a layer of nodes
vector<double> relu_prime(const vector<double>& z){
	vector<double> x(z.size());
//#pragma omp for
	for(unsigned int node=0;node<z.size();node++){
		x[node] = relu_prime(z[node]);
	}
	return x;
};

//sigmoid for a single node
double sigmoid(const double z) {
	double x;
	if (z > 0.0) {
		x = 1.0 / (1.0 + exp(-z));
	} else {
		x = exp(z) / (1.0 + exp(z));
	}
	return x;
};

//sigmoid for a layer of nodes
//not using as fails when multithreading
vector<double> sigmoid(const vector<double>& z){
	vector<double> x(z.size());
	for(unsigned int node=0;node<z.size();node++){
		x[node] = sigmoid(z[node]);
	}
	return x;
};

//multithreading-safe version
void sigmoid(const vector<double>& z,vector<double>& x){
	for(unsigned int node=0;node<z.size();node++){
		if (z[node] > 0.0) {
			x[node] = 1.0 / (1.0 + exp(-z[node]));
		} else {
			x[node] = exp(z[node]) / (1.0 + exp(z[node]));
		}
	}
};

//sigmoid_prime for a single node
double sigmoid_prime(const double z) {
	double x;
	if (z > 0.0) {
		x = 1.0 / (1.0 + exp(-z));
	} else {
		x = exp(z) / (1.0 + exp(z));
	}
	return x * (1.0 - x);
};

//sigmoid_prime for a layer of nodes
vector<double> sigmoid_prime(const vector<double>& z){
	vector<double> x(z.size());
//#pragma omp for
	for(unsigned int node=0;node<z.size();node++){
		x[node] = sigmoid_prime(z[node]);
	}
	return x;
};

#endif //TN_AI_ACTIVATIONFUNCS