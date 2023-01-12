/**************************
Copyright Ben F McLean 2023
ben@finlaytech.com.au
**************************/

#include <vector>


#ifndef TN_AI_MATH
#define TN_AI_MATH


vector<double> matmul(	const vector<vector<double> >& W, 
						const vector<double>& x, const vector<double>& b){
	
	vector<double> z(W.size(), 0.0);
	
	int W_size = W.size();
	int W_inner_size = W[0].size(); //size is same for all W[i]...
#ifdef FGL_PARALLEL
#pragma omp parallel for
#endif
	for (int i=0; i<W_size; ++i) {
		z[i] = 0.0;
		for (int j=0; j<W_inner_size; ++j) {
			z[i] += W[i][j] * x[j];
		}
		z[i] += b[i];
	}
    return z;
};

vector<double>& matmul(	const vector<vector<double> >& W, 
						const vector<double>& x, const vector<double>& b,
						vector<double>& z){
	
	int W_size = W.size();
	int W_inner_size = W[0].size(); //size is same for all W[i]...

#ifdef FGL_PARALLEL
#pragma omp parallel for
#endif
	for (int i=0; i<W_size; ++i) {
		z[i] = 0.0;
		for (int j=0; j<W_inner_size; ++j) {
			z[i] += W[i][j] * x[j];
		}
		z[i] += b[i];
	}
    return z;
};

double average(const vector<double>& myvec){
	
	double cumulative = 0.0;
	for(unsigned int i=0;i<myvec.size();i++){
		cumulative += myvec[i];
	}
	return cumulative/myvec.size();

};

#endif //#TN_AI_MATH