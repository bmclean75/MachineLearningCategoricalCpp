
/**************************
Copyright Ben F McLean 2023
ben@finlaytech.com.au
**************************/


#ifndef TN_AI_NORMALIZATION
#define TN_AI_NORMALIZATION

//*********************************
//Normalization and Standardization
//*********************************

//Note, in NORMALISATION, minimum and maximum value of features are used for scaling
//in STANDARDISATION, mean and standard deviation is used for scaling.

void get_data_range(const vector<vector<double> >& datapoints,double& xmin,double& xmax,double& ymin,double& ymax){
	xmax=datapoints[0][0];
	xmin=datapoints[0][0];
	ymax=datapoints[0][1];
	ymin=datapoints[0][1];
	
	for(unsigned int i=0;i<datapoints.size();i++){
		if(datapoints[i][0] < xmin){ xmin = datapoints[i][0]; }
		if(datapoints[i][0] > xmax){ xmax = datapoints[i][0]; }
		if(datapoints[i][1] < ymin){ ymin = datapoints[i][1]; }
		if(datapoints[i][1] > ymax){ ymax = datapoints[i][1]; }
	}
};

void get_data_range(const vector<vector<double> >& datapoints,double& xmin,double& xmax){
	xmax=datapoints[0][0];
	xmin=datapoints[0][0];
	
	for(unsigned int i=0;i<datapoints.size();i++){
		for(unsigned int j=0;j<datapoints[i].size();j++){
			if(datapoints[i][j] < xmin){ xmin = datapoints[i][j]; }
			if(datapoints[i][j] > xmax){ xmax = datapoints[i][j]; }
		}
	}
};

void normalise(vector<vector<double> >& datapoints,const double xmin,const double xmax,const double ymin,const double ymax){
	for(unsigned int i=0;i<datapoints.size();i++){
		datapoints[i][0] = 2.0 * (datapoints[i][0] - xmin) / (xmax - xmin) - 1.0;
		datapoints[i][1] = 2.0 * (datapoints[i][1] - ymin) / (ymax - ymin) - 1.0;
	}
};

void normalise(vector<vector<double> >& datapoints,const double xmin,const double xmax){
	for(unsigned int i=0;i<datapoints.size();i++){
		for(unsigned int j=0;j<datapoints[i].size();j++){
			datapoints[i][j] = 2.0 * (datapoints[i][j] - xmin) / (xmax - xmin) - 1.0;
		}
	}
};

void normalise_per_feature(vector<vector<double> >& input_datapoints){
	//indexing by column j, ie per feature of the input
	for(unsigned int j=0;j<input_datapoints[0].size();j++){
		double max = input_datapoints[0][j];
		double min = input_datapoints[0][j];
		for(unsigned int i=1;i<input_datapoints.size();i++){
			if(input_datapoints[i][j] < min){ min = input_datapoints[i][j]; }
			if(input_datapoints[i][j] > max){ max = input_datapoints[i][j]; }
		}
		for(unsigned int i=1;i<input_datapoints.size();i++){
			input_datapoints[i][j] = 2.0 * (input_datapoints[i][j] - min) / (max - min) - 1.0;
		}
	}
};

void normalise(vector<vector<double> >& datapoints){
	double xmax=datapoints[0][0];
	double xmin=datapoints[0][0];
	double ymax=datapoints[0][1];
	double ymin=datapoints[0][1];
	
	for(unsigned int i=0;i<datapoints.size();i++){
		if(datapoints[i][0] < xmin){ xmin = datapoints[i][0]; }
		if(datapoints[i][0] > xmax){ xmax = datapoints[i][0]; }
		if(datapoints[i][1] < ymin){ ymin = datapoints[i][1]; }
		if(datapoints[i][1] > ymax){ ymax = datapoints[i][1]; }
	}
	
	for(unsigned int i=0;i<datapoints.size();i++){
		datapoints[i][0] = 2.0 * (datapoints[i][0] - xmin) / (xmax - xmin) - 1.0;
		datapoints[i][1] = 2.0 * (datapoints[i][1] - ymin) / (ymax - ymin) - 1.0;
	}
};

void standardise(vector<double>& datapoints){
	double mean = 0.0;
	double sd = 0.0;
	
	for(unsigned int i=0;i<datapoints.size();i++){
		mean += datapoints[i];
	}
	mean = mean/double(datapoints.size());
	
	for(unsigned int i=0;i<datapoints.size();i++){
		sd += pow(datapoints[i]-mean,2.0);
	}
	sd = sqrt(sd/datapoints.size());
	
	//data can be normalized by subtracting the mean (µ) of each feature 
	//and dividing by the standard deviation (σ)
	for(unsigned int i=0;i<datapoints.size();i++){
		datapoints[i] = (datapoints[i]-mean)/sd;
	}
	
	//cout << mean << " " << sd << endl;
	
};



//normalise W per node - not as good as normalising W per layer?
void nodewise_minibatch_standardise(vector<vector<double> >& W){
	for(unsigned int i=0;i<W.size();i++){
		standardise(W[i]);
	}
};

#endif //TN_AI_NORMALIZATION