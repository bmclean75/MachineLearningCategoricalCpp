
#ifndef TN_AI_DATASET
#define TN_AI_DATASET

using namespace std;

//consider making 7 data types 
//https://towardsdatascience.com/7-data-types-a-better-way-to-think-about-data-types-for-machine-learning-939fae99a689
//one-hot encoding:
//https://towardsdatascience.com/the-definitive-way-to-deal-with-continuous-variables-in-machine-learning-edb5472a2538

class TN_AI_Dataset{
	
	//vector<TN_AI_Datapoint> datapoints;
	vector<vector<double>> input_datapoints;
	vector<vector<double>> output_datapoints; //(not all Datapoints will have this)
	
	vector<double> minimums;
	vector<double> maximums;
	bool data_range_set = false;
	int n_input;
	int n_output;
	bool has_output;
	bool input_first = true;
	vector<string> input_names;
	vector<string> output_names;
	
	public:
	
	TN_AI_Dataset(int _n_input){
		n_input = _n_input;
		n_output = 0;
		has_output = false;
		//input_names.resize(n_input);
	};
	
	TN_AI_Dataset(int _n_input,int _n_output){
		n_input = _n_input;
		n_output = _n_output;
		has_output = true;
		//input_names.resize(n_input);
		//output_names.resize(n_output);
	}
	
	void set_input_first(){
		input_first = true;
	};
	
	void set_output_first(){
		input_first = false;
	};
	
	void add_input_datapoint(const vector<double>& input){
		//TN_AI_Datapoint mypoint(input);
		input_datapoints.push_back(input);
	};
	
	void add_output_datapoint(const vector<double>& input, const vector<double>& output){
		//TN_AI_Datapoint mypoint(input,output);
		input_datapoints.push_back(input);
		output_datapoints.push_back(output);
	};
	
	//read in .csv file with column format
	//input1,input2,...,inputN,output1,output2,...,outputM
	void read_file(string filename){
		
		ifstream input_file;
		input_file.open(filename,ios::in);
		if(input_file.bad()){
			std::cerr << "cannot open input file " << filename << std::endl;
			exit(8);
        }
		
		//read in header line
		string line;
		string item;
		getline(input_file,line,'\n'); 
		stringstream ss(line);
		
		if(input_first==true){
			for(int i=0;i<n_input;i++){
				getline(ss, item, ',');
				input_names.push_back(item);
			}
			
			for(int i=0;i<n_output;i++){
				getline(ss, item, ',');
				output_names.push_back(item);
			}
		}else{
			for(int i=0;i<n_output;i++){
				getline(ss, item, ',');
				output_names.push_back(item);
			}
			for(int i=0;i<n_input;i++){
				getline(ss, item, ',');
				input_names.push_back(item);
			}
		}
			
			
		
		//read in data
		while(getline(input_file,line,'\n')){
			double in;
			double out;
			vector<double> input_data;
			vector<double> output_data;
			stringstream ss(line);
			if(input_first==true){
				for(int i=0;i<n_input;i++){
					getline(ss, item, ',');
					stringstream ss_item(item);
					ss_item >> in;
					input_data.push_back(in);
				}
				for(int i=0;i<n_output;i++){
					getline(ss, item, ',');
					stringstream ss_item(item);
					ss_item >> out;
					output_data.push_back(out);
				}
			}else{
				for(int i=0;i<n_output;i++){
					getline(ss, item, ',');
					stringstream ss_item(item);
					ss_item >> out;
					output_data.push_back(out);
				}
				for(int i=0;i<n_input;i++){
					getline(ss, item, ',');
					stringstream ss_item(item);
					ss_item >> in;
					input_data.push_back(in);
				}
			}
			
			input_datapoints.push_back(input_data);
			output_datapoints.push_back(output_data);
		}
		
		input_file.close();
	};
	
	void output_to_screen(){
		
		for(int i=0;i<n_input;i++){
			cout << input_names[i] << " ";
		}
		cout << "| ";
		for(int i=0;i<n_output;i++){
			cout << output_names[i] << " ";
		}
		cout << endl;
		
		
		for(unsigned int point=0;point<input_datapoints.size();point++){
			for(int i=0;i<n_input;i++){
				cout << input_datapoints[point][i] << " ";
			}
			cout << "| ";
			for(int i=0;i<n_output;i++){
				cout << output_datapoints[point][i] << " ";
			}
			cout << endl;
		}
	};
	
	
	void set_data_range(){
		minimums.resize(input_datapoints[0].size());
		maximums.resize(input_datapoints[0].size());
		
		for(unsigned int i=0;i<input_datapoints[0].size();i++){
			minimums[i] = input_datapoints[0][i];
			maximums[i] = input_datapoints[0][i];
		}
		
		for(unsigned int i=0;i<input_datapoints.size();i++){
			for(unsigned int j=0;j<input_datapoints[0].size();j++){
				if(input_datapoints[i][j] < minimums[j]){ minimums[j] = input_datapoints[i][j]; }
				if(input_datapoints[i][j] > maximums[j]){ maximums[j] = input_datapoints[i][j]; }
			}
		}
		data_range_set = true;
	};
	
	/*
	void normalise(){
		if(data_range_set == false){ //if data_range already set, do not reset
			set_data_range();
		}
		for(unsigned int i=0;i<input_datapoints.size();i++){
			for(unsigned int j=0;j<input_datapoints[0].size();j++){
				input_datapoints[i][j] = 2.0 * (input_datapoints[i][j] - minimums[j]) / (maximums[j] - minimums[j]) - 1.0;
			}
		}
	};*/
	
	//normalise per feature, rather than using the global dataset. Allows features to have very different ranges and still norm ok.
	void normalise_per_feature(){
		//indexing by column j, ie per feature of the input
		cout << "input_datapoints[0].size() = " << input_datapoints[0].size() << endl;
		cout << "input_datapoints.size() = " << input_datapoints.size() << endl;
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
	
	void copy_data_range(const TN_AI_Dataset& dataset){
		minimums = dataset.minimums;
		maximums = dataset.maximums;
		data_range_set = true;
	};
	
	inline const vector<double>& get_input_datapoint(int i) const {
		return input_datapoints[i];
	};
	
	inline const vector<double>& get_output_datapoint(int i) const {
		return output_datapoints[i];
	};
	
	int size() const {
		return input_datapoints.size();
	};
	
	void add_noise_to_inputs(double mean, double std_dev){
		
		mt19937 gen(clock());
		normal_distribution<double> rand_normal(mean, std_dev);

		for(unsigned int i=0; i<input_datapoints.size();i++){
			for(unsigned int j=0; j<input_datapoints[i].size();j++){
				input_datapoints[i][j] += rand_normal(gen);
			}
		}
	};
	
	
	
	
};


#endif //TN_AI_DATASET