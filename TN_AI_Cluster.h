/**************************
Copyright Ben F McLean 2022
ben@finlaytech.com.au
**************************/

//********************
//class TN_AI_Cluster
//********************

#include <vector>

#ifndef TN_AI_CLUSTER
#define TN_AI_CLUSTER


class TN_AI_Cluster{
	
	//first vector contains layers of nodes
	//second vector contains node values within a layer
	//weights are unique in having a vector of values for each node in each layer
	vector<vector<vector<double> > > W; //weights
	vector<vector<vector<double> > > dW; //gradient of weights
	vector<vector<double> > x; //post-activation values
	vector<vector<double> > b; //biases
	vector<vector<double> > db; //derivatives of biases
	vector<vector<double> > z; //pre-activation values
	vector<vector<double> > delta; //differences?
	vector<double> input; //input data, whether training or actual
	vector<double> L1_norms; //https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/
	vector<double> L2_norms; //https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/
	vector<double> y; //is the true training value (only used once, in feedbackward compute_delta_initial)
	
	//for batch norm:
	vector<double> BN_beta, BN_gamma; //learnable parameters, per layer
	vector<double> BN_moving_ave_mean, BN_moving_ave_var; //saved parameters
	
	//for dropout:
	vector<vector<double> > node_dropout_map;
	
	
	int n_layers=0; //number of layers in cluster, inc input, hidden, and output layers
	int n_inputnodes=0;
	int n_outputnodes=0;
	
	public:
	
	TN_AI_Cluster(int n_input,int n_layer1,int n_layer2=0,int n_layer3=0,int n_layer4=0,int n_layer5=0,int n_layer6=0, int n_layer7=0){
		
		
		n_inputnodes = n_input;
		vector<int> set_of_layer_ns({n_layer1,n_layer2,n_layer3,n_layer4,n_layer5,n_layer6,n_layer7});
		//set_of_layer_ns.push_back(n_layer1); //start to fill a vector of layer n's
		if(n_layer2==0){n_layers = 1; n_outputnodes = n_layer1;} //minimum possible is 1 layer...
		else if(n_layer3==0){n_layers = 2; n_outputnodes = n_layer2;}
		else if(n_layer4==0){n_layers = 3; n_outputnodes = n_layer3;}
		else if(n_layer5==0){n_layers = 4; n_outputnodes = n_layer4;}
		else if(n_layer6==0){n_layers = 5; n_outputnodes = n_layer5;}
		else if(n_layer7==0){n_layers = 6; n_outputnodes = n_layer6;}
		else if(n_layer7!=0){n_layers = 7; n_outputnodes = n_layer7;}
		
		cout << "n layers = " << n_layers << endl;
		cout << "n input nodes = " << n_inputnodes << endl;
		cout << "n hidden nodes = ";
		for (int i=0;i<n_layers-1;i++){
			if(set_of_layer_ns[i]!=0){
				cout << set_of_layer_ns[i] << ", ";
			}
		}
		cout << endl << "n output nodes = " << n_outputnodes << endl;
		
		//build all vectors
		y.resize(n_outputnodes,0.0);
		input.resize(n_inputnodes,0.0);
		
		W.resize(n_layers);
		dW.resize(n_layers);
		x.resize(n_layers);
		b.resize(n_layers);
		db.resize(n_layers);
		z.resize(n_layers);
		delta.resize(n_layers);
		L1_norms.resize(n_layers);
		L2_norms.resize(n_layers);
		
		BN_beta.resize(n_layers,0.0); //for Batch Normalisation
		BN_gamma.resize(n_layers,1.0); //for Batch Normalisation
		BN_moving_ave_mean.resize(n_layers,0.0); //for Batch Normalisation
		BN_moving_ave_var.resize(n_layers,0.0); //for Batch Normalisation
		
		node_dropout_map.resize(n_layers);
		
		for(int layer=0;layer<n_layers;layer++){
			
			if(layer==0){
				W[layer].resize(set_of_layer_ns[layer], vector<double>(n_inputnodes, 0.0));
				dW[layer].resize(set_of_layer_ns[layer], vector<double>(n_inputnodes, 0.0));
			}else{
				W[layer].resize(set_of_layer_ns[layer], vector<double>(set_of_layer_ns[layer-1], 0.0));
				dW[layer].resize(set_of_layer_ns[layer], vector<double>(set_of_layer_ns[layer-1], 0.0));
			}
			
			x[layer].resize(set_of_layer_ns[layer],0.0);
			b[layer].resize(set_of_layer_ns[layer],0.0);
			db[layer].resize(set_of_layer_ns[layer],0.0);
			z[layer].resize(set_of_layer_ns[layer],0.0);
			delta[layer].resize(set_of_layer_ns[layer],0.0);
			
			node_dropout_map[layer].resize(set_of_layer_ns[layer],1.0);
			
		}
	};
	
	void set_input(const vector<double>& input_data){
		//cout << "input.size() = " << input.size() << ", input_data.size() = " << input_data.size() << endl;
		assert(input.size() == input_data.size());
		input = input_data;
	};
	
	void set_output(const vector<double>& output_answers){
		assert(output_answers.size()==y.size());
		for(unsigned int i=0;i<y.size();i++){
			y[i] = output_answers[i];
		}
	};
	
	void set_output(const vector<int>& output_answers){
		assert(output_answers.size()==y.size());
		for(unsigned int i=0;i<y.size();i++){
			y[i] = output_answers[i];
		}
	};
	
	vector<double>& get_output(){
		return x[n_layers-1];
	};
	
	int get_n_layers(){
		return n_layers;
	};
	
	int get_layer(){
		return n_layers;
	};
	
	vector<vector<vector<double> > >& get_W(){
		return W;
	};
	
	vector<vector<double> >& get_W(int layer){
		return W[layer];
	};
	
	//*******
	//Dropout
	//*******
	
	void build_node_dropout_map(){
		mt19937 gen(clock());
		uniform_real_distribution<double> rand(0.0,100.0);
		
		for(unsigned int i=0;i<node_dropout_map.size();i++){
			for(unsigned int j=0;j<node_dropout_map[i].size();j++){
				node_dropout_map[i][j] = 100.0/(100.0-node_dropout_rate); //weights z values
				//note input layer could theoretically have dropout (closer to 1.0)
				//but this has not been implemented. Only hidden layers have dropout atm.
				if(rand(gen)<node_dropout_rate){
					node_dropout_map[i][j]=0.0;
				}
			}
		}
		/*
		cout << "building node dropout map..." << endl;
		for(unsigned int i=0;i<node_dropout_map.size();i++){
			for(unsigned int j=0;j<node_dropout_map[i].size();j++){
				cout << node_dropout_map[i][j] << " " ;
			}
			cout << endl;
		}*/
	};
	
	void build_node_dropout_map_gaussian(){
		double mean = 1.0;
		double std_dev = 0.9;
		mt19937 gen(clock()); //produces random numbers via the Mersenne twister algorithm
		normal_distribution<double> rand_normal(mean, std_dev);
		//cout << "building node dropout map" << endl;

		for(unsigned int i=0;i<node_dropout_map.size();i++){
			for(unsigned int j=0;j<node_dropout_map[i].size();j++){
				node_dropout_map[i][j] = rand_normal(gen);
			}
		}
	};
	
	//Batch Normalization - not working, or at least not helping...
	//*******************
	
	//normalise W per layer - suggested for deep models - https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks
	//Batch Norm Explained Visually — How it works, and why neural networks need it
	//https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
	void batch_norm(int layer){
		
		//indexing by column j, ie per feature of the input, not per sample
		for(unsigned int j=0;j<W[layer][0].size();j++){
			
			double mean = 0.0;
			double sd = 0.0;
			
			//calc mean
			for(unsigned int i=0;i<W[layer].size();i++){
				mean += W[layer][i][j];
			}
			mean = mean/double(W[layer].size());
			BN_moving_ave_mean[layer] = BN_alpha*BN_moving_ave_mean[layer] + (1.0 -  BN_alpha)*mean; //collects whole-dataset stat
			
			//calc SD
			for(unsigned int i=0;i<W[layer].size();i++){
				sd += pow(W[layer][i][j]-mean,2.0)/double(W[layer].size());
			}
			sd = sqrt(sd + BN_epsilon); //small constant Epsilon ensures we dont get sd = 0.0, and subsequently divide by 0...
			BN_moving_ave_var[layer] = BN_alpha*BN_moving_ave_var[layer] + (1.0 - BN_alpha)*sd; //collects whole-dataset stat
			
			//update and transform layer Weights
			for(unsigned int i=0;i<W[layer].size();i++){
				// data can be standardised by subtracting the mean (µ) of each feature and dividing by the standard deviation (σ)
				W[layer][i][j] = (W[layer][i][j]-mean)/sd;
				W[layer][i][j] = BN_gamma[layer]*W[layer][i][j] + BN_beta[layer];
				
			}
		}
	};
	
	void W_stats(){
		
		double absmean = 0.0;
		double sd = 0.0;
		int n = 0;
		
		for(unsigned int i=0;i<W.size();i++){
			for(unsigned int j=0;j<W[i].size();j++){
				for(unsigned int k=0;k<W[i][j].size();k++){
					
					absmean += abs(W[i][j][k]);
					n++;
				}
			}
		}
		
		absmean = absmean/double(n);
		
		for(unsigned int i=0;i<W.size();i++){
			for(unsigned int j=0;j<W[i].size();j++){
				for(unsigned int k=0;k<W[i][j].size();k++){
					
					sd += pow(abs(W[i][j][k])-absmean,2.0)/double(n);
				}
			}
		}
		
		cout << "W abs_mean = " << absmean << ", SD = " << sd << endl;
	};
	
	//https://agustinus.kristia.de/techblog/2016/07/04/batchnorm/
	//https://kevinzakka.github.io/2016/09/14/batch_normalization/
	void batch_norm_backward(int layer){
		
		//intermediate partial derivatives
		//dxhat = dout * gamma
		//double dxhat = delta[layer] * BN_gamma[layer];
		
		//indexing by column j, ie per feature of the input, not per sample
		for(unsigned int j=0;j<W[layer][0].size();j++){
			
			//final partial derivatives
			//dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0) - x_hat*np.sum(dxhat*x_hat, axis=0))
			//dbeta = np.sum(dout, axis=0)
			//dgamma = np.sum(x_hat*dout, axis=0)
			
			
		}
	};
		
	//for batch norm during inference see
	//https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
	
	
	void add_noise(double mean, double std_dev){
		mt19937 gen(clock()); //produces random numbers via the Mersenne twister algorithm
		normal_distribution<double> rand_normal(mean, std_dev);
		
		for (unsigned int i=0; i<W.size(); ++i) {
			for (unsigned int j=0; j<W[i].size(); ++j) {
				for (unsigned int k=0; k<W[i][j].size(); ++k) {
					W[i][j][k] += rand_normal(gen);
				}
			}
		}
	};
	
	//"he" initialisation
	//*******************
	//see https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
	//and https://arxiv.org/abs/1502.01852 - https://arxiv.org/pdf/1502.01852.pdf just after equation (10).
	void he_initialization() {
		double mean = 0.0;
		double std_dev = sqrt(2.0 / static_cast<double>(W[0].size()));
		mt19937 gen(clock()); //produces random numbers via the Mersenne twister algorithm
		normal_distribution<double> rand_normal(mean, std_dev);

		for (unsigned int i=0; i<W.size(); ++i) {
			for (unsigned int j=0; j<W[i].size(); ++j) {
				//b[i][j] = rand_normal(gen); //initialising biases seems to have no effect
				for (unsigned int k=0; k<W[i][j].size(); ++k) {
					W[i][j][k] = rand_normal(gen);
					//cout << "W: " << W[i][j][k] << endl;
				}
			}
		}
	};
	
	
	
	void feedforward() { //(with dropout where applicable - not for validation/prediction)
		
		//batch_norm(0);
		matmul(W[0], input, b[0], z[0]);
		relu(z[0],x[0]);
		//(nodewise dropout not applied to input layer)

		for(int layer=1;layer<n_layers-1;layer++){
			
			//batch_norm(layer);
			matmul(W[layer], x[layer-1], b[layer], z[layer]);
			relu(z[layer],x[layer]);
#ifdef NODEWISE_DROPOUT
			node_dropout(x[layer],node_dropout_map[layer]);
#endif
		}
		
		//batch_norm(n_layers-1);
		//(note nodewise dropout not applied to output layer)
		matmul(W[n_layers-1], x[n_layers-2], b[n_layers-1], z[n_layers-1]);
		
#if defined(CATEGORICAL)
		sigmoid(z[n_layers-1],x[n_layers-1]);
#elif defined (CONTINUOUS)
		//x[n_layers-1] = z[n_layers-1];
		relu(z[n_layers-1],x[n_layers-1]);
#else
		cout << "in feedforward(), CATEGORICAL or CONTINUOUS not defined" << endl;
#endif
	};
	
	void feedforward_validation(){ //no dropout!
		
		matmul(W[0], input, b[0], z[0]);
		relu(z[0],x[0]);

		for(int layer=1;layer<n_layers-1;layer++){
			matmul(W[layer], x[layer-1], b[layer], z[layer]);
			relu(z[layer],x[layer]);
		}
		
		matmul(W[n_layers-1], x[n_layers-2], b[n_layers-1], z[n_layers-1]);
#if defined(CATEGORICAL)
		sigmoid(z[n_layers-1],x[n_layers-1]);
#elif defined (CONTINUOUS)
		x[n_layers-1] = z[n_layers-1];
#else
		cout << "in feedforward_validation(), CATEGORICAL or CONTINUOUS not defined" << endl;
#endif
		
	};
	
	
	
	
	void feedbackward() {
		//https://stats.stackexchange.com/questions/219236/dropout-forward-prop-vs-back-prop-in-machine-learning-neural-network
		
		int layer = n_layers-1;
		
		compute_delta_initial(delta[layer],z[layer],x[layer],y);
		compute_gradients(dW[layer],db[layer],x[layer-1],delta[layer]);

		for(layer=n_layers-2;layer>0;layer--){
			compute_delta(W[layer+1],z[layer],delta[layer+1],delta[layer]);
#ifdef NODEWISE_DROPOUT
			node_dropout(delta[layer],node_dropout_map[layer]);
#endif
			compute_gradients(dW[layer],db[layer],x[layer-1],delta[layer]);

		}
	
		compute_delta(W[1],z[0],delta[1],delta[0]);
		compute_gradients(dW[0],db[0],input,delta[0]);

	};
	
	void gradient_descent(){
		
		compute_L1_norms(W,L1_norms);
		compute_L2_norms(W,L2_norms);
		
		for(int layer=n_layers-1;layer>-1;layer--){
			
#ifdef DEEP_DEBUG			
			cout << "gradient_descent" << endl;
			cout << "****************" << endl;
			cout << "W.size() = " << W.size() << endl;
			cout << "b.size() = " << b.size() << endl;
			cout << "dW.size() = " << dW.size() << endl;
			cout << "db.size() = " << db.size() << endl;
			cout << "layer = " << layer << endl << endl;
#endif
			
			//descent(W[layer],b[layer],dW[layer],db[layer]); //****************************************************test
			descent(W[layer],b[layer],dW[layer],db[layer]/*,L1_norms[layer],L2_norms[layer]*/);
		}
	};
	
	
	//checks that ALL outputs are correct...
	//depreciated
	double check_accuracy(const vector<vector<double>>& validation_data,const vector<vector<int>>& validation_output){
		
		int correct_count = 0;
		int n_validations = validation_data.size();
		vector<double> computed_answers = this->get_output();
		
		for(int sample=0;sample<n_validations;sample++){
			//assign starting values
			this->set_input(validation_data[sample]);
			
			//forward model
			this->feedforward_validation();
			//mycluster.visualise();
			
			vector<double> computed_answers = this->get_output();
			vector<int> output(computed_answers.size());
			
			//cout << "computed answers size = " << computed_answers.size() << endl;
			for(unsigned int i=0;i<computed_answers.size();i++){
				if(computed_answers[i] > 0.5){
					output[i] = 1;
				}else{
					output[i] = 0;
				}
			}
			
			int n_parts_correct = 0;
			for(unsigned int i=0;i<computed_answers.size();i++){
				if(output[i] == validation_output[sample][i]){
					n_parts_correct +=1;
				}
			}
			if(n_parts_correct==int(output.size())){correct_count += 1;} //only if ALL parts are correct, item is correct
		}
	
		double accuracy = double(correct_count)/double(n_validations)*100.0;
		return accuracy;
	};
	
	//checks that ALL outputs are correct...
	//valid for continuous outputs.
	double check_accuracy(const vector<vector<double>>& validation_data,const vector<vector<double>>& validation_output){
		
		int n_validations = validation_data.size();
		//vector<double> computed_answers = this->get_output();
		double pc_accuracy = 0.0;
		for(int sample=0;sample<n_validations;sample++){
			
			//assign starting values
			this->set_input(validation_data[sample]);
			
			//forward model
			this->feedforward_validation();
			
			vector<double> computed_answers = this->get_output();
			//vector<double> pc_accuracy(computed_answers.size());
			
			
			//cout << "computed answers size = " << computed_answers.size() << endl;
			for(unsigned int i=0;i<computed_answers.size();i++){
				//cout << "validation_output[sample][i] = " << validation_output[sample][i] << endl;
				//cout << "computed_answers[i] = " << computed_answers[i] << endl;
				
				pc_accuracy += ((validation_output[sample][i]-abs(validation_output[sample][i]-computed_answers[i]))/validation_output[sample][i])*100.0;
				//cout << "	inner pc_accuracy = " << pc_accuracy << endl;
			}
			pc_accuracy = pc_accuracy/computed_answers.size();
			//cout << "	middle pc_accuracy = " << pc_accuracy << endl;
		}
		
		pc_accuracy = pc_accuracy/n_validations;
		cout << "outer pc_accuracy = " << pc_accuracy << endl;
		return pc_accuracy;
	};
	
	
	//checks that ALL outputs are correct...
	double check_accuracy(const TN_AI_Dataset& validation_dataset){
		
		int correct_count = 0;
		int n_validations = validation_dataset.size();
		//vector<double> computed_answers = this->get_output(); //???????????????????
		
		for(int sample=0;sample<n_validations;sample++){
			//assign starting values
			this->set_input(validation_dataset.get_input_datapoint(sample));
			
			//forward model
			this->feedforward_validation();
			//mycluster.visualise();
			
			vector<double> computed_answers = this->get_output();
			vector<double> output(computed_answers.size());
			
			//calculate computed answers
			for(unsigned int i=0;i<computed_answers.size();i++){
				if(computed_answers[i] > 0.5){
					output[i] = 1.0;
				}else{
					output[i] = 0.0;
				}
			}
			
			int n_parts_correct = 0;
			for(unsigned int i=0;i<computed_answers.size();i++){
				
				if(output[i] == validation_dataset.get_output_datapoint(sample)[i]){
					n_parts_correct +=1;
					
				}
			}
			
			
			if(n_parts_correct==int(output.size())){correct_count += 1;} //only if ALL parts are correct, item is correct
		}
	
		double accuracy = double(correct_count)/double(n_validations)*100.0;
		return accuracy;
	};
	
	
	
	
	double compute_loss(){
		double loss = 0.0;
		for (unsigned int i=0; i<y.size(); ++i) {
			loss += pow(y[i] - x[W.size()-1][i], 2);
		}
		return loss/y.size();
	};
	
	void output_essential_data(string data,string filename){
		ofstream output_file;
		output_file.open(filename,ios::out);
		if(output_file.bad()){
			std::cerr << "cannot open output file " << filename << std::endl;
			exit(8);
        }
		output_file << data;
		output_file.close();
	};
	
	
	
	void output_b(string filename){
		
		ofstream output_file;
		output_file.open(filename,ios::out | ios::binary | ios::trunc);
		if(output_file.bad()){
			std::cerr << "cannot open output file " << filename << std::endl;
			exit(8);
        }
		for (unsigned int i=0; i<b.size(); ++i) {
			for (unsigned int j=0; j<b[i].size(); ++j) {
				output_file.write(reinterpret_cast<const char*>(&b[i][j]),sizeof(double));
			}
		}
		output_file.close();
	};
	
	void output_W(string filename){
		
		ofstream output_file;
		output_file.open(filename,ios::out | ios::binary | ios::trunc);
		if(output_file.bad()){
			std::cerr << "cannot open output file " << filename << std::endl;
			exit(8);
        }
		for (unsigned int i=0; i<W.size(); ++i) {
			for (unsigned int j=0; j<W[i].size(); ++j) {
				for (unsigned int k=0; k<W[i][j].size(); ++k) {
					output_file.write(reinterpret_cast<const char*>(&W[i][j][k]),sizeof(double));
					//output_file.write(reinterpret_cast<const char*>(W[i][j].data()), W[i][j].size() * sizeof(double));
				}
			}
		}
		output_file.close();
	};
	
	void input_b(string filename){
		
		ifstream input_file;
		input_file.open(filename,ios::in | ios::binary);
		if(input_file.bad()){
			std::cerr << "cannot open input file " << filename << std::endl;
			exit(8);
        }
		
		for (unsigned int i=0; i<b.size(); ++i) {
			for (unsigned int j=0; j<b[i].size(); ++j) {
				input_file.read(reinterpret_cast<char*>(&b[i][j]),sizeof(double));
			}
		}
		input_file.close();
	};
	
	
	void input_W(string filename){
		
		ifstream input_file;
		input_file.open(filename,ios::in | ios::binary);
		if(input_file.bad()){
			std::cerr << "cannot open input file " << filename << std::endl;
			exit(8);
        }
		
		for (unsigned int i=0; i<W.size(); ++i) {
			for (unsigned int j=0; j<W[i].size(); ++j) {
				for (unsigned int k=0; k<W[i][j].size(); ++k) {
					input_file.read(reinterpret_cast<char*>(&W[i][j][k]),sizeof(double));
					//input_file.read(reinterpret_cast<char*>(W[i][j].data()), W[i][j].size() * sizeof(double));
				}
			}
		}
		input_file.close();
	};
	
	void output_W(){
		
		
		cout << "W: " << endl;
		cout << W[0][0][0] << endl;
		/*
		for (unsigned int i=0; i<W.size(); ++i) {
			for (unsigned int j=0; j<W[i].size(); ++j) {
				for (unsigned int k=0; k<W[i][j].size(); ++k) {
					cout << W[i][j][k] << endl;
				}
			}
		}*/
	};
	
};
#endif //TN_AI_CLUSTER