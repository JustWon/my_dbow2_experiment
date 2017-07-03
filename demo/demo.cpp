/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <iomanip>

// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadSURFFeatures(vector<vector<vector<float> > > &features, int from, int to, bool isTraining);
void loadDCGANFeatures(vector<vector<vector<float> > > &features, int from, int to, bool isTraining);
void loadBRIEFFeatures(vector<vector<vector<float> > > &features, int from, int to, bool isTraining);

void changeStructure(const vector<float> &plain, vector<vector<float> > &out, int L);
void testDatabase(const vector<vector<vector<float> > > &features);

void createVocabulary(DCGANVocabulary &voc, const vector<vector<vector<float> > > &training_features);
void validateVocabulary(DCGANVocabulary &voc, const vector<vector<vector<float> > > &validation_features, int num);
void createVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &training_features);
void validateVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &validation_features, int num);
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
const int start_idx = 0;
const int NIMAGES = 100;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = true;

bool Proposed_Method_Test = true;
bool SURF_Test = true;

const int num_training = 100;
const int num_test = 2474;
const int stride = 1;

vector<string> getFileNames (string dir)
{
	vector<string> file_lists;

	DIR *dp;
	struct dirent *ep;
	dp = opendir (dir.c_str());

	if (dp != NULL)
	{
		while (ep = readdir (dp)){
			if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, ".."))
				file_lists.push_back(dir + "/"+ ep->d_name);
		}

		(void) closedir (dp);
	}
	else
		perror ("Couldn't open the directory");

	return file_lists;
}

int main()
{
	// Proposed Method
	if (Proposed_Method_Test){
		// branching factor and depth levels
		cout << "Extracting DCGAN features..." << endl;
		const int k = 10;
		const int L = 5;
		const WeightingType weight = TF_IDF;
		const ScoringType score = DOT_PRODUCT;
		cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;

		DCGANVocabulary voc(k, L, weight, score);
		vector<vector<vector<float> > > training_features;
		loadDCGANFeatures(training_features, 1, num_training, true);
		createVocabulary(voc, training_features);

		vector<vector<vector<float> > > validation_features;
		loadDCGANFeatures(validation_features, 1, num_test, false);
		validateVocabulary(voc, validation_features, num_test);
	}

	// SURF features
	if (SURF_Test){
	    // branching factor and depth levels
		cout << "Extracting SURF features..." << endl;
		const int k = 10;
		const int L = 5;
		const WeightingType weight = TF_IDF;
		const ScoringType score = L2_NORM;
		cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;

		Surf64Vocabulary voc(k, L, weight, score);
		vector<vector<vector<float> > > training_features;
		loadSURFFeatures(training_features, 1, num_training, true);
		createVocabulary(voc, training_features);

		vector<vector<vector<float> > > validation_features;
		loadSURFFeatures(validation_features, 1, num_test, false);
		validateVocabulary(voc, validation_features, num_test);
	}
	return 0;
}

void loadDCGANFeatures(vector<vector<vector<float> > > &features, int from, int to, bool isTraining)
{
	features.clear();
	features.reserve(NIMAGES);

	string test_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre ManualLC/descs";
	vector<string> file_lists = getFileNames(test_dir_path.c_str());
	sort(file_lists.begin(),file_lists.end());
	FILE *fp = fopen("DCGAN_image_order.txt","wt");
	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
		fprintf(fp, "%s\n", (*iter).c_str());
//		cout << *iter << endl;
	}
	fclose(fp);

	vector<vector<float> > descriptors;
	for(int i = from; i <= to ; i++)
	{
		vector<vector<float> > descriptors;
		descriptors.clear();

		char filename[1024];
		if (isTraining)
			sprintf(filename, "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/descs/20170702/Places365_val_%08d.desc", i);
		//		sprintf(filename, "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre/images/%04d.jpg", i);
		else
			sprintf(filename, file_lists[i].c_str());

		printf("%s\n", filename);

		FILE *fp = fopen(filename, "rt");
		for (int j = 0 ; j< 300 ; j++) // The # of patches in an image
		{
			vector<float> descriptor;
			descriptor.clear();
			for (int k = 0 ; k < 128 ; k++) // dimension of the descriptor
			{
				float temp;
				fscanf(fp, "%f", &temp);
				descriptor.push_back(temp);
			}
			descriptors.push_back(descriptor);
		}
		fclose(fp);

		features.push_back(descriptors);

		if (i == file_lists.size()-1)
			break;
	}
}

void createVocabulary(DCGANVocabulary &voc, const vector<vector<vector<float> > > &training_features)
{
	voc.create(training_features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
			<< voc << endl << endl;
}

void validateVocabulary (DCGANVocabulary &voc, const vector<vector<vector<float> > > &validation_features, int num)
{
	num = validation_features.size()-1;
	cout << "Vocabulary information: " << endl << voc << endl << endl;

	FILE *fp = fopen("DCGAN_corr_matrix.txt","wt");
	// lets do something with this vocabulary
	cout << "Matching images against themselves (0 low, 1 high): " << endl;
	BowVector v1, v2;
	for(int i = 0; i < num; i+=stride)
	{
		double max_score = 0.0;
		int most_related_idx = 0;

		voc.transform(validation_features[i], v1);
		for(int j = 0; j < num; j+=stride)
		{
			// for the upper triangular matrix
			if (i <= j) {
				voc.transform(validation_features[j], v2);

				double score = voc.score(v1, v2);
				fprintf(fp, "%lf ", score);

				if (max_score < score && score < 0.99)
				{
					max_score = score;
					most_related_idx = j+1;
				}
			}
			else
			{
				fprintf(fp, "%lf ", 0);
			}
		}
		fprintf(fp, "\n");
		//fprintf(fp, "current_idx=%d, max_score=%lf, most_related_idx=%d\n", start_idx+i+1, max_score, start_idx+most_related_idx);
		printf("current_idx=%d, max_score=%lf, most_related_idx=%d\n", start_idx+i+1, max_score, start_idx+most_related_idx);
	}

	fclose(fp);
}


void loadSURFFeatures(vector<vector<vector<float> > > &features, int from, int to, bool isTraining)
{
	features.clear();
	features.reserve(to - from+1);

	string test_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre ManualLC/images";
	vector<string> file_lists = getFileNames(test_dir_path.c_str());
	sort(file_lists.begin(),file_lists.end());

	FILE *fp = fopen("SURF_image_order.txt","wt");
	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
		fprintf(fp, "%s\n", (*iter).c_str());
//		cout << *iter << endl;
	}
	fclose(fp);

	cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(300, 4, 2, EXTENDED_SURF);

	cout << "Extracting SURF features..." << endl;
	for(int i = from; i <= to; ++i)
	{
		char filename[1024];
		if (isTraining)
			sprintf(filename, "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/images/Places365_val_%08d.jpg", i);
		//		sprintf(filename, "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre/images/%04d.jpg", i);
		else
//			sprintf(filename, "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre/images/%04d.jpg", i);
			sprintf(filename, file_lists[i].c_str());


		printf("%s\n", filename);
		cv::Mat image = cv::imread(filename, 0);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		vector<float> descriptors;

		surf->detectAndCompute(image, mask, keypoints, descriptors);

		features.push_back(vector<vector<float> >());
		changeStructure(descriptors, features.back(), surf->descriptorSize());

		if (i == file_lists.size()-1)
			break;
	}
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out, int L)
{
	out.resize(plain.size() / L);

	unsigned int j = 0;
	for(unsigned int i = 0; i < plain.size(); i += L, ++j)
	{
		out[j].resize(L);
		std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
	}
}

void createVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &training_features)
{
	voc.create(training_features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
			<< voc << endl << endl;
}

void validateVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &validation_features, int num)
{
	num = validation_features.size()-1;
	cout << "Vocabulary information: " << endl << voc << endl << endl;

	FILE *fp = fopen("SURF_corr_matrix.txt","wt");
	// lets do something with this vocabulary
	cout << "Matching images against themselves (0 low, 1 high): " << endl;
	BowVector v1, v2;
	for(int i = 0 ; i < num; i+=stride)
	{
		double max_score = 0.0;
		int most_related_idx = 0;

		voc.transform(validation_features[i], v1);
		for(int j = 0; j < num; j+=stride)
		{
			if (i <= j) {
				voc.transform(validation_features[j], v2);

				double score = voc.score(v1, v2);
				fprintf(fp, "%lf ", score);

				if (max_score < score && score < 0.99)
				{
					max_score = score;
					most_related_idx = j+1;
				}
			}
			else
			{
				fprintf(fp, "%lf ", 0);
			}
		}
		// fprintf(fp, "current_idx=%d, max_score=%lf, most_related_idx=%d\n", start_idx+i+1, max_score, start_idx+most_related_idx);
		fprintf(fp, "\n");
		printf("current_idx=%d, max_score=%lf, most_related_idx=%d\n", start_idx+i+1, max_score, start_idx+most_related_idx);
	}
	fclose(fp);
}
