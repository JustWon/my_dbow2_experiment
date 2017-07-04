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
#include <string.h>
#include <DWConfig.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;


void loadDCGANFeatures(vector<vector<vector<float> > > &features, bool isTraining)
{
	features.clear();
	vector<string> file_lists;

	if (isTraining)
		file_lists = getFileNames(train_desc_dir_path.c_str());
	else
		file_lists = getFileNames(test_desc_dir_path.c_str());

//	FILE *fp = fopen("DCGAN_image_order.txt","wt");
//	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
//		fprintf(fp, "%s\n", (*iter).c_str());
////		cout << *iter << endl;
//	}
//	fclose(fp);

	vector<vector<float> > descriptors;
	for(int i = 0; i < file_lists.size() ; i++)
	{
		vector<vector<float> > descriptors;
		descriptors.clear();

		printf("%s\n", file_lists[i].c_str());
		FILE *fp = fopen(file_lists[i].c_str(), "rt");
		for (int j = 0 ; j< desc_num ; j++) // The # of patches in an image
		{
			vector<float> descriptor;
			descriptor.clear();
			for (int k = 0 ; k < desc_dim ; k++) // dimension of the descriptor
			{
				float temp;
				fscanf(fp, "%f", &temp);
				descriptor.push_back(temp);
			}
			descriptors.push_back(descriptor);
		}
		fclose(fp);

		features.push_back(descriptors);

//		if (i == file_lists.size()-1)
//			break;
	}
}

void createVocabulary(DCGANVocabulary &voc, const vector<vector<vector<float> > > &training_features)
{
	voc.create(training_features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl	<< voc << endl << endl;
}

void validateVocabulary (DCGANVocabulary &voc, const vector<vector<vector<float> > > &validation_features)
{
	int num = validation_features.size();
	cout << "Vocabulary information: " << endl << voc << endl << endl;

	FILE *fp = fopen(DCGAN_corr_matrix_output.c_str(),"wt");
	// lets do something with this vocabulary
	cout << "Matching images against themselves (0 low, 1 high): " << endl;
	BowVector v1, v2;
	for(int i = 0; i < num; i++)
	{
		double max_score = 0.0;
		int most_related_idx = 0;

		voc.transform(validation_features[i], v1);
		for(int j = 0; j < num; j++)
		{
			// for the upper triangular matrix
			if (i <= j) {
				voc.transform(validation_features[j], v2);

				double score = voc.score(v1, v2);
				fprintf(fp, "%lf ", score);

				if (max_score < score && score < 0.99)
				{
					max_score = score;
					most_related_idx = j;
				}
			}
			else
			{
				fprintf(fp, "%lf ", 0);
			}
		}
		fprintf(fp, "\n");
		//fprintf(fp, "current_idx=%d, max_score=%lf, most_related_idx=%d\n", start_idx+i+1, max_score, start_idx+most_related_idx);
		printf("current_idx=%d, max_score=%lf, most_related_idx=%d\n", i+1, max_score, most_related_idx);
	}

	fclose(fp);
}


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

void loadSURFFeatures(vector<vector<vector<float> > > &features, bool isTraining)
{
	features.clear();
	vector<string> file_lists;
	if (isTraining)
		file_lists = getFileNames(train_img_dir_path.c_str());
	else
		file_lists = getFileNames(test_img_dir_path.c_str());

//	FILE *fp = fopen("SURF_image_order.txt","wt");
//	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
//		fprintf(fp, "%s\n", (*iter).c_str());
////		cout << *iter << endl;
//	}
//	fclose(fp);

	cv::Ptr<cv::xfeatures2d::SURF> surf =
			cv::xfeatures2d::SURF::create(surf_param[0], surf_param[1], surf_param[2], EXTENDED_SURF);

	cout << "Extracting SURF features..." << endl;
	for(int i = 0; i <= train_set_num; ++i)
	{
		printf("%s\n", file_lists[i].c_str());
		cv::Mat image = cv::imread(file_lists[i].c_str(), 0);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		vector<float> descriptors;

		surf->detectAndCompute(image, mask, keypoints, descriptors);

		features.push_back(vector<vector<float> >());
		changeStructure(descriptors, features.back(), surf->descriptorSize());

		printf("# of desc : %d\n", descriptors.size());
		if (i == file_lists.size()-1)
			break;
	}
}

// ----------------------------------------------------------------------------

void createVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &training_features)
{
	voc.create(training_features);
	cout << "... done!" << endl;
	cout << "Vocabulary information: " << endl << voc << endl << endl;
}

void validateVocabulary(Surf64Vocabulary &voc, const vector<vector<vector<float> > > &validation_features)
{
	int num = validation_features.size();
	cout << "Vocabulary information: " << endl << voc << endl << endl;

	FILE *fp = fopen(SURF_corr_matrix_output.c_str(),"wt");
	// lets do something with this vocabulary
	cout << "Matching images against themselves (0 low, 1 high): " << endl;
	BowVector v1, v2;
	for(int i = 0 ; i < num; i++)
	{
		double max_score = 0.0;
		int most_related_idx = 0;

		voc.transform(validation_features[i], v1);
		for(int j = 0; j < num; j++)
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
		printf("current_idx=%d, max_score=%lf, most_related_idx=%d\n", i+1, max_score, most_related_idx);
	}
	fclose(fp);
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
		loadDCGANFeatures(training_features, true);
		createVocabulary(voc, training_features);

		vector<vector<vector<float> > > validation_features;
		loadDCGANFeatures(validation_features, false);
		validateVocabulary(voc, validation_features);
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
		loadSURFFeatures(training_features, true);
		createVocabulary(voc, training_features);

		vector<vector<vector<float> > > validation_features;
		loadSURFFeatures(validation_features, false);
		validateVocabulary(voc, validation_features);
	}
	return 0;
}
