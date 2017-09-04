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
const bool EXTENDED_SURF = true;


DW_Config dw_config;

void loadDCGANFeatures(vector<vector<vector<float> > > &features, bool isTraining)
{
	features.clear();
	vector<string> file_lists;

	if (isTraining)
		file_lists = dw_config.getFileNames(dw_config.getTrainDescDirPath().c_str());
	else
		file_lists = dw_config.getFileNames(dw_config.getTestDescDirPath().c_str());

	FILE *fp = fopen("DCGAN_image_order.txt","wt");
	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
		fprintf(fp, "%s\n", (*iter).c_str());
//		cout << *iter << endl;
	}
	fclose(fp);

	int desc_num = dw_config.getDescNum(), desc_dim = dw_config.getDescDim();
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

	FILE *fp = fopen(dw_config.getCorrMatrixOutput().c_str(),"wt");
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
	int set_num = 0;
	if (isTraining)
	{
		file_lists = dw_config.getFileNames(dw_config.getTrainImgDirPath().c_str());
		set_num = dw_config.getTrainSetNum();
	}
	else
	{
		file_lists = dw_config.getFileNames(dw_config.getTestImgDirPath().c_str());
		set_num = dw_config.getTestSetNum();
	}

//	FILE *fp = fopen("SURF_image_order.txt","wt");
//	for (vector<string>::iterator iter = file_lists.begin(); iter != file_lists.end(); ++iter){
//		fprintf(fp, "%s\n", (*iter).c_str());
////		cout << *iter << endl;
//	}
//	fclose(fp);

	float* surf_param = dw_config.getSurfParam();
	cv::Ptr<cv::xfeatures2d::SURF> surf =
			cv::xfeatures2d::SURF::create(surf_param[0], surf_param[1], surf_param[2], EXTENDED_SURF);

	cout << "Extracting SURF features..." << endl;
	for(int i = 0; i <= set_num; ++i)
	{
		printf("%s\n", file_lists[i].c_str());
		cv::Mat image = cv::imread(file_lists[i].c_str(), 0);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		vector<float> descriptors;

		surf->detectAndCompute(image, mask, keypoints, descriptors);
//		vector<float>::const_iterator begin = descriptors.begin();
//		vector<float>::const_iterator last = descriptors.begin()+300;
//		vector<float> limited_desc(begin, last);
//		printf("# of desc : %f\n", float(descriptors.size()/64));

		features.push_back(vector<vector<float> >());
		changeStructure(descriptors, features.back(), surf->descriptorSize());

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

	FILE *fp = fopen(dw_config.getCorrMatrixOutput().c_str(),"wt");
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
	const int k = dw_config.getClusterCenterNum();
	const int L = dw_config.getDepthLevelNum();
	const WeightingType weight = dw_config.getWeightingType();
	const ScoringType score = dw_config.getScoringType();

	vector<vector<vector<float> > > training_features;
	vector<vector<vector<float> > > validation_features;

	if (dw_config.getEvalMethod() == "proposed method"){
		dw_config.resultLogOrganization();
		DCGANVocabulary voc(k, L, weight, score);
		loadDCGANFeatures(training_features, true);
		loadDCGANFeatures(validation_features, false);

		createVocabulary(voc, training_features);
		validateVocabulary(voc, validation_features);
	}

	if (dw_config.getEvalMethod() == "SURF"){
		dw_config.resultLogOrganization();
		Surf64Vocabulary voc(k, L, weight, score);
		loadSURFFeatures(training_features, true);
		loadSURFFeatures(validation_features, false);

		createVocabulary(voc, training_features);
		validateVocabulary(voc, validation_features);
	}

	return 0;
}
