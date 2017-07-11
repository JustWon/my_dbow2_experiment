#include <string.h>
#include <dirent.h>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include "DBoW2.h"


using namespace std;
using namespace DBoW2;

const string dataset_name = "New College";
const string network_model = "scale normalization";

// configurable parameters
const string train_desc_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/descs/" + network_model;
const string test_desc_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/" + dataset_name + " ManualLC/descs/" + network_model;

const string train_img_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/images";
const string test_img_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/"+ dataset_name +" ManualLC/images";

const string DCGAN_corr_matrix_output = "DCGAN_corr_matrix.txt";
const string SURF_corr_matrix_output = "SURF_corr_matrix.txt";
string corr_matrix_output = "";

const int desc_num = 300;
const int desc_dim = 512;

const int train_set_num = 100;
const int test_set_num = 1000;

const bool Proposed_Method_Test = true;
const bool SURF_Test = false;

const float surf_param[3] = {300,4,2};

const ScoringType g_score = L1_NORM; // L1_NORM, L2_NORM, CHI_SQUARE, KL, BHATTACHARYYA, DOT_PRODUCT


void strCurrentTime(ostringstream& ss)
{
	time_t t = time(0);   // get time now
	struct tm * now = localtime( & t );
	ss << (now->tm_year + 1900) << '-'
		 << setfill('0') << setw(2) << (now->tm_mon + 1) << '-'
		 << setfill('0') << setw(2) << now->tm_mday << '-'
		 << setfill('0') << setw(2) << now->tm_hour << '-'
		 << setfill('0') << setw(2) << now->tm_min << '-'
		 << setfill('0') << setw(2) << now->tm_sec;
}

struct stat st = {0};
void makeLogDir(ostringstream& ss)
{
	string log_dir = "result/" + ss.str();

	if (stat(log_dir.c_str(), &st) == -1) {
 	   mkdir(log_dir.c_str(), 0700);
	}
}

string scoringTypeToString(ScoringType score)
{
	if (score == L1_NORM)
		return string("L1_NORM");
	else if (score == L2_NORM)
		return string("L2_NORM");
	else if (score == CHI_SQUARE)
		return string("CHI_SQUARE");
	else if (score == KL)
		return string("KL");
	else if (score == BHATTACHARYYA)
		return string("BHATTACHARYYA");
	else if (score == DOT_PRODUCT)
		return string("DOT_PRODUCT");
}

void resultLogOrganization()
{
	ostringstream cur_time_str;
	strCurrentTime(cur_time_str);
	makeLogDir(cur_time_str);

	corr_matrix_output = "result/" + cur_time_str.str() + "/corr_matrix.txt";
	cout << corr_matrix_output.c_str() << endl;

	ofstream ofs(("result/"+cur_time_str.str()+"/parameters.cfg").c_str());
	if (Proposed_Method_Test)
	{
		ofs << "[General]" << endl;
		ofs << "Method = proposed_method" << endl;
		ofs << "Dataset = " << dataset_name << endl;
		ofs << "Scoring type = " << scoringTypeToString(g_score) << endl;
		ofs << "Network model = " << network_model << endl;
	}
	else if(SURF_Test)
	{
		ofs << "[General]" << endl;
		ofs << "Method = SURF" << endl;
		ofs << "Dataset = " << dataset_name << endl;
		ofs << "Scoring type = " << scoringTypeToString(g_score)  << endl;
		ofs << "SURF params = {" << surf_param[0] <<',' << surf_param[1]<<',' << surf_param[2] << '}'<<endl;
	}
}

// functions
vector<string> getFileNames (string dir)
{
	vector<string> file_lists;

	DIR *dp;
	struct dirent *ep;
	dp = opendir (dir.c_str());

	if (dp != NULL)
	{
		while (ep = readdir (dp)){
			if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, "..") && strcmp(ep->d_name, "temp"))
				file_lists.push_back(dir + "/"+ ep->d_name);
		}

		(void) closedir (dp);
	}
	else
		perror ("Couldn't open the directory");

	sort(file_lists.begin(),file_lists.end());

	return file_lists;
}
