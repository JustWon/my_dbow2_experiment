#include <string.h>

using namespace std;

// configurable parameters
const string train_desc_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/descs/20170702";
const string test_desc_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre ManualLC/descs";
const string train_img_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/Places365/Large_images/val_large/images";
const string test_img_dir_path = "/media/dongwonshin/Ubuntu Data/Datasets/FAB-MAP/Image Data/City Centre ManualLC/images";

const string DCGAN_corr_matrix_output = "DCGAN_corr_matrix.txt";
const string SURF_corr_matrix_output = "SURF_corr_matrix.txt";

const int desc_num = 300;
const int desc_dim = 128;

const int train_set_num = 100;
const int test_set_num = 1000;

const bool Proposed_Method_Test = true;
const bool SURF_Test = true;


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
			if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, ".."))
				file_lists.push_back(dir + "/"+ ep->d_name);
		}

		(void) closedir (dp);
	}
	else
		perror ("Couldn't open the directory");

	sort(file_lists.begin(),file_lists.end());

	return file_lists;
}
