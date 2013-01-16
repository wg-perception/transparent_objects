//The code is imported (with modifications) from the ROS TOD package

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/Image.h>

#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include "boost/format.hpp"

#include <iostream>
#include <string>
#include <fstream>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define foreach         BOOST_FOREACH
#define reverse_foreach BOOST_REVERSE_FOREACH

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using std::cout;
using std::endl;
using std::cerr;

using namespace cv;

namespace
{
struct Options
{
  std::string bag_file;
  std::string object_name;
  std::string base_path;

  std::string topic_image, topic_depth_image;
  fs::path full_path;
};

struct ImagePointsCamera
{
  sensor_msgs::ImageConstPtr img;
  sensor_msgs::ImageConstPtr depth_img;
  bool full() const
  {
    return (img != NULL && depth_img != NULL);
  }
  void clear()
  {
    img.reset();
    depth_img.reset();
  }
};

int options(int ac, char ** av, Options& opts)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message.");
  desc.add_options()("bag,B", po::value<std::string>(&opts.bag_file),
                     "The bag file with Image messages. Required.");
  desc.add_options()("object_name,N", po::value<std::string>(&opts.object_name),
                     "The unique name of the object, must be a valid directory name, please no spaces. Required.");
  desc.add_options()("base_path,P", po::value<std::string>(&opts.base_path),
                     "The absolute path of the training base. The object will be stored here <base_path>/<object_name> Required.");

  desc.add_options()("image", po::value<std::string>(&opts.topic_image)->default_value("/camera/rgb/image_color"),
                     "image topic");
  desc.add_options()("depth_image", po::value<std::string>(&opts.topic_depth_image)->default_value("/camera/depth_registered/image"),
                     "depth_image topic");


  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (!vm.count("bag"))
  {
    cout << "Must supply a bag file name" << "\n";
    cout << desc << endl;
    return 1;
  }

  if (!vm.count("object_name"))
  {
    cout << "Must supply an object_name" << "\n";
    cout << desc << endl;
    return 1;
  }
  if (!vm.count("base_path"))
  {
    cout << "Must supply an base_path" << "\n";
    cout << desc << endl;
    return 1;
  }
  opts.full_path = opts.full_path / opts.base_path / opts.object_name;
  cout << "created path: " << opts.full_path.string() << endl;
  fs::create_directories(opts.full_path);
  if (!fs::exists(opts.full_path))
  {
    cerr << "could not create : " << opts.full_path << endl;
    return 1;
  }

  return 0;
}

class TodProcessor
{
public:
  TodProcessor(Options opts) :
    n(0), opts(opts)
  {
  }
  void process(const ImagePointsCamera& ipc)
  {
    {
      //write image to file
      const cv::Mat cv_image = cv_bridge::toCvShare(ipc.img, "bgr8")->image;
      cv::imwrite((opts.full_path / str(boost::format("image_%05d.png") % n)).string(), cv_image);
    }
    {
      //write image to file
      const cv::Mat cv_image = cv_bridge::toCvShare(ipc.depth_img)->image;
      std::string path = (opts.full_path / str(boost::format("depth_image_%05d.xml.gz") % n)).string();
      cv::FileStorage fs(path, cv::FileStorage::WRITE);
      CV_Assert(fs.isOpened());
      fs << "depth_image" << cv_image;
      fs.release();
    }

    n++;

  }
  int n;
  Options opts;
};

}

int main(int argc, char ** argv)
{
  Options opts;
  if (options(argc, argv, opts))
    return 1;

  rosbag::Bag bag;
  bag.open(opts.bag_file, rosbag::bagmode::Read);

  std::vector<std::string> topics;
  topics.push_back(opts.topic_image);
  topics.push_back(opts.topic_depth_image);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  ImagePointsCamera ipc_package;

  TodProcessor p(opts);
  cout << "dumping the bag file...  " << std::flush;
  foreach(rosbag::MessageInstance const m, view)
  {
    if(m.getTopic() == opts.topic_image)
    {
        sensor_msgs::ImageConstPtr img = m.instantiate<sensor_msgs::Image> ();
        if (img != NULL)
        {
          ipc_package.img = img;
        }
    }

    if(m.getTopic() == opts.topic_depth_image)
    {
        sensor_msgs::ImageConstPtr depth_img = m.instantiate<sensor_msgs::Image> ();
        if (depth_img != NULL)
        {
          ipc_package.depth_img = depth_img;
        }
    }

    if (ipc_package.full())
    {
      p.process(ipc_package);
      ipc_package.clear();
    }
  }
  bag.close();
  cout << "done." << endl;

  std::string testImagesFilename = (opts.full_path/"testImages.txt").string();
  std::ofstream fout(testImagesFilename.c_str());
  CV_Assert(fout.is_open());
  for (int i = 0; i < p.n; ++i)
  {
      fout << i << '\n';
  }
  fout.close();
}
