#include <opencv2/opencv.hpp>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkGeodesicActiveContourLevelSetImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include "itkImportImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSigmoidImageFilter.h"

#include "itk/itkOpenCVImageBridge.h"

#include "edges_pose_refiner/geodesicActiveContour.hpp"
#include "edges_pose_refiner/utils.hpp"

#define DEBUG_GAC

using namespace cv;
using std::cout;
using std::endl;

//TODO: remove global declarations
typedef float InternalPixelType;
const unsigned int Dimension = 2;
typedef itk::Image<InternalPixelType, Dimension> InternalImageType;
typedef unsigned char OutputPixelType;
typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
typedef itk::OpenCVImageBridge BridgeType;
typedef itk::GeodesicActiveContourLevelSetImageFilter<InternalImageType, InternalImageType> GeodesicActiveContourFilterType;

void cv2itk(const cv::Mat &mat, itk::Image<InternalPixelType, Dimension>::Pointer &itkImage)
{
  typedef itk::ImportImageFilter<InternalPixelType, Dimension> ImportFilterType;
  ImportFilterType::Pointer importFilter = ImportFilterType::New();

  ImportFilterType::SizeType size;
  size[0] = mat.cols;
  size[1] = mat.rows;

  ImportFilterType::IndexType start;
  start.Fill(0);

  ImportFilterType::RegionType region;
  region.SetSize(size);
  region.SetIndex(start);

  importFilter->SetRegion(region);

  double origin[Dimension];
  origin[0] = 0.0;
  origin[1] = 0.0;
  importFilter->SetOrigin(origin);

  double spacing[Dimension];
  spacing[0] = 1.0;
  spacing[1] = 1.0;
  importFilter->SetSpacing(spacing);

  const unsigned int numberOfPixels = mat.total();
  InternalPixelType *inputPixels = new InternalPixelType[numberOfPixels];
  CV_Assert(mat.isContinuous());
  CV_Assert(mat.type() == CV_32FC1);
  //TODO: get rid of the cast
  float *matData = reinterpret_cast<float*> (mat.data);
  std::copy(matData, matData + numberOfPixels, inputPixels);

  //TODO: is it a memory leak?
  const bool importImageFilterWillOwnTheBuffer = false;
  importFilter->SetImportPointer (inputPixels, numberOfPixels, importImageFilterWillOwnTheBuffer);
  importFilter->Update();
  itkImage = importFilter->GetOutput();
}

class GACIterationUpdate : public itk::Command
{
  public:
    typedef GACIterationUpdate Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    itkNewMacro(Self);
  protected:
    GACIterationUpdate();
    cv::Mat bgrImage;
    int iterationStep;
  public:
    void Execute(itk::Object *caller, const itk::EventObject &event);
    void Execute(const itk::Object *object, const itk::EventObject &event);

    void setBgrImage(const cv::Mat &bgrImage);
};

GACIterationUpdate::GACIterationUpdate()
{
  //TODO: move up
  iterationStep = 50;
}

void GACIterationUpdate::setBgrImage(const cv::Mat &_bgrImage)
{
  bgrImage = _bgrImage.clone();
}

void GACIterationUpdate::Execute(itk::Object *caller, const itk::EventObject &event)
{
  Execute(static_cast<const itk::Object*>(caller), event);
}

void GACIterationUpdate::Execute(const itk::Object *object, const itk::EventObject &event)
{
  if (typeid(event) != typeid(itk::IterationEvent))
  {
    return;
  }

  const GeodesicActiveContourFilterType *geodesicActiveContour = dynamic_cast<const GeodesicActiveContourFilterType*>(object);
  if (geodesicActiveContour->GetElapsedIterations() == 1 || geodesicActiveContour->GetElapsedIterations() % iterationStep == 0)
  {
    std::cout << geodesicActiveContour->GetElapsedIterations()<< "   " << std::flush;
    typedef itk::BinaryThresholdImageFilter<InternalImageType, OutputImageType> ThresholdingFilterType;
    ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
    thresholder->SetLowerThreshold(-std::numeric_limits<float>::max());
    thresholder->SetUpperThreshold(0.0);
    thresholder->SetOutsideValue(0);
    thresholder->SetInsideValue(255);
    thresholder->SetInput(geodesicActiveContour->GetOutput());
    thresholder->Update();

    cv::Mat currentSegmentation = BridgeType::ITKImageToCVMat<OutputImageType>(thresholder->GetOutput());
    Mat currentCurve = drawSegmentation(bgrImage, currentSegmentation, Scalar(255, 0, 0), 2);
    imshow("current curve", currentCurve);
/*
    std::stringstream filename;
    static int i = 10;
    filename << i << ".png";
    imwrite(filename.str(), currentCurve);
    i += 10;
*/
    waitKey(5);
  }
}

void initializeGeodesicActiveContour(const cv::Mat &edges, cv::Mat &initialLevelSet, cv::Mat &featureImage)
{
  //TODO: move up
  const int initialContourMargin = 10;

  //TODO: use FastMarching
  Mat initialContour(edges.size(), CV_8UC1, Scalar(0));
  Point tl(initialContourMargin, initialContourMargin);
  Point br(edges.cols - 1 - initialContourMargin, edges.rows - 1 - initialContourMargin);
  Rect contourRect(tl, br);
  rectangle(initialContour, contourRect, Scalar(255));
  Mat distanceMap;

  distanceTransform(~initialContour, distanceMap, CV_DIST_L2, CV_DIST_MASK_PRECISE);
  Mat contourInterior = distanceMap(contourRect);
  contourInterior *= -1;
  //TODO: distanceMap *= -1 ?
  initialLevelSet = distanceMap;

  Mat edgesFloat;
  edges.convertTo(edgesFloat, CV_32FC1);
  featureImage = edgesFloat;

  imshow("featureImage", featureImage);
//  imshow("featureImage divided ", featureImage / 100.0);
  waitKey();
}

void geodesicActiveContour(const cv::Mat &bgrImage, const cv::Mat &edges, cv::Mat &segmentation)
{
  //TODO: move up
//  const float alpha = -1.0f;
 const float alpha = -0.2f;
//  const float beta = 3.0f;
//  const float beta = 0.01f;
  const float beta = 0.38f;

  const float propagationScaling = -0.3f;
  const float curvatureScaling = 1.0f;
  const float advectionScaling = 1.0f;

  const float maximumRMSError = 0.001f;
  const int numberOfIterations = 6000;

/*
  const float seedX = 320.0f;
  const float seedY = 240.0f;
  const float initialDistance = 230.0f;
  const float speedConstant = 1.0f;
*/

  Mat initialLevelSetMat, featureImageMat;
  initializeGeodesicActiveContour(edges, initialLevelSetMat, featureImageMat);

  InternalImageType::Pointer initialLevelSet;
  cv2itk(initialLevelSetMat, initialLevelSet);
  //TODO: use ITK OpenCV Bridge
  //initialLevelSet = BridgeType::CVMatToITKImage<InternalImageType>(initialLevelSetMat);

  InternalImageType::Pointer featureImage;
  cv2itk(featureImageMat, featureImage);
  //TODO: use ITK OpenCV Bridge
  //featureImage = BridgeType::CVMatToITKImage<InternalImageType>(featureImageMat);

  CV_Assert(initialLevelSet->GetRequestedRegion().GetSize()[0] == edges.cols);
  CV_Assert(initialLevelSet->GetRequestedRegion().GetSize()[1] == edges.rows);

  CV_Assert(featureImage->GetRequestedRegion().GetSize()[0] == edges.cols);
  CV_Assert(featureImage->GetRequestedRegion().GetSize()[1] == edges.rows);

/*
  typedef itk::FastMarchingImageFilter<InternalImageType, InternalImageType> FastMarchingFilterType;
  FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();

  typedef FastMarchingFilterType::NodeContainer NodeContainer;
  typedef FastMarchingFilterType::NodeType NodeType;
  NodeContainer::Pointer seeds = NodeContainer::New();

  InternalImageType::IndexType seedPosition;
  seedPosition[0] = seedX;
  seedPosition[1] = seedY;

  NodeType node;
  const double seedValue = -initialDistance;

  node.SetValue( seedValue );
  node.SetIndex( seedPosition );

  seeds->Initialize();
  seeds->InsertElement(0, node);
  fastMarching->SetTrialPoints(seeds);
  fastMarching->SetSpeedConstant(speedConstant);

  fastMarching->SetOutputSize(featureImage->GetRequestedRegion().GetSize());
*/

  typedef itk::SigmoidImageFilter<InternalImageType, InternalImageType> SigmoidFilterType;
  SigmoidFilterType::Pointer sigmoid = SigmoidFilterType::New();
  sigmoid->SetOutputMinimum(0.0);
  sigmoid->SetOutputMaximum(1.0);

  sigmoid->SetInput(featureImage);
  sigmoid->SetAlpha(alpha);
  sigmoid->SetBeta(beta);


  GeodesicActiveContourFilterType::Pointer geodesicActiveContour = GeodesicActiveContourFilterType::New();

  geodesicActiveContour->SetPropagationScaling(propagationScaling);
  geodesicActiveContour->SetCurvatureScaling(curvatureScaling);
  geodesicActiveContour->SetAdvectionScaling(advectionScaling);

  geodesicActiveContour->SetMaximumRMSError(maximumRMSError);
  geodesicActiveContour->SetNumberOfIterations(numberOfIterations);

//  geodesicActiveContour->SetInput(fastMarching->GetOutput());
  geodesicActiveContour->SetInput(initialLevelSet);
//  geodesicActiveContour->SetFeatureImage(featureImage);
  geodesicActiveContour->SetFeatureImage(sigmoid->GetOutput());


  GACIterationUpdate::Pointer observer = GACIterationUpdate::New();
  observer->setBgrImage(bgrImage);
  geodesicActiveContour->AddObserver(itk::IterationEvent(), observer);


  typedef itk::BinaryThresholdImageFilter<InternalImageType, OutputImageType> ThresholdingFilterType;
  ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();

  thresholder->SetLowerThreshold(-std::numeric_limits<float>::max());
  thresholder->SetUpperThreshold(0.0);
  thresholder->SetOutsideValue(0);
  thresholder->SetInsideValue(255);

  thresholder->SetInput( geodesicActiveContour->GetOutput() );

  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(thresholder->GetOutput());
  std::string tmpOutputFilename = "gac.png";
  writer->SetFileName(tmpOutputFilename);
  writer->Update();

#ifdef DEBUG_GAC
  typedef itk::RescaleIntensityImageFilter<InternalImageType, OutputImageType> CastFilterType;
  WriterType::Pointer fmWriter = WriterType::New();
  CastFilterType::Pointer fmCaster = CastFilterType::New();
/*
  fmCaster->SetInput(fastMarching->GetOutput());
  fmWriter->SetInput(fmCaster->GetOutput());
  fmWriter->SetFileName("FastMarchingFilterOutput.png");
  fmCaster->SetOutputMinimum(0);
  fmCaster->SetOutputMaximum(255);
  fmWriter->Update();
*/

  {
    WriterType::Pointer fmWriter = WriterType::New();
    CastFilterType::Pointer fmCaster = CastFilterType::New();
    fmCaster->SetInput(sigmoid->GetOutput());
    fmWriter->SetInput(fmCaster->GetOutput());
    fmWriter->SetFileName("sigmoid.png");
    fmCaster->SetOutputMinimum(0);
    fmCaster->SetOutputMaximum(255);
    fmWriter->Update();
  }


  {
    WriterType::Pointer fmWriter = WriterType::New();
    CastFilterType::Pointer fmCaster = CastFilterType::New();
    fmCaster->SetInput(featureImage);
    fmWriter->SetInput(fmCaster->GetOutput());
    fmWriter->SetFileName("featureImage.png");
    fmCaster->SetOutputMinimum(0);
    fmCaster->SetOutputMaximum(255);
    fmWriter->Update();
  }

  std::cout << std::endl;
  std::cout << "Max. no. iterations: " << geodesicActiveContour->GetNumberOfIterations() << std::endl;
  std::cout << "Max. RMS error: " << geodesicActiveContour->GetMaximumRMSError() << std::endl;
  std::cout << std::endl;
  std::cout << "No. elpased iterations: " << geodesicActiveContour->GetElapsedIterations() << std::endl;
  std::cout << "RMS change: " << geodesicActiveContour->GetRMSChange() << std::endl;
#endif

  //TODO: don't use files
  sleep(2);
  segmentation = imread(tmpOutputFilename, CV_LOAD_IMAGE_GRAYSCALE);
}
