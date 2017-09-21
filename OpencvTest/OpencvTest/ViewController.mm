//
//  ViewController.m
//  OpencvTest
//
//  Created by Odie Edo-Osagie on 12/09/2017.
//  Copyright © 2017 Odie Edo-Osagie. All rights reserved.
//

#import "ViewController.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgcodecs/ios.h>
#include <opencv2/objdetect/objdetect.hpp>



@interface ViewController (){
    BOOL isCameraOn;
    cv::Mat lastFrame;
}

@property (nonatomic, strong) CvVideoCamera *videoCamera;

@end

@implementation ViewController

cv::CascadeClassifier face_cascade;

- (void)viewDidLoad {
    [super viewDidLoad];
    
    // Do any additional setup after loading the view, typically from a nib.
    self.view.backgroundColor = [UIColor whiteColor];
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.imageView];
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPresetLow;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    self.videoCamera.delegate = self;
    
    // Initialize face classifier
    char *cascadeName = (char*)"haarcascade_frontalface_alt";
    char *cascadeType = (char*)"xml";
    if (face_cascade.load([self getBundlePathForResourceWithName:cascadeName andType:cascadeType])){
        printf("Load complete");
    }else{
        printf("Load error");
    }
    
    //[self skinColor];
    cv::Mat img;
    UIImageToMat([UIImage imageNamed:@"face3.jpg"], img, false);
    self.imageView.image = MatToUIImage([self skinSegmentation:img]);
    
    /* Example Pipeline */
    /*
    cv::Mat img;
    cv::Mat img_copy;
    UIImageToMat([UIImage imageNamed:@"face11.jpg"], img);
    cvtColor(img, img, CV_BGRA2BGR);
    cvtColor(img, img_copy, CV_BGRA2BGR);
    
    [self findDifferenceBetweenSkinAndForegroundInImage:img withCopy:img_copy toDest:img];
    [self imwrite:img withName:@"xy"];
    [self findContoursInImage:img];
    
    // apply mask
    img.convertTo(img, CV_8U);
    cv::cvtColor(img, img, CV_BGR2GRAY);
    img_copy.setTo(cv::Scalar(0,0,255), img);
    
    [self imwrite:img_copy withName:@"x"];
    self.imageView.image = MatToUIImage(img_copy);
    NSLog(@"done.");
     */
}

- (std::string) getBundlePathForResourceWithName:(char *)name andType:(char *)type
{
    // get main app bundle
    NSBundle *appBundle = [NSBundle mainBundle];
    
    // constant file name
    NSString *fileName = [NSString stringWithUTF8String:name];
    NSString *fileType = [NSString stringWithUTF8String:type];
    
    // get file path in bundle
    NSString *pathInBundle = [appBundle pathForResource:fileName ofType:fileType];
    
    // convert NSString to std::string
    std::string stringPath([pathInBundle UTF8String]);
    
    return stringPath;
}

- (IBAction)didPressButton:(id)sender {
    isCameraOn = !isCameraOn;
    if(isCameraOn){
        [self.videoCamera start];
        [self.cameraButton setTitle:@"STOP" forState:UIControlStateNormal];
    }
    else{
        [self.videoCamera stop];
        [self.cameraButton setTitle:@"START" forState:UIControlStateNormal];
        self.imageView.image = MatToUIImage(lastFrame);
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();//CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat cvMat(rows, cols, CV_8UC3); // 8 bits per component, 4 channels (color channels + alpha)
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    return cvMat;
}

- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    return cvMat;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return finalImage;
}

- (UIImage *) thresholdImage:(UIImage *)img
{
    int threshold_value = 0;
    
    int const max_BINARY_value = 2147483647;
    cv::Mat src_gray=[self cvMatFromUIImage:img];
    cv::Mat dst;
    dst=src_gray;
    cv::cvtColor(src_gray, dst, cv::COLOR_RGB2GRAY);
    cv::Mat canny_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::RNG rng(12345);
    
    cv::threshold( dst, dst, threshold_value, max_BINARY_value, cv::THRESH_OTSU );
    
    return [self UIImageFromCVMat:dst];
}

- (cv::Mat) grabCut
{
    // Declare vars
    cv::Mat src;
    cv::Mat src8UC3;
    cv::Mat grabCut;
    
    // Load image
    UIImageToMat([UIImage imageNamed:@"face1.jpg"], src, false);
    
    // Convert to CV_8UC3 because grabCut() needs it
    cvtColor(src, src8UC3, CV_BGRA2BGR);
    
    /*
    if(src8UC3.type() == CV_8UC3){
        NSLog(@"pikeachu.");
    }
    else if(src8UC3.type() == CV_8UC4){
        NSLog(@"raihu.");
    }
     */
    
    // Initialise stuff for grabCut
    cv::Mat result(src8UC3.size(), src8UC3.type());
    cv::Mat bgModel;// background model
    cv::Mat fgModel;// foreground model
    
    // Draw a rectangle
    cv::Rect rectangle(1,1,src8UC3.cols-1,src8UC3.rows-1);
    
    // Perform grabcut
    cv::grabCut(src8UC3, result, rectangle, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);
    cv::compare(result, cv::Scalar(3,3,3), result, cv::CMP_EQ);
    grabCut = cv::Mat(src8UC3.size(), CV_8UC3, cv::Scalar(255,255,255));
    src8UC3.copyTo(grabCut, result);
    
    //self.imageView.image = MatToUIImage(grabCut);
    return grabCut;
}

- (cv::Mat) grabCut:(cv::Mat)src
{
    // Declare vars
    cv::Mat src8UC3;
    cv::Mat grabCut;
    
    // Convert to CV_8UC3 because grabCut() needs it
    cvtColor(src, src8UC3, CV_BGRA2BGR);
    
    /*
     if(src8UC3.type() == CV_8UC3){
     NSLog(@"pikeachu.");
     }
     else if(src8UC3.type() == CV_8UC4){
     NSLog(@"raihu.");
     }
     */
    
    // Initialise stuff for grabCut
    cv::Mat result(src8UC3.size(), src8UC3.type());
    cv::Mat bgModel;// background model
    cv::Mat fgModel;// foreground model
    
    // Draw a rectangle
    cv::Rect rectangle(1,1,src8UC3.cols-1,src8UC3.rows-1);
    
    // Perform grabcut
    cv::grabCut(src8UC3, result, rectangle, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);
    cv::compare(result, cv::Scalar(3,3,3), result, cv::CMP_EQ);
    grabCut = cv::Mat(src8UC3.size(), CV_8UC3, cv::Scalar(255,255,255));
    src8UC3.copyTo(grabCut, result);
    
    //self.imageView.image = MatToUIImage(grabCut);
    return grabCut;
}


- (void) grabCutImage:(cv::Mat&)src8UC3 withCopy:(cv::Mat&)src_copy
{
     if(src8UC3.type() != CV_8UC3 || src_copy.type() != CV_8UC3){
         [NSException raise:@"Images must be of type CV_8UC3" format:@""];
     }
    
    // Initialise stuff for grabCut
    cv::Mat result(src8UC3.size(), src8UC3.type());
    cv::Mat bgModel;// background model
    cv::Mat fgModel;// foreground model
    
    // Draw a rectangle
    cv::Rect rectangle(1,1,src8UC3.cols-1,src8UC3.rows-1);
    
    // Perform grabcut
    cv::grabCut(src8UC3, result, rectangle, bgModel, fgModel, 10, cv::GC_INIT_WITH_RECT);
    cv::compare(result, cv::Scalar(3,3,3), result, cv::CMP_EQ);
    src8UC3.setTo(cv::Scalar(255,255,255));
    src_copy.copyTo(src8UC3, result);
}

- (void) performskinSegmentationOnImage:(cv::Mat&)grabCut
{
    // Ensure BGR
    cv::Mat grabCut_copy;
    cvtColor(grabCut, grabCut_copy, CV_RGB2BGR);
    
    cvtColor(grabCut, grabCut, CV_RGB2BGR);
    grabCut.setTo(cv::Scalar(0,0,255));
    cv::Mat skinMask;
    cv::Mat hsvMatrix;
    
    cv::Scalar lower(0,48,80);//lower(120,120,120);
    cv::Scalar upper(20,255,255);//upper(240,180,280);
    
    cvtColor(grabCut_copy, hsvMatrix, CV_BGR2HSV);
    cv::inRange(hsvMatrix, lower, upper, skinMask);
    
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));
    cv::erode(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    
    cv::GaussianBlur(skinMask, skinMask, cv::Size(3,3), 0);
    cv::bitwise_and(grabCut_copy, grabCut_copy, grabCut, skinMask);
    [self imwrite:grabCut withName:@"skin"];
}

- (cv::Mat) skinSegmentation:(cv::Mat)grabCut
{
    cvtColor(grabCut, grabCut, CV_RGB2BGR);
    //self.imageView.image = MatToUIImage(grabCut);
    
    
    cv::Mat skinDetection(grabCut.size(), grabCut.type());
    skinDetection.setTo(cv::Scalar(0,0,255));
    cvtColor(skinDetection, skinDetection, CV_RGB2BGR);
    cv::Mat skinMask;
    cv::Mat hsvMatrix;
    
    cv::Scalar lower(40,0,0);//lower(120,120,120);
    cv::Scalar upper(180,255,255);//upper(240,180,280);
    
    cvtColor(grabCut, hsvMatrix, CV_BGR2HSV);
    cv::inRange(hsvMatrix, lower, upper, skinMask);
    [self imwrite:skinMask withName:@"mask"];
    
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));
    cv::erode(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    
    cv::GaussianBlur(skinMask, skinMask, cv::Size(3,3), 0);
    cv::bitwise_and(grabCut, grabCut, skinDetection, skinMask);
    
    //self.imageView.image = MatToUIImage(skinDetection);
    [self uiimwrite:MatToUIImage(skinDetection) withName:@"skin_hsv"];
    return skinDetection;
}

- (cv::Mat) skinSegmentation
{
//    cv::Mat src;
//    cv::Mat src8UC3;
//    UIImageToMat([UIImage imageNamed:@"face1.jpg"], src, false);
//    cvtColor(src, src8UC3, CV_BGRA2BGR);
    
    cv::Mat grabCut = [self grabCut];
    cvtColor(grabCut, grabCut, CV_RGB2BGR);
    //self.imageView.image = MatToUIImage(grabCut);

    
    cv::Mat skinDetection(grabCut.size(), grabCut.type());
    skinDetection.setTo(cv::Scalar(0,0,255));
    cvtColor(skinDetection, skinDetection, CV_RGB2BGR);
    cv::Mat skinMask;
    cv::Mat hsvMatrix;
    
    cv::Scalar lower(120,0,0);//lower(120,120,120);
    cv::Scalar upper(137,255,255);//upper(240,180,280);
    
    cvtColor(grabCut, hsvMatrix, CV_BGR2HSV);
    cv::inRange(hsvMatrix, lower, upper, skinMask);
    
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));
    cv::erode(skinMask, skinMask, kernel);
    cv::erode(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    
    cv::GaussianBlur(skinMask, skinMask, cv::Size(3,3), 0);
    cv::bitwise_and(grabCut, grabCut, skinDetection, skinMask);
    
    //self.imageView.image = MatToUIImage(skinDetection);
    return skinDetection;
}


- (void) detectFace
{
    std::vector<cv::Rect> faces;
    cv::Mat frame;
    cv::Mat frame_gray;
    
    UIImageToMat([UIImage imageNamed:@"face5.jpg"], frame, false);
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    
    for(size_t i = 0; i < faces.size(); i++){
        cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        
        // draw around face
        cv::rectangle(frame, cv::Point(faces[i].x,faces[i].y), cv::Point(faces[i].x+faces[i].width,faces[i].y+faces[i].height), cv::Scalar(255,0,0),2);
        //cv::ellipse(frame, center, cv::Size(faces[i].width*0.5,faces[i].height*0.5), 0, 0, 360, cv::Scalar(255,0,255), 4, 8, 0);
        
        // TODO: Maybe detect eyes to make face skin colour detection more accurate
    }
    
    self.imageView.image = MatToUIImage(frame);
}

- (void) markFacesInFrame:(cv::Mat)frame
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    
    cvtColor(frame, frame_gray, CV_BGRA2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    
    for(size_t i = 0; i < faces.size(); i++){
        cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        
        // draw around face
        cv::rectangle(frame, cv::Point(faces[i].x,faces[i].y), cv::Point(faces[i].x+faces[i].width,faces[i].y+faces[i].height), cv::Scalar(0,255,0), 4);
        //cv::ellipse(frame, center, cv::Size(faces[i].width*0.5,faces[i].height*0.5), 0, 0, 360, cv::Scalar(255,0,255), 4, 8, 0);
        // TODO: Maybe detect eyes to make face skin colour detection more accurate
    }
}

- (std::vector<cv::Rect>) detectFacesInImage:(cv::Mat)frame
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    
    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
    
    return faces;
}

- (void) histogramsForImage:(cv::Mat)src
{
    // convert image to hsv
    cv::Mat src_hsv;
    cvtColor(src, src_hsv, CV_BGR2HSV);
    
    // separate channels into individual matrices
    std::vector<cv::Mat> hsv_planes;
    cv::split(src_hsv, hsv_planes);
    
    // Establish the number of bins
    int h_histSize = 180;
    int s_histSize = 256;
    int v_histSize = 256;
    
    // Set the ranges
    float range_h[] = { 0, 180 } ;
    float range_s[] = { 0, 256 } ;
    float range_v[] = { 0, 256 } ;
    const float* h_histRange = { range_h };
    const float* s_histRange = { range_s };
    const float* v_histRange = { range_v };
    
    bool uniform = true; bool accumulate = false;
    cv::Mat h_hist, s_hist, v_hist;
    
    // Compute the histograms
    cv::calcHist(&hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &h_histSize, &h_histRange, uniform, accumulate);
    cv::calcHist(&hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &s_histSize, &s_histRange, uniform, accumulate);
    cv::calcHist(&hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &v_histSize, &v_histRange, uniform, accumulate);
    double min,max;
    cv::minMaxLoc(h_hist, &min, &max);
    NSLog(@"MIN,MAX VALS => %f,%f", min, max);
    
    // Draw the histogram for h
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/h_histSize );
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255,255,255));
    
    // Normalize the result to [ 0, histImage.rows ]
    normalize(h_hist, h_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for( int i = 1; i < h_histSize; i++ )
    {
        cv::line( histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
                 cv::Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
                 cv::Scalar( 255, 0, 0), 2, 8, 0  );
    }
    [self imwrite:histImage withName:@"h_hist"];
    
    // Draw the histogram for s
    bin_w = cvRound( (double) hist_w/s_histSize );
    histImage.setTo(cv::Scalar(255,255,255));
    normalize(s_hist, s_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for( int i = 1; i < s_histSize; i++ )
    {
        cv::line( histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(s_hist.at<float>(i-1)) ) ,
                 cv::Point( bin_w*(i), hist_h - cvRound(s_hist.at<float>(i)) ),
                 cv::Scalar( 0, 255, 0), 2, 8, 0  );
    }
    [self imwrite:histImage withName:@"s_hist"];
    
    // Draw the histogram for v
    bin_w = cvRound( (double) hist_w/v_histSize );
    histImage.setTo(cv::Scalar(255,255,255));
    normalize(v_hist, v_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    for( int i = 1; i < v_histSize; i++ )
    {
        cv::line( histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(v_hist.at<float>(i-1)) ) ,
                 cv::Point( bin_w*(i), hist_h - cvRound(v_hist.at<float>(i)) ),
                 cv::Scalar( 0, 0, 255), 2, 8, 0  );
    }
    [self imwrite:histImage withName:@"v_hist"];
    
    
    [self imwrite:src withName:@"face"];
}

- (void) skinColor
{
    cv::Mat frame;
    cv::Mat faceROI;
    cv::Mat faceROI_hsv;
    
    // detect face regions
    UIImageToMat([UIImage imageNamed:@"face1.jpg"], frame, false);
    
    std::vector<cv::Rect> faces = [self detectFacesInImage:frame];
    
    for(size_t i = 0; i < faces.size(); i++){
        //Get face as ROI image
        faceROI = frame(faces[i]);
        
        // find average of pixel values in face
        cv::Mat1b meanMask(faceROI.rows, faceROI.cols);
//        cv::Scalar mean = cv::mean(faceROI, meanMask); // Vec3b where each element contains the mean for each channel ordered BGR
//        NSLog(@"BGR MEAN: (%f, %f, %f)", mean[0], mean[1], mean[2]);
//        
//        cvtColor(faceROI, faceROI_hsv, CV_BGR2HSV);
//        cv::Mat1b meanMask_hsv(faceROI_hsv.rows, faceROI_hsv.cols);
//        cv::Scalar mean_hsv = cv::mean(faceROI_hsv, meanMask_hsv);
//        NSLog(@"HSV MEAN: (%f, %f, %f)", mean_hsv[0], mean_hsv[1], mean_hsv[2]);
        cv::Scalar mean_hsv;
        cv::Scalar stdDev_hsv;
        cvtColor(faceROI, faceROI_hsv, CV_BGR2HSV);
        cv::meanStdDev(faceROI_hsv, mean_hsv, stdDev_hsv);
        NSLog(@"HSV MEAN: (%f, %f, %f)", mean_hsv[0], mean_hsv[1], mean_hsv[2]);
        NSLog(@"HSV STD_DEV: (%f, %f, %f)", stdDev_hsv[0], stdDev_hsv[1], stdDev_hsv[2]);
        [self histogramsForImage:faceROI];
    }
    
    //[self imwrite:faceROI_hsv];
}

- (cv::Mat) differenceBetweenSkinAndFullImage
{
    /* Prep */
    // Declare vars
    cv::Mat src;
    
    // Load image
    UIImageToMat([UIImage imageNamed:@"face5.jpg"], src, false);
    cvtColor(src, src, CV_BGRA2BGR);
    
    
    // Grabcut
    cv::Mat grabCut = [self grabCut:src];
   // [self imwrite:grabCut withName:@"grabcut"];
    
    // Skin detection
    cv::Mat skinDetection = [self skinSegmentation:grabCut];
    //[self imwrite:skinDetection withName:@"skin_rgb"];
    cvtColor(skinDetection, skinDetection, CV_RGB2BGR);
    
    
    cv::Mat difference(src.size(), src.type());
    difference.setTo(cv::Scalar(255,255,255));
    int rows = difference.rows;
    int cols = difference.cols;
    
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            cv::Vec3b grabcut_pixel_vals = grabCut.at<cv::Vec3b>(r, c);
            cv::Vec3b skin_pixel_vals = skinDetection.at<cv::Vec3b>(r, c);
            //extract those pixels which are non blue/nonwhite in 1st image and red in 2nd image
            if( ((grabcut_pixel_vals[0] != 255 ) && (grabcut_pixel_vals[1] != 255 ) && (grabcut_pixel_vals[2] != 255)) &&
                ((skin_pixel_vals[0] == 0) && (skin_pixel_vals[1] == 0) &&(skin_pixel_vals[2] == 255) )){
                difference.at<cv::Vec3b>(r, c) = src.at<cv::Vec3b>(r, c);
                //NSLog(@"pikachu.");
            }
        }
    }
    
    //self.imageView.image = MatToUIImage(difference);
    //[self imwrite:difference withName:@"diff"];
    
    
    /****** EROSION & DILUTION *******/
    cv::Mat morph(src.size(), src.type());
    int erosionSize = 2;
    cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(2*erosionSize+1,2*erosionSize+1));
    cv::erode(difference, morph, erosionKernel);
    
    self.imageView.image = MatToUIImage(difference);
    //[self imwrite:difference withName:@"erosion"];
    
    return difference;
}

- (void) findDifferenceBetweenSkinAndForegroundInImage:(cv::Mat)img withCopy:(cv::Mat)src toDest:(cv::Mat&)difference
{
    cv::Mat img_copy;
    cvtColor(img, img, CV_BGRA2BGR);
    cvtColor(img, img_copy, CV_BGRA2BGR);
    
    [self grabCutImage:img withCopy:img_copy];
    cv::Mat grabCut(img.size(), img.type());
    cvtColor(img, grabCut, CV_RGB2BGR);
    
    [self performskinSegmentationOnImage:img];
    cv::Mat skinDetection(img.size(), img.type());
    cvtColor(img, skinDetection, CV_RGB2BGR);
    
    difference.setTo(cv::Scalar(255,255,255));
    int rows = difference.rows;
    int cols = difference.cols;
    
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++){
            cv::Vec3b grabcut_pixel_vals = grabCut.at<cv::Vec3b>(r, c);
            cv::Vec3b skin_pixel_vals = skinDetection.at<cv::Vec3b>(r, c);
            
            //extract those pixels which are non blue/nonwhite in 1st image and red in 2nd image
            if( ((grabcut_pixel_vals[0] != 255 ) && (grabcut_pixel_vals[1] != 255 ) && (grabcut_pixel_vals[2] != 255)) &&
               ((skin_pixel_vals[0] == 255) && (skin_pixel_vals[1] == 0) &&(skin_pixel_vals[2] == 0) )){
                difference.at<cv::Vec3b>(r, c) = src.at<cv::Vec3b>(r, c);
                //NSLog(@"pikachu.");
            }
        }
    }
    
    
    
//    //self.imageView.image = MatToUIImage(difference);
//    //[self imwrite:difference withName:@"diff"];
//    
//    
//    /****** EROSION & DILUTION *******/
//    cv::Mat morph(src.size(), src.type());
//    int erosionSize = 2;
//    cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(2*erosionSize+1,2*erosionSize+1));
//    cv::erode(difference, morph, erosionKernel);
//    
//    self.imageView.image = MatToUIImage(difference);
//    //[self imwrite:difference withName:@"erosion"];
//    
//    return difference;
}

- (void) findContours
{
    cv::Mat grayImage;
    cv::Mat cannyImage;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat difference = [self differenceBetweenSkinAndFullImage];
    self.imageView.image = MatToUIImage(difference);
    
    cvtColor(difference, grayImage, CV_BGR2GRAY);
    cv::Canny(grayImage, cannyImage, 100, 200);
    
    //morph edge detected image to improve egde connectivity
    cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(5,5));
    cv::morphologyEx(cannyImage, cannyImage, cv::MORPH_CLOSE, kernel);
    [self imwrite:cannyImage withName:@"canny"];
    
    cv::Mat hierarchy;
    cv::findContours(cannyImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat imgContoured(cannyImage.size(), CV_8UC3);
    imgContoured.setTo(cv::Scalar(255,255,255));
    
    double maxArea = cv::contourArea(contours.at(0));
    int maxAreaIndex = 0;
    std::vector<cv::Point> temp_contour;
    for(int i = 1; i < contours.size(); i++){
        temp_contour = contours.at(i);
        double curr_cont_area = cv::contourArea(temp_contour);
        if(maxArea < curr_cont_area){
            maxArea = curr_cont_area;
            maxAreaIndex = i;
        }
    }
    cv::drawContours(imgContoured, contours, maxAreaIndex, cv::Scalar(0,0,0), -1);
    
    // Create mask for finding hair
    
    
    self.imageView.image = MatToUIImage(imgContoured);
}

- (void) findContoursInImage:(cv::Mat)img
{
    cv::Mat grayImage;
    cv::Mat cannyImage;
    std::vector<std::vector<cv::Point>> contours;
    self.imageView.image = MatToUIImage(img);
    
    cvtColor(img, grayImage, CV_BGR2GRAY);
    cv::Canny(grayImage, cannyImage, 100, 200);
    
    //morph edge detected image to improve egde connectivity
    cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(5,5));
    cv::morphologyEx(cannyImage, cannyImage, cv::MORPH_CLOSE, kernel);
    [self imwrite:cannyImage withName:@"canny"];
    
    cv::Mat hierarchy;
    cv::findContours(cannyImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    img.setTo(cv::Scalar(0,0,0));
    
    double maxArea = cv::contourArea(contours.at(0));
    int maxAreaIndex = 0;
    std::vector<cv::Point> temp_contour;
    for(int i = 1; i < contours.size(); i++){
        temp_contour = contours.at(i);
        double curr_cont_area = cv::contourArea(temp_contour);
        if(maxArea < curr_cont_area){
            maxArea = curr_cont_area;
            maxAreaIndex = i;
        }
    }
    cv::drawContours(img, contours, maxAreaIndex, cv::Scalar(255,255,255), -1);
    
    // Create mask for finding hair
    // make img a binary mask
    cvtColor(img, img, CV_BGR2GRAY);
    
}

#pragma mark - Protocol CvVideoCameraDelegate

- (void)processImage:(cv::Mat&)image;
{
    /*
    [self markFacesInFrame:image];
    lastFrame = image;
     */
    cv::Mat img_copy;
    cvtColor(image, image, CV_BGRA2BGR);
    cvtColor(image, img_copy, CV_BGRA2BGR);
    
    [self findDifferenceBetweenSkinAndForegroundInImage:img_copy withCopy:image toDest:img_copy];
    [self findContoursInImage:img_copy];
    
    // apply mask
    img_copy.convertTo(img_copy, CV_8U);
    cv::cvtColor(img_copy, img_copy, CV_BGR2GRAY);
    image.setTo(cv::Scalar(0,0,255), img_copy);
    cvtColor(image, image, CV_BGR2RGB);
}

- (void) imwrite:(cv::Mat)img
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *directory = [paths objectAtIndex:0];
    NSString *filePath = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"hand%f.jpg", CFAbsoluteTimeGetCurrent()]];
    const char* filePathC = [filePath cStringUsingEncoding:NSMacOSRomanStringEncoding];
    
    const cv::String thisPath = (const cv::String)filePathC;
    
    //Save image
    imwrite(thisPath, img);
    
    NSLog(@"\nIMAGE SAVED TO %@\n", filePath);
}

- (void) uiimwrite:(UIImage *)img
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *directory = [paths objectAtIndex:0];
    NSString *filePath = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"hand%f.jpg", CFAbsoluteTimeGetCurrent()]];
    
    [UIImageJPEGRepresentation(img, 1.0) writeToFile:filePath atomically:YES];
    
    NSLog(@"\nIMAGE SAVED TO %@\n", filePath);
}

- (void) imwrite:(cv::Mat)img withName:(NSString *)name
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *directory = [paths objectAtIndex:0];
    NSString *filePath = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"%@_%f.jpg", name, CFAbsoluteTimeGetCurrent()]];
    const char* filePathC = [filePath cStringUsingEncoding:NSMacOSRomanStringEncoding];
    
    const cv::String thisPath = (const cv::String)filePathC;
    
    //Save image
    imwrite(thisPath, img);
    
    NSLog(@"\nIMAGE SAVED TO %@\n", filePath);
}

- (void) uiimwrite:(UIImage *)img withName:(NSString *)name
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *directory = [paths objectAtIndex:0];
    NSString *filePath = [directory stringByAppendingPathComponent:[NSString stringWithFormat:@"%@_%f.jpg", name, CFAbsoluteTimeGetCurrent()]];
    
    // Save image.
    [UIImageJPEGRepresentation(img, 1.0) writeToFile:filePath atomically:YES];
    
    NSLog(@"\nIMAGE SAVED TO %@\n", filePath);
}

@end
