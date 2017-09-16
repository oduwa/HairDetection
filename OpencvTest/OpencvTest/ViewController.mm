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

@interface ViewController ()

@end

@implementation ViewController

cv::CascadeClassifier face_cascade;

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Initialize face classifier
    char *cascadeName = (char*)"haarcascade_frontalface_alt";
    char *cascadeType = (char*)"xml";
    if (face_cascade.load([self getBundlePathForResourceWithName:cascadeName andType:cascadeType])){
        printf("Load complete");
    }else{
        printf("Load error");
    }
    
//    cv::Mat grabCut = [self grabCut];
//    self.imageView.image = MatToUIImage(grabCut);
    
    //[self skinSegmentation];
    //[self detectFace];
    [self skinColor];
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
    self.imageView.image = [UIImage imageNamed:@"face5.jpg"];
    UIImageToMat([UIImage imageNamed:@"face5.jpg"], src, false);
    
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

- (void) skinDetection
{
    cv::Mat src;
    cv::Mat src8UC3;
    UIImageToMat([UIImage imageNamed:@"face1.jpg"], src, false);
    cvtColor(src, src8UC3, CV_BGRA2BGR);
    
    cv::Mat grabCut = [self grabCut];
    cv::Mat src_YCrCb(grabCut.size(), CV_8SC3);
    cv::Mat src_hsv(grabCut.size(), CV_8SC3);
    cv::Mat skinDetection(grabCut.size(), grabCut.type());
    skinDetection.setTo(cv::Scalar(0,0,255));
    
    cvtColor(grabCut, src_YCrCb, CV_BGR2YCrCb);
    grabCut.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    cv::normalize(src_hsv, src_hsv, 0.00, 255.00, cv::NORM_MINMAX, CV_32FC3);
    
    for(int r = 0; r < grabCut.rows; r++){
        for(int c = 0; c< grabCut.cols; c++){
            cv::Vec3b pixel_val_rgb = grabCut.at<cv::Vec3b>(r, c);
            int b = (int) pixel_val_rgb[0];
            int g = (int) pixel_val_rgb[1];
            int r = (int) pixel_val_rgb[2];
            bool a1 = R1(r, g, b);
            
            cv::Vec3b Pixel_val_YCrCb = src_YCrCb.at<cv::Vec3b>(r, c);
            int Y = (int) Pixel_val_YCrCb[0];
            int Cr = (int) Pixel_val_YCrCb[1];
            int Cb = (int) Pixel_val_YCrCb[2];
            bool a2 = R2(Y, Cr, Cb);
            
            cv::Vec3b pixel_val_hsv = src_hsv.at<cv::Vec3b>(r, c);
            int h = (int) pixel_val_hsv[0];
            int s = (int) pixel_val_hsv[1];
            int v = (int) pixel_val_hsv[2];
            bool a3 = R3(h, s, v);
            
            if(!(a1 && a2 && a3)){
                skinDetection.at<cv::Vec3b>(r, c) = cv::Vec3b(0,0,255);
            }
            else{
                skinDetection.at<cv::Vec3b>(r, c) = src8UC3.at<cv::Vec3b>(r, c);
            }
        }
    }
    
    self.imageView.image = MatToUIImage(skinDetection);
}

- (void) skinSegmentation
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
    
    cv::Scalar lower(0,48,80);//lower(120,120,120);
    cv::Scalar upper(20,255,255);//upper(240,180,280);
    
    cvtColor(grabCut, hsvMatrix, CV_BGR2HSV);
    cv::inRange(hsvMatrix, lower, upper, skinMask);
    
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));
    cv::erode(skinMask, skinMask, kernel);
    cv::erode(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    cv::dilate(skinMask, skinMask, kernel);
    
    cv::GaussianBlur(skinMask, skinMask, cv::Size(3,3), 0);
    cv::bitwise_and(grabCut, grabCut, skinDetection, skinMask);
    
    self.imageView.image = MatToUIImage(skinDetection);
}

bool R1(int R,int G,int B)
{
    bool e1 = (R>95) && (G>40) && (B>20) && ((MAX(R,MAX(G,B)) - MIN(R,MIN(G,B)))>15) && (ABS(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (ABS(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}

bool R2(float Y, float Cr, float Cb)
{
    bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V)
{
    return (H<25) || (H > 230);
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

- (void) skinColor
{
    cv::Mat frame;
    cv::Mat faceROI;
    cv::Mat faceROI_hsv;
    
    // detect face regions
    UIImageToMat([UIImage imageNamed:@"face5.jpg"], frame, false);
    
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
    }
    
    [self imwrite:faceROI_hsv];
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


@end
