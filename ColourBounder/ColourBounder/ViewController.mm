//
//  ViewController.m
//  ColourBounder
//
//  Created by Odie Edo-Osagie on 20/05/2018.
//  Copyright © 2018 Odie Edo-Osagie. All rights reserved.
//

#import "ViewController.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgcodecs/ios.h>

@interface ViewController (){
    float upperRValue;
    float upperGValue;
    float upperBValue;
    float lowerRValue;
    float lowerGValue;
    float lowerBValue;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    
    upperRValue = 60;
    upperGValue = 110;
    upperBValue = 255;
    lowerRValue = 50;
    lowerGValue = 100;
    lowerBValue = 100;
    
    // Load test image
    self.imageView.image = [UIImage imageNamed:@"blocks.png"];
}

- (void) viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    
//    // Initialze sliders
//    [self.upperRSlider setValue:110 animated:YES]; //. value = 110;
//    [self.upperGSlider setValue:60 animated:YES];//.value = 60;
//    [self.upperBSlider setValue:255 animated:YES];//.value = 255;
//    [self.lowerRSlider setValue:100 animated:YES];//value = 100;
//    [self.lowerGSlider setValue:50 animated:YES];//.value = 50;
//    [self.lowerBSlider setValue:0 animated:YES];//.value = 0;
//    NSLog(@"%f", upperGValue);
}

- (void) extractColourWithBounds
{
    // Load image
    cv::Mat img;
    UIImageToMat([UIImage imageNamed:@"blocks.png"], img, false);
    
    // Create matrix for colour detected image
    cv::Mat colourDetection(img.size(), img.type());
    //colourDetection.setTo(cv::Scalar(255,0,0));
    cvtColor(img, colourDetection, CV_RGB2BGR);
    
    self.imageView.image = [self UIImageFromCVMat:colourDetection];
    
    // Filter using bounds
    cv::Mat colourMask;
    cv::Scalar lower(lowerBValue, lowerGValue, lowerRValue);
    cv::Scalar upper(upperBValue, upperGValue, upperRValue);
    cv::inRange(colourDetection, lower, upper, colourMask);
    
    //self.imageView.image = [self UIImageFromCVMat:colourMask];

    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11,11)));
    cv::erode(colourMask, colourMask, kernel);
    cv::dilate(colourMask, colourMask, kernel);

    cv::GaussianBlur(colourMask, colourMask, cv::Size(3,3), 0);
    cv::bitwise_and(img, img, colourDetection, colourMask);

    self.imageView.image = [self UIImageFromCVMat:colourDetection];
    
    NSLog(@"\nLOWER: (%f, %f, %f)\nUPPER: (%f, %f, %f)", lowerRValue, lowerGValue, lowerBValue, upperRValue, upperGValue, upperBValue);
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

# pragma mark - Helper

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

# pragma mark - Actions
    
- (IBAction)didChangeUpperRSlider:(id)sender
{
    upperRValue = self.upperRSlider.value;
    self.upperRLabel.text = [NSString stringWithFormat:@"%.1f", upperRValue];
    [self extractColourWithBounds];
}

- (IBAction)didChangeUpperGSlider:(id)sender
{
    upperGValue = self.upperGSlider.value;
    self.upperGLabel.text = [NSString stringWithFormat:@"%.1f", upperGValue];
    [self extractColourWithBounds];
}

- (IBAction)didChangeUpperBSlider:(id)sender
{
    upperBValue = self.upperBSlider.value;
    self.upperBLabel.text = [NSString stringWithFormat:@"%.1f", upperBValue];
    [self extractColourWithBounds];
}

- (IBAction)didChangeLowerRSlider:(id)sender
{
    lowerRValue = self.lowerRSlider.value;
    self.lowerRLabel.text = [NSString stringWithFormat:@"%.1f", lowerRValue];
    [self extractColourWithBounds];
}

- (IBAction)didChangeLowerGSlider:(id)sender
{
    lowerGValue = self.lowerGSlider.value;
    self.lowerGLabel.text = [NSString stringWithFormat:@"%.1f", lowerGValue];
    [self extractColourWithBounds];
}

- (IBAction)didChangeLowerBSlider:(id)sender
{
    lowerBValue = self.lowerBSlider.value;
    self.lowerBLabel.text = [NSString stringWithFormat:@"%.1f", lowerBValue];
    [self extractColourWithBounds];
}





@end
