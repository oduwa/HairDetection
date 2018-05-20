//
//  ViewController.h
//  ColourBounder
//
//  Created by Odie Edo-Osagie on 20/05/2018.
//  Copyright © 2018 Odie Edo-Osagie. All rights reserved.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController

    @property (weak, nonatomic) IBOutlet UIImageView *imageView;
    @property (weak, nonatomic) IBOutlet UISlider *upperRSlider;
    @property (weak, nonatomic) IBOutlet UISlider *upperGSlider;
    @property (weak, nonatomic) IBOutlet UISlider *upperBSlider;
    @property (weak, nonatomic) IBOutlet UISlider *lowerRSlider;
    @property (weak, nonatomic) IBOutlet UISlider *lowerGSlider;
    @property (weak, nonatomic) IBOutlet UISlider *lowerBSlider;
    @property (weak, nonatomic) IBOutlet UILabel *upperRLabel;
    @property (weak, nonatomic) IBOutlet UILabel *upperGLabel;
    @property (weak, nonatomic) IBOutlet UILabel *upperBLabel;
    @property (weak, nonatomic) IBOutlet UILabel *lowerRLabel;
    @property (weak, nonatomic) IBOutlet UILabel *lowerGLabel;
    @property (weak, nonatomic) IBOutlet UILabel *lowerBLabel;
    
    
@end
