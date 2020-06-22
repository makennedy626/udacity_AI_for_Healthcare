Project Specification

Pneumonia Detection from Chest X-Rays

Exploratory Data Analysis
Criteria 	Meets Specifications

The student can create visualizations of the metadata that inform model training
	

    Students create distributions of diseases and comorbidities in their dataset
    Students create distributions of basic demographics of the patients who make up their datasets (such as age, gender, patient position,etc.)
    Students can use the above distributions to draw conclusions about how they will need to set up their model training

The student can visualize relevant properties of pixel-level data
	

    Students use python’s imshow to visualize medical images during EDA
    Students create distributions of intensity values of the pixel-level data within images and compare them both within and across diagnoses
    Students use both of these methods of inspecting images to draw meaningful conclusions about what their model will train on

Model Building & Training
Criteria 	Meets Specifications

The student creates an appropriate train-test split of the data
	

    Students create a set of training data and a set of validation data that each have the appropriate proportions of positive and negative cases for their intended use (training and validation)

The student implements appropriate data augmentation to their training data
	

    Student implements a class such as ImageDataGenerator from Keras to augment their training data only
    Student should not augment testing/validation data
    Student uses types of augmentation that are appropriate for medical imaging. There are no required types of augmentation
    Students should normalize the imaging data so the model weights do not go to infinity.

The student evaluates the performance of their model using the appropriate statistics
	

    Student monitors the training progress of their model using log loss
    Student changes training parameters to avoid overfitting and compares performances of different training paradigms.
    Student trains enough epochs until the loss is “stable”
    After training, student uses precision, recall, and F1 score to actually evaluate the utility of their model.
    Find a threshold to classify if an image is pneumonia or not.
    Students should show precision-recall curve and a curve of F1-score vs. threshold

The student can integrate their model with real-world medical imaging data
	

    Student can check DICOM header for image position, image type and body part on ALL .dcm files to check validity for their model using the pydicom python package.
    Student can read imaging data in from a .dcm file, preprocess the image and feed it into their model using the pydicom python package.

FDA Description and Validation Plan
Criteria 	Meets Specifications

The student can describe the intended population and the clinical impact of their model
	

    Student should provide an intended use statement
    Student should point to data from their EDA to describe who their algorithm is indicated for and what the clinical setting is in which their algorithm would be used
    Student should describe limitations of their algorithm and how false positives or false negatives might affect a patient

The student can describe how their model was designed and trained
	

    Students provide a flowchart or architecture diagram of their model
    Students should describe the DICOM checks they use before sending an image through their algorithm
    Students should describe the preprocessing steps they use.
    Students should describe the architecture of the classifier
    Students should describe augmentation and its parameters used
    Students describe the parameters used for training
    Students should show the behavior of training and validating loss
    Students should describe the performance statistics and threshold used in final validation

The student can describe the dataset used to train the algorithm and how the ground truth was created
	

    Students should provide information for the training set
    Students should provide information for the validating set
    Students should describe how the ground truth of the NIH dataset is created, the benefit and limitations.

The student describes how they would create a FDA Validation set, ground truth, and what performance metric they would hold their algorithm to for FDA validation of their model.
	

    Students should describe the ideal dataset that they would receive from a clinical partner for their FDA Validation Dataset
    Students should describe how they would ideally create ground truth for this FDA Validation Dataset
    Students should describe the performance metric and the metic value that they would hold their algorithm to, supported by literature

Suggestions to Make Your Project Stand Out!

    Create some of your own custom image augmentation (such as different image filtering techniques) rather than solely using those predefined by Keras’ ImageDataGenerator.
    Try creating two ‘nested’ models to specifically predict pneumonia. One that predicts pneumonia and/or infiltrates at the top level, and then a second model that specifically predicts pneumonia from the positive cases returned by the first model.
    Have your model output a class activation map in addition to a single binary prediction of pneumonia. This map will help a clinician to understand what the model is detecting as probable pneumonia in each image.

