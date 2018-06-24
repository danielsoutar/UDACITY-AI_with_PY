#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: Emily
# DATE CREATED: May 31, 2018
# REVISED DATE: 06/21/2018 <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():
    # TODO: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # TODO: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()

    # TODO: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # TODO: 4. Define classify_images() function to create the classifier
    # labels with the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # TODO: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # TODO: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # TODO: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\nTotal Elapsed Runtime:", str(int((tot_time / 3600))) + ":" +
          str(int(((tot_time % 3600) / 60))) + ":" +
          str(round(((tot_time % 3600) % 60))))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# parameters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Argument 1: dir
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the pet image files')

    # Argument 2: arch
    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use for image classification')

    # Argument 3: dogfile
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='Text file that contains all labels associated to dogs')
    return parser.parse_args()

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """

    # Retrieve the filenames from folder pet_images/
    filename_list = listdir(image_dir)

    petlabels_dic = dict()

    for idx in range(0, len(filename_list), 1):
        image_name = filename_list[idx].split("_")
        pet_label = "" #temp label to hold pet label name
        for word in image_name:
            if word.isalpha():
                pet_label += word.lower() + " "
        pet_label = pet_label.strip()
        if filename_list[idx] not in petlabels_dic:
            petlabels_dic[filename_list[idx]] = pet_label
        else:
            print("** Warning: Key=", filename_list[idx],
                  "already exists in pet_dic with value =", petlabels_dic[filename_list[idx]])
    return petlabels_dic

def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    results_dic = dict()

    for key in petlabel_dic:
        # runs classifier function to classify the images classifier function
        # inputs: path + filename and model, returns model_label as classifier label
        model_label = classifier(images_dir+key, model)

        # Processes the results so they can be compaed with pet image labels
        # set labels to lowercase and strips off whitespace
        model_label = model_label.lower().strip()

        # defines 'truth' as pet image label and tries to find it using find()
        # string function 'found' to find it within classifier label (model_label)
        truth = petlabel_dic[key]
        found = model_label.find(truth)

        # if found (0 or greater) then make sure true answer wasn't found within
        # another word and thus not really found, if truely found then add to
        # results dictionary and set match=1(yes) otherwise as match=0(no)
        if found >= 0: # if found>0, pet label found in classifier label
            if  ((found == 0 and len(truth) == len(model_label)) or
                 (((found == 0) or (model_label[found-1] == ' '))  and
                  ((found + len(truth) == len(model_label)) or
                   (model_label[found + len(truth): found+len(truth)+1] in
                    (',', ' '))
                  )
                 )
                ):
                # found label as stand-alone term (not within label)
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
            # found within a word/term not a label existing on its own
            else:
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 0]
        # if not found set results dictionary with match=0 (no)
        else:
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 0]

    return results_dic

def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """

    dognames_dic = dict() # dictionary holds all dognames in dogsfile
    with open(dogsfile, "r") as infile: #opens dogsfile
        line = infile.readline() # read first line
        while line != "": # while line isn't empty
            line = line.rstrip() # strip newline from line
            if line not in dognames_dic: # if dog name doesn't already exist in dict
                dognames_dic[line] = 1 # add line to dictionary
            else: # otherwise there's a warning message of duplicate
                print("**Warning: Duplicate dognames", line)
            line = infile.readline() # reads in the next line to see if it is empty
    for key in results_dic: # go through results_dic dictionary
        if results_dic[key][0] in dognames_dic: # if the pet image label is found in dognames_dic
            # if the classifier label is found in dognames_dic (is a dog)
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1)) # appends (1, 1) because both labels are dogs
            else:
                results_dic[key].extend((1, 0)) # appends (1, 0) because only pet label is a dog
        else: # pet image label not found in dognames_dic
            # if the classifier label is found in dognames_dic (is a dog)
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1)) # appends (0, 1) because both labels are dogs
            else:
                results_dic[key].extend((0, 0)) # appends (0, 0) because none are dogs

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """

    results_stats = dict() # dictionary to hold counts and percentages
    # et results to 0
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0

    for key in results_dic: # iterate through results_dic dictionary
        if results_dic[key][2] == 1: # if labels match
            results_stats['n_match'] += 1 # increment count of matches

        if sum(results_dic[key][2:]) == 3: # if label is a dog and labels match
            results_stats['n_correct_breed'] += 1 # increment count of correct breed

        if results_dic[key][3] == 1: # if label is a dog
            results_stats['n_dogs_img'] += 1 # increment count of dogs
            if results_dic[key][4] == 1: # if pet image is a dog
                results_stats['n_correct_dogs'] += 1 # increment correct dog classification count

        else: # if pet label is not a dog
            if results_dic[key][4] == 0: # if pet image is not a dog
                results_stats['n_correct_notdogs'] += 1 # increment incorrect dog classif. count


    results_stats['n_images'] = len(results_dic) # calculates total images using len(results_dic)

    # calc total imgs
    results_stats['n_notdogs_img'] = (results_stats['n_images'] - results_stats['n_dogs_img'])

    # calculates % match of dog and imags
    results_stats['pct_match'] = (results_stats['n_match'] /
                                  results_stats['n_images'])*100.0

    # calculates % correctly classified dogs
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] /
                                         results_stats['n_dogs_img'])*100.0
    # calculates % correctly classified breeds
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] /
                                          results_stats['n_dogs_img'])*100.0

    if results_stats['n_notdogs_img'] > 0: # if there is more than 0 non-dog images
        # calculates % correctly classified non-dogs
        results_stats['pct_correct_notdogs'] = (results_stats['n_correct_notdogs'] /
                                                results_stats['n_notdogs_img'])*100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0 # if there are no non-dog images, set % to 0

    return results_stats

def print_results(results_dic, results_stats, model,
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print("\n\n*** Results summary for CNN Model Architecture", model.upper(), "***")
    print("%20s: %3d" % ("N Images", results_stats['n_images'])) # prints number of images
    print("%20s: %3d" % ("N Dog Images", results_stats['n_dogs_img'])) # prints number of dog images

    # prints number of non-dog images
    print("%20s: %3d" % ("N Not-Dog Images", results_stats['n_notdogs_img']))

    # prints summary statistics (%) on model run
    print(" ")
    for key in results_stats: # iterates through results_stats dictionary
        if key[0] == "p": # if statistic keys start with 'p'
            print("%20s: %5.1f" % (key, results_stats[key])) # then print out percentages

    # prints out dog misclassifications if print_incorrect_dogs is true AND
    # if there is dog misclassification
    if  (print_incorrect_dogs and
         ((results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs'])
          != results_stats['n_images'])
        ):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        # iterate through reuslts_dic dictionary and finds and prints misclassifications
        for key in results_dic:
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s  Classifier: %-30s" % (results_dic[key][0],
                                                          results_dic[key][1]))

    # prints out dog breed misclassifications if print_incorrect_dogs is true AND
    # if there is dog misclassification
    if  (print_incorrect_breed and
         ((results_stats['n_correct_dogs'] + results_stats['n_correct_breed'])
          != results_stats['n_images'])
        ):
        print("\nINCORRECT Dog Breed Assignments:")
        # iterate through reuslts_dic dictionary and finds and prints misclassifications
        for key in results_dic:
            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print("Real: %-26s  Classifier: %-30s" % (results_dic[key][0],
                                                          results_dic[key][1]))

# Call to main function to run the program
if __name__ == "__main__":
    main()
