import os
import glob
import csv

import torch
import random


def load_data(path="data/train/"):
    
    """
    load_data() -> Dictionary, List
    
    Returns a dictionary filled with characters as keys and a list with file-paths as values.
        Returns a list with all found characters.
    
    Keywords:
        path - data folder consists "train" and "test" folder.
    """
    
    # initialize dictionary and list
    character_files = {}
    all_characters = []
    
    # iterate over files in path
    for filename in glob.glob(path+"*csv"):
        # get character from filename
        character = os.path.splitext(os.path.basename(filename))[0][0]
        
        # collect all characters
        if (character in all_characters) == False:
            all_characters.append(character)
        
        # expand lists in dictionary for character (key)
        character_files.setdefault(character, []).append(filename)

    # sort list
    all_characters.sort()
    
    return character_files, all_characters
        
    
def file_to_tensor(file, batch=1, input_size=3):
    
    """
    file_to_tensor(file, batch=1, input_size=3) -> Tensor
    
    Returns a tensor of shape (seq_len, batch, input_size).
    
    Keywords:
        file - e.g. "data/train\a__02621.csv"
        batch - total number of training examples present in a single batch
        input_size - the number of expected features in the input x
    """
    
    # initialize lists for acceleration data
    x_acc = []
    y_acc = []
    z_acc = []

    with open(file, newline="") as csvdatei:
        # add rows as string to list
        csv_reader_object = csv.reader(csvdatei, delimiter=",")

        # ground truth and identifier
        gt, idf = csvdatei.name.split("__")
        # ground truth = last character
        gt = gt[-1]
        # identifier without ".csv" as integer
        idf = int(idf[0:-4])
        
        # append list with acceleration data
        for row in csv_reader_object:
            cal, x, y, z = row[0].split(";")
            # cal = 1 --> calibration completed
            if cal == "1":
                x_acc.append(float(x))
                y_acc.append(float(y))
                z_acc.append(float(z))
                
        # get sequence length
        seq_len = len(x_acc)
        
        # create empty tensor
        tensor = torch.zeros(seq_len, batch, input_size, dtype=torch.float32)
        
        # fill tensor
        for i in range(seq_len):
            tensor[i][0][0] = x_acc[i]
            tensor[i][0][1] = y_acc[i]
            tensor[i][0][2] = z_acc[i]
        
    return tensor


def character_to_tensor(all_characters, character):
    
    """
    character_to_tensor(all_characters, character) -> Tensor
    
    Returns a one-hot encoded tensor of shape <1 x n_characters>.
        Tensor represents given character.
        
    Keywords:
        all_characters - list with all characters
        character - character that will be one-hot encoded
    """
    
    # number of all characters
    n_characters = len(all_characters)
    # create empty tensor
    tensor = torch.zeros(1, n_characters)
    # label of character in tensor
    tensor[0][all_characters.index(character)] = 1
    
    return tensor
    
    
def random_choice(a):

    """
    random_choice(a) --> random Value

    Returns a random value from a given list.

    Keywords:
        a - list filled values (e.g. integers, strings, etc.) 
    """

    # get random value from list
    random_idx = random.randint(0, len(a) - 1)

    return a[random_idx]

    
def random_training_example(character_files, all_characters):
    
    """
    random_training_example(character_files, all_characters) -> String, String, Tensor, Tensor
    
    Returns a random training example. Character and file as string,
        character-label and whole file as tensor.
        
    Keywords:
        character_files - dictionary filled with characters as keys and
            a list with file-paths as values
        all_characters - list with all characters     
    """
        
    # random character
    character = random_choice(all_characters)
    # random file from selected character
    file = random_choice(character_files[character])
    # label as tensor; dtype = 64-bit integer
    character_tensor = torch.tensor([all_characters.index(character)], dtype=torch.long)
    # transformation file to tensor
    file_tensor = file_to_tensor(file)
    
    return character, file, character_tensor, file_tensor