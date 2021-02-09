import os
import glob
import csv
import torch
import random


def load_data(subfolder="train"):
    
    """
    load_data() -> Dictionary, List
    
    Returns a dictionary filled with characters as keys and a list with file-paths as values.
        Returns a list with all found characters.
    
    Keywords:
        subfolder - "train" oder "test". Default: "train"
    """
    
    # Initialisierung Dictionary und Listen
    character_files = {}
    all_characters = []
    all_files = []
    
    # Dateien im Verzeichnis suchen
    def find_files(path):
        return glob.glob(path)
    
    for filename in find_files("data/"+subfolder+"/*csv"):
        # Zeichen aus Dateiname extrahieren
        character = os.path.splitext(os.path.basename(filename))[0][0]
        
        # Sammeln der gefunden Zeichen in Liste
        if (character in all_characters) == False:
            all_characters.append(character)
        
        # Erweiterung der Listen im Dictionary für entsprechenden Key
        character_files.setdefault(character, []).append(filename)
    
    return character_files, all_characters
        
    
def file_to_tensor(file, batch=1, input_size=3):
    
    """
    file_to_tensor(file, batch=1, input_size=3) -> Tensor
    
    Returns a tensor of shape (seq_len, batch, input_size).
    
    Keywords:
        file - e.g. "data/train\\a__02621.csv"
        batch - total number of training examples present in a single batch
        input_size - the number of expected features in the input x
    """
    
    # Listen für Beschleunigungswerte
    x_acc = []
    y_acc = []
    z_acc = []

    with open(file, newline="") as csvdatei:
        # Einlesen der Zeilen als String in Liste
        csv_reader_object = csv.reader(csvdatei, delimiter=",")

        # Ground Truth und Identifier
        gt, idf = csvdatei.name.split("__")
        # Ground Truth = letztes Zeichen
        gt = gt[-1]
        #print("Ground Truth: {}".format(gt))
        # Identifier ohne ".csv" als Integer
        idf = int(idf[0:-4])
        #print("Identifier: {}".format(idf))
        
        # Listen mit Beschleunigungswerte füllen
        for row in csv_reader_object:
            kal, x, y, z = row[0].split(";")
            # Bei kal = 1 --> Kalibierung abgeschlossen
            if kal == "1":
                x_acc.append(float(x))
                y_acc.append(float(y))
                z_acc.append(float(z))
                
        # Sequenzlänge bestimmen
        seq_len = len(x_acc)
        
        # Tensor erstellen
        tensor = torch.zeros(seq_len, batch, input_size, dtype=torch.float32)
        
        # Tensor füllen
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
    
    # Anzahl aller Charaktere/ Zeichen
    n_characters = len(all_characters)
    # Leeren Tensor erstellen
    tensor = torch.zeros(1, n_characters)
    # Kennzeichnung des Charakters im Tensor
    tensor[0][all_characters.index(character)] = 1
    
    return tensor
    
    
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
    
    # Auswahl eines zufälligen Wertes aus Liste
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
        
    # Zufälliger Charakter
    character = random_choice(all_characters)
    # Zufällige Datei des Charakters
    file = random_choice(character_files[character])
    # Label als Tensor; dtype = 64-bit integer
    character_tensor = torch.tensor([all_characters.index(character)], dtype=torch.long)
    # Umwandlung file in Tensor
    file_tensor = file_to_tensor(file)
    
    return character, file, character_tensor, file_tensor