import numpy as np
import pandas as pd
import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
import os
from tqdm import tqdm
from spacy.tokens import DocBin
from spacy.training.example import Example
import json
import csv


TRAIN_DATA = [
         
        ('SHRADDHA INFOSYSTEMS', {'entities': [(0, 20, 'ORG')]}),
        ('accenture', {'entities': [(0, 9, 'ORG')]}),
        ('adam young', {'entities': [(0, 10, 'ORG')]}),
        ('all the best people inc', {'entities': [(0, 23, 'ORG')]}),
        ('amazon', {'entities': [(0, 6, 'ORG')]}),
        ('artists resource agency inc', {'entities': [(0, 27, 'ORG')]}),
        ('association adapei de l\'oise', {'entities': [(0, 28, 'ORG')]}),
        ('at&t', {'entities': [(0, 4, 'ORG')]}),
        ('aqua rock enterprises ', {'entities': [(0, 22, 'ORG')]}),
        ('bank of america', {'entities': [(0, 15, 'ORG')]}),
        ('black onyx corporation, llc', {'entities': [(0, 27, 'ORG')]}),
        ('boeing', {'entities': [(0, 6, 'ORG')]}),
        ('capgemini', {'entities': [(0, 9, 'ORG')]}),
        ('citi', {'entities': [(0, 4, 'ORG')]}),
        ('cognizant technology solutions', {'entities': [(0, 30, 'ORG')]}),
        ('detalles de moda srl', {'entities': [(0, 20, 'ORG')]}),
        ('deloitte', {'entities': [(0, 8, 'ORG')]}),
        ('domex ltd', {'entities': [(0, 9, 'ORG')]}),
        ('dominie press', {'entities': [(0, 13, 'ORG')]}),
        ('education nationale', {'entities': [(0, 19, 'ORG')]}),
        ('elm building service engineers ltd', {'entities': [(0, 34, 'ORG')]}),
        ('ericsson', {'entities': [(0, 8, 'ORG')]}),
        ('erlandsen & associates', {'entities': [(0, 22, 'ORG')]}),
        ('europgs', {'entities': [(0, 7, 'ORG')]}),
        ('excellera consultant', {'entities': [(0, 20, 'ORG')]}),
        ('ey', {'entities': [(0, 2, 'ORG')]}),
        ('food manufacture and foodmanjobs', {'entities': [(0, 32, 'ORG')]}),
        ('futura zorg', {'entities': [(0, 11, 'ORG')]}),
        ('galway technology centre', {'entities': [(0, 24, 'ORG')]}),
        ('gendex', {'entities': [(0, 6, 'ORG')]}),
        ('google', {'entities': [(0, 6, 'ORG')]}),
        ('gos products for business', {'entities': [(0, 25, 'ORG')]}),
        ('hewlett-packard', {'entities': [(0, 15, 'ORG')]}),
        ('herb real estate inc.', {'entities': [(0, 21, 'ORG')]}),
        ('hsbc', {'entities': [(0, 4, 'ORG')]}),
        ('howell carnegie district library', {'entities': [(0, 32, 'ORG')]}),
        ('ibm', {'entities': [(0, 3, 'ORG')]}),
        ('indigo', {'entities': [(0, 6, 'ORG')]}),
        ('infosys', {'entities': [(0, 7, 'ORG')]}),
        ('interglobe aviation limited', {'entities': [(0, 27, 'ORG')]}),
        ('jesus mission', {'entities': [(0, 13, 'ORG')]}),
        ('jpmorgan chase & co.', {'entities': [(0, 20, 'ORG')]}),
        ('kotak mahindra bank ltd', {'entities': [(0, 23, 'ORG')]}),
        ('land warfare centre', {'entities': [(0, 19, 'ORG')]}),
        ('lauyans & company, inc.', {'entities': [(0, 23, 'ORG')]}),
        ('legal & commercial solutions', {'entities': [(0, 28, 'ORG')]}),
        ('make my trip', {'entities': [(0, 12, 'ORG')]}),
        ("mcdonald's corporation", {'entities': [(0, 22, 'ORG')]}),
        ('media leaders', {'entities': [(0, 13, 'ORG')]}),
        ('microsoft', {'entities': [(0, 9, 'ORG')]}),
        ('natraj enterprises', {'entities': [(0, 18, 'ORG')]}),
        ('nokia', {'entities': [(0, 5, 'ORG')]}),
        ('nj division of motor vehicles', {'entities': [(0, 29, 'ORG')]}),
        ('nhs', {'entities': [(0, 3, 'ORG')]}),
        ('oracle', {'entities': [(0, 6, 'ORG')]}),
        ('out of the blue arts and education trust', {'entities': [(0, 40, 'ORG')]}),
        ('paladin true', {'entities': [(0, 12, 'ORG')]}),
        ('parsonstko', {'entities': [(0, 10, 'ORG')]}),
        ('peco controls corporation', {'entities': [(0, 25, 'ORG')]}),
        ('pwc', {'entities': [(0, 3, 'ORG')]}),
        ('punjab national bank ', {'entities': [(0, 21, 'ORG')]}),
        ('reliance homes, inc.', {'entities': [(0, 20, 'ORG')]}),
        ('shradhha infosystems', {'entities': [(0, 20, 'ORG')]}),
        ('siemens', {'entities': [(0, 7, 'ORG')]}),
        ('socar malaysia', {'entities': [(0, 14, 'ORG')]}),
        ('st. joseph school, mechanicsburg, pa', {'entities': [(0, 36, 'ORG')]}),
        ('tata consultancy services', {'entities': [(0, 25, 'ORG')]}),
        ('torq', {'entities': [(0, 4, 'ORG')]}),
        ('united states air force', {'entities': [(0, 23, 'ORG')]}),
        ('united states postal service', {'entities': [(0, 28, 'ORG')]}),
        ('us army', {'entities': [(0, 7, 'ORG')]}),
        ('us navy', {'entities': [(0, 7, 'ORG')]}),     
        ('AQUA ROCK ENTERPRISES', {'entities': [(0, 21, 'ORG')]}),
        ('InterGlobe Aviation Limited', {'entities': [(0, 27, 'ORG')]}),
        ('Kotak Mahindra Bank Ltd', {'entities': [(0, 23, 'ORG')]}),
        ('NATRAJ ENTERPRISES', {'entities': [(0, 18, 'ORG')]}),
        ('SHRADDHA INFOSYSTEMS', {'entities': [(0, 20, 'ORG')]}),
        ('aqua rock enterprises ', {'entities': [(0, 22, 'ORG')]}),
        ('black onyx corporation, llc', {'entities': [(0, 27, 'ORG')]}),
        ('cyan digital house', {'entities': [(0, 18, 'ORG')]}),
        ('dmx africa', {'entities': [(0, 10, 'ORG')]}),
        ('domex ltd', {'entities': [(0, 9, 'ORG')]}),
        ('food manufacture and foodmanjobs', {'entities': [(0, 32, 'ORG')]}),
        ('gos products for business', {'entities': [(0, 25, 'ORG')]}),
        ('holland surfing association', {'entities': [(0, 27, 'ORG')]}),
        ('herb real estate inc.', {'entities': [(0, 21, 'ORG')]}),
        ('hdfc bank', {'entities': [(0, 9, 'ORG')]}),
        ('howell carnegie district library', {'entities': [(0, 32, 'ORG')]}),
        ('indigo', {'entities': [(0, 6, 'ORG')]}),
        ('legal & commercial solutions', {'entities': [(0, 28, 'ORG')]}),
        ('make my trip', {'entities': [(0, 12, 'ORG')]}),
        ('media leaders', {'entities': [(0, 13, 'ORG')]}),
        ('natraj enterprises', {'entities': [(0, 18, 'ORG')]}),
        ('nj division of motor vehicles', {'entities': [(0, 29, 'ORG')]}),
        ('parsonstko', {'entities': [(0, 10, 'ORG')]}),
        ('peco controls corporation', {'entities': [(0, 25, 'ORG')]}),
        ('punjab national bank ', {'entities': [(0, 21, 'ORG')]}),
        ('richini', {'entities': [(0, 7, 'ORG')]}),
        ('shraddha infosystems', {'entities': [(0, 20, 'ORG')]}),
        ('valeo design & marketing', {'entities': [(0, 24, 'ORG')]}),
        ('vyas infosys', {'entities': [(0, 12, 'ORG')]}),
        ('VYAS INFOSYS', {'entities': [(0, 12, 'ORG')]}),
        ('Tata Group', {'entities': [(0, 10, 'ORG')]}),
        ('Reliance Industries Limited', {'entities': [(0, 29, 'ORG')]}),
        ('Infosys', {'entities': [(0, 7, 'ORG')]}),
        ('Wipro', {'entities': [(0, 5, 'ORG')]}),
        ('Mahindra & Mahindra', {'entities': [(0, 19, 'ORG')]}),
        ('Hindustan Unilever Limited (HUL)', {'entities': [(0, 31, 'ORG')]}),
        ('Adani Group', {'entities': [(0, 11, 'ORG')]}),
        ('ITC Limited', {'entities': [(0, 11, 'ORG')]}),
        ('Larsen & Toubro (L&T)', {'entities': [(0, 21, 'ORG')]}),
        ('HDFC Bank', {'entities': [(0, 9, 'ORG')]}),
        ('State Bank of India (SBI)', {'entities': [(0, 24, 'ORG')]}),
        ('Bharat Petroleum Corporation Limited (BPCL)', {'entities': [(0, 43, 'ORG')]}),
        ('HCL Technologies', {'entities': [(0, 17, 'ORG')]}),
        ('Tech Mahindra', {'entities': [(0, 13, 'ORG')]}),
        ('Sun Pharmaceutical Industries Ltd.', {'entities': [(0, 33, 'ORG')]}),
        ('Bajaj Auto Limited', {'entities': [(0, 19, 'ORG')]}),
        ('Asian Paints', {'entities': [(0, 12, 'ORG')]}),
        ('Maruti Suzuki India Limited', {'entities': [(0, 28, 'ORG')]}),
        ('Titan Company Limited', {'entities': [(0, 21, 'ORG')]}),
        ('ONGC (Oil and Natural Gas Corporation Limited)', {'entities': [(0, 43, 'ORG')]}),
        ('Cafe Beanstalk', {'entities': [(0, 14, 'ORG')]}),
        ('The Foodie Hub', {'entities': [(0, 15, 'ORG')]}),
        ('The Oriental Bistro', {'entities': [(0, 20, 'ORG')]}),
        ('Chill & Grill Cafe', {'entities': [(0, 19, 'ORG')]}),
        ('Creamy Delights', {'entities': [(0, 16, 'ORG')]}),
        ('Pizzeria Paradise', {'entities': [(0, 18, 'ORG')]}),
        ('Berqhotel Grosse Scheidegg', {'entities': [(0, 26, 'ORG')]}),
        ('The Bake Shoppe', {'entities': [(0, 15, 'ORG')]}),
        ('Munchies Cafe', {'entities': [(0, 13, 'ORG')]}),
        ('Dine & Wine', {'entities': [(0, 11, 'ORG')]}),
        ('The Fusion Junction', {'entities': [(0, 19, 'ORG')]}),
        ('The Grand Hotel', {'entities': [(0, 15, 'ORG')]}),
        ('Exquisite Resorts', {'entities': [(0, 17, 'ORG')]}),
        ('Harmony Inn', {'entities': [(0, 11, 'ORG')]}),
        ('Royal Retreat', {'entities': [(0, 13, 'ORG')]}),
        ('The Elegant Lodge', {'entities': [(0, 17, 'ORG')]}),
        ('Paradise Palms', {'entities': [(0, 14, 'ORG')]}),
        ('Sunset View Hotel', {'entities': [(0, 17, 'ORG')]}),
        ('Golden Sands Resort', {'entities': [(0, 18, 'ORG')]}),
        ('Mountain Serenity', {'entities': [(0, 17, 'ORG')]}),
        ('Seaside Retreat', {'entities': [(0, 15, 'ORG')]}),
        ('The Food Factory', {'entities': [(0, 16, 'ORG')]}),
        ('Spice Noodle House', {'entities': [(0, 18, 'ORG')]}),
        ('Crispy Bites', {'entities': [(0, 12, 'ORG')]}),
        ('Gourmet Garden', {'entities': [(0, 14, 'ORG')]}),
        ('Taste of Heaven', {'entities': [(0, 15, 'ORG')]}),
        ('Flavorsome Delicacies', {'entities': [(0, 21, 'ORG')]}),
        ('The Hungry Platter', {'entities': [(0, 18, 'ORG')]}),
        ('Fresh & Fiery', {'entities': [(0, 13, 'ORG')]}),
        ('Savory Spices', {'entities': [(0, 13, 'ORG')]}),
        ('The Sweet Tooth', {'entities': [(0, 15, 'ORG')]})
        ]

#nlp = spacy.blank("en") # load a new spacy model
nlp = spacy.load("en_core_web_sm") # load other spacy model

db = DocBin() # create a DocBin object

for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

os.chdir(r'C:\Users\owner\OCR_READER')
db.to_disk("./train.spacy") # save the docbin object