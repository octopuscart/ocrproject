from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from flask_restful import Resource, Api, reqparse
#import mysql.connector
import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from PIL import Image
import pandas as pd
from pdf2image import convert_from_bytes
from num2words import num2words
import json
import base64
from io import BytesIO
import PyPDF2
from datetime import date
from datetime import datetime
from collections import OrderedDict


app = Flask(__name__)


@app.route('/upload/ocr', methods=['POST', 'GET'])
def upload_file():
    if request.method == "POST":
        # getting input with base64_string = base64data in HTML form
        base64_string = request.form.get("base64data")

        # decode base64 string to PIL Image object or PDF bytes
        if base64_string.startswith('data:image'):
            image = decode_base64(base64_string)
            data, encoded_image = process_image(image)
        else:
            try:
                # Attempt to decode the base64 string as PDF bytes
                pdf_bytes = base64.b64decode(base64_string)
                data = process_pdf(base64_string)
                encoded_image = None
            except:
                return "Error: Invalid base64 string"
        
        # render the HTML template with the extracted data and the processed image
        return render_template("detect.html", data=data, image=encoded_image)
    
    # if the request method is GET, return the upload form
    return render_template("index.html")

def process_image(image):
    # convert PIL Image object to NumPy array
    img_array = convert_to_numpy(image)
    
    # perform text detection on the image
    gray = convert_to_grayscale(img_array)
    
    thresh = apply_thresholding(gray)
    
    closing = apply_noise_removal(thresh)
    
    inverted = invert_image(closing)
    
    dilated = dilate_image(inverted)
    
    processed_image = perform_text_detection(dilated, img_array)

     # save the processed image to a temporary file
    temp_file = r"C:\Users\owner\OCR_READER\Images\processed_image.jpg"
    cv2.imwrite(temp_file, processed_image)

    # read the saved image file and encode it to base64 string
    with open(temp_file, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Extract text from preprocessed image
    data = extract_text(image)
    
    # save the extracted data as JSON in the static folder
    temp_json_path = r"C:\Users\owner\OCR_READER\static\extracted_data.json"
    with open(temp_json_path, "w") as json_file:
        json.dump(data, json_file)
        
    return data, encoded_image
      

def process_pdf(pdf_bytes):
    # convert PDF bytes to PIL Image objects
    pdf_images = convert_from_bytes(pdf_bytes)
    
    # convert PDF bytes to PIL Image objects
    pdf_images = convert_from_bytes(pdf_bytes)

    # process each page of the PDF
    data = []
    for image in pdf_images:
        processed_data = process_image(image)
        data.append(processed_data)

    return data


def decode_base64(base64_string):
    if base64_string:
        image_bytes = base64.b64decode(base64_string.split(',')[1])
        return Image.open(BytesIO(image_bytes))
    else:
        # Handle the case when base64_string is None
        return "Error: Invalid base64 string"
    

def convert_to_numpy(image):
    return np.array(image)


def convert_to_grayscale(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # # If the image data type is not supported, convert it to a compatible data type
    # if gray.dtype != np.uint8:
    #     gray = cv2.convertScaleAbs(gray)
    return gray


def apply_thresholding(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def apply_noise_removal(thresh):
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing

def invert_image(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def dilate_image(image):
    dilated_image = cv2.dilate(image, None, iterations=5)
    return dilated_image


def perform_text_detection(thresh, img_array):
    # Apply OCR to the image
    d = pytesseract.image_to_data(img_array, output_type=Output.DICT)

    # Create a copy of the image
    img_copy = img_array.copy()

    # Track the maximum amount and its bounding box
    max_amount = 0
    max_amount_bbox = None
    
    # Track the labels and their bounding boxes
    total_labels = []
    total_label_bboxes = []

    # Initialize the variables for tracking the date label and the first date encountered
    date_label_encountered = False
    first_date_encountered = False
    date_label_bbox = None
    
    # Initialize the list for tracking detected dates
    detected_dates = []
    
    # Preceding label for GSTIN
    preceding_label = 'GSTIN Number'
    label_pattern = re.escape(preceding_label) + r"\b)\s*(\w+)"
 
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            # Check if the detected text contains specific information
            text = d['text'][i]
            if 'Date' in text or 'Dated' in text or 'Invoice Date' in text:
                if not date_label_encountered:
                    date_label_bbox = (x, y, w, h)
                    date_label_encountered = True
                match_date = re.search(date_pattern, text)
                if match_date and not first_date_encountered:
                    # Highlight the first encountered date
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for date
                    first_date_encountered = True

            elif 'Invoice No.' in text or 'Invoice Number' in text or 'Inv No' in text or 'Invoice No:' in text:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for invoice number
                # Highlight the value of the invoice number with red color
                match_invoice = re.search(r':\s*(\w+)', text)
                if match_invoice:
                    invoice_number = match_invoice.group(1)
                    cv2.putText(img_copy, invoice_number, (x + w + 5, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
            elif 'GSTIN Number:' in text or 'GSTIN/UIN:' in text or 'GSTIN No:' in text:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan color for GST number
            
            # Check if the detected text contains specific information
            elif 'Total' in text or 'Grand Total' in text or 'Invoice Total' in text:
                total_labels.append(text)
                total_label_bboxes.append((x, y, w, h))
        
            else:
                # Check for specific patterns within the text and highlight accordingly
                invoice_number_patterns = [
                            r'Invoice Number\s*\n\s*\n(.+?)\s*\n',
                            r'Invoice No\. : (\d+)',
                            r'Invoice Number : (\d+)',
                            r'Inv No\. : (\d+)',
                            r'Inv #: (\d+)',
                            r'Bill No\. : (\d+)',
                            r'Bill #: (\d+)',
                            r'Number : (\d+)',
                            r'Ref No\. : (\d+)',
                            r'Reference No\. : (\d+)',
                            r'Document No\. : (\d+)',
                            r'Order No\. : (\d+)',
                            r'Transaction ID : (\d+)',
                            r'ID : (\d+)',
                            r'Number : (.+?)\s',
                            r'Invoice No\.\s*\n(.+?)\s+',
                            r'Invoice No \: ([^\n]+)',
                            r'Invoice No\.\s*\n([^:\n]+)', 
                            r'Invoice No\.\s*\n([\w\d/-]+)', 
                            r'Invoice No\.?\s*:?\s*\n(.+?)\s'
                            r'Invoice No\. :\s*([A-Za-z]+[\w\s]*)' 
                        ]
                
                gst_pattern = [
                    r"\S+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}",
                    r"\d{2}\s+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}"
                ]
                
                date_pattern = r'\d[1-2][0-9][0-9][0-9][/]\d{1,2}[/]\d{1,2}|' \
                               r'\d[1-2][0-9][0-9][0-9][-]\d{1,2}[-]\d{1,2}|' \
                               r'[a-zA-Z]{3}[.-]\d[0-3][0-3][.-]\d[1-2][0-9][0-9][0-9]|' \
                               r'[a-zA-Z]{3}\s\d{1,2}\s\d[1-4]|' \
                               r'\d[0-3][0-3]\s[a-zA-Z]{3}\s\d[1-4]|' \
                               r'\d[1-2][0-9][0-9][0-9][.-][a-zA-Z]{3}[.-]\d{2}|' \
                               r'\d{1,2}[/-][a-zA-Z]{3}[/-]\d{2,4}|' \
                               r'\d{1,2}[/]\d{1,2}[/]\d[1-4][0-9][0-9][0-9]|' \
                               r'\d{1,2}[-]\d{1,2}[-]\d[1-4][0-9][0-9][0-9]|' \
                               r'\d[1-4][,-]\s?[a-zA-Z]{3}[,-]\s?\d{1,2}|' \
                               r'\d{1,2}[/]\d{1,2}[/]\d{2,4}|' \
                               r'\d{1,2}[-]\d{1,2}[-]\d{2,4}'
                
                amount_pattern = r"[\d,]+\.\d{2}"

                match_gst = any(re.search(pattern, text) for pattern in gst_pattern)
                match_date = re.search(date_pattern, text)
                match_amount = re.search(amount_pattern, text)
                match_invoice = any(re.search(pattern, text) for pattern in invoice_number_patterns)

                if match_date and match_date.group() not in detected_dates:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for date
                    detected_dates.append(match_date.group())
                    
                elif match_invoice:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for invoice number

                elif match_gst:
                   cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan color for second GST number
                
                elif match_amount:
                    # Extract the amount from the text
                    amount = float(match_amount.group().replace(',', ''))
                    if amount > max_amount:
                        max_amount = amount
                        max_amount_bbox = (x, y, w, h)

    # Highlight the bounding box of the maximum amount
    if max_amount_bbox:
        (x, y, w, h) = max_amount_bbox
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for max amount

    # Highlight the bounding box of the date label
    if date_label_bbox:
        (x, y, w, h) = date_label_bbox
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for date label
    
  # Find the closest Total label above or before the max_amount
    closest_label = None
    closest_bbox = None
    if max_amount_bbox:
        for i in range(len(total_label_bboxes)):
            (x, y, w, h) = total_label_bboxes[i]
            if y < max_amount_bbox[1] or (y == max_amount_bbox[1] and x < max_amount_bbox[0]):
                if closest_bbox is None or y > closest_bbox[1] or (y == closest_bbox[1] and x > closest_bbox[0]):
                    closest_label = total_labels[i]
                    closest_bbox = (x, y, w, h)

    # Highlight the closest Total label above or before the max_amount
    if closest_bbox:
        (x, y, w, h) = closest_bbox
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color for closest label

    # Return the processed image
    return img_copy


#  Extracting Invoice Number
def extract_invoice_number(invoice_text):
     # Preprocess the text
    cleaned_text = invoice_text.replace('\n', ' ').strip()

    # Define potential invoice number keywords
    invoice_keywords = ["Invoice No.", "Invoice Number", "Inv #", "Account No.", 
                        "Bill/invoice No","BilINo :", "BillNo. "," Bill  #",
                        "Bill No. : ","BillNo ;", "Bill No ", "Number", "Rech.Nr."]

    # Initialize the invoice number variable
    invoice_number = None

    # Iterate over the keywords
    for keyword in invoice_keywords:
        # Search for the keyword in the text
        match = re.search(f"{keyword}\s*[:.]?\s*([^\n\s]+)", cleaned_text, re.IGNORECASE)
        if match:
            # Extract the invoice number
            invoice_number = match.group(1).strip()
            break

    return invoice_number


# extracting invoice date
def extract_date(extracted_text):
    # Regular expressions for different date formats
    date_patterns = [
        r'\d{1,2}\s(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s\d{2,4}',  # DD Mon YY or DD Mon YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2}\s\d[1-4]',  # Mon D YYYY
        r'\d[0-3][0-3]\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d[1-4]',  # D Mon YYYY
        r'\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}',  # DD MON YYYY
        r'\d{1,2}[/-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[/-]\d{2,4}',  # Mon D, Yr or D Mon, Yr
        r'\d{1,2}\s(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s\d{4}',  # DD Month YYYY
        r'\d{1,2}[/]\d{1,2}[/]\d{2,4}',  # MM/DD/YY or DD/MM/YY or YY/MM/DD
        r'\d{1,2}[-]\d{1,2}[-]\d{2,4}',  # MM-DD-YY or DD-MM-YY or YY-MM-DD      
        r'\d[1-2][0-9][0-9][0-9][/]\d{1,2}[/]\d{1,2}',  # YYYY/MM/DD 
        r'\d[1-2][0-9][0-9][0-9][-]\d{1,2}[-]\d{1,2}', # YYYY-MM-DD
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD 
        r'\d{1,2}[/]\d{1,2}[/]\d[1-4][0-9][0-9][0-9]',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}[-]\d{1,2}[-]\d[1-4][0-9][0-9][0-9]', # YYYY-MM-DD or YYYY-MM-DD
        r'(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[-]\d[0-3][0-3][-]\d[1-2][0-9][0-9][0-9]', # Mon-DD-YYYY or DD-Mon-YYYY
        r'(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[.]\d[0-3][0-3][.]\d[1-2][0-9][0-9][0-9]', # Mon.DD.YYYY or DD.Mon.YYYY
        r'\d{1,2}-(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)-\d[1-2]',    # Mon-DD-YY or DD-Mon-YY
        r'(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s\d{1,2}\s\d[1-4]',  # Mon D YYYY
        r'\d[0-3][0-3]\s(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s\d[1-4]',   # D Mon YYYY
        r'\d[1-2][0-9][0-9][0-9][.-](?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[.-]\d{2}',  # YYYYY-Mon-DD or YYYY-Mon-DD
        r'\d{1,2}[/-](?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)[/-]\d{2,4}',  # Mon D, Yr or D Mon, Yr
        r'\d[1-4][,-]\s?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,-]\s?\d{1,2}',  # Yr, Month D or YYYY, Mon DD
        r'\d{1,2}\s-\s(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s-\s\d{4}',  # DD - Mon - YYYY
        r'\d{1,2}\s/\s(?:[Jj]an|[Ff]eb|[Mm]ar|[Aa]pr|[Mm]ay|[Jj]un|[Jj]ul|[Aa]ug|[Ss]ep|[Oo]ct|[Nn]ov|[Dd]ec)\s/\s\d{4}',  # DD / Mon / YYYY
        r'\d{1,2}\s-\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s-\s\d{4}',  # DD - Mon - YYYY
        r'\d{1,2}\s/\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s/\s\d{4}',  # DD / Mon / YYYY
        r'\d{1,2}[.]\d{1,2}[.]\d{4}'  # DD.MM.YYYY
    ]
    
      # Define the labels and their corresponding preceding patterns
    preceding_labels = {
        'Date': [
            'Date :',
            'Date  :',
            'DT:',
            'DT :'
        ],
        'Billing Date': [
            'Billing Date:',
            'Bill Date:',
            'Bill Date',
            'Bill Dt:',
            'Invoice Date:',
            'Receipt Date:',
        ],
        'Document Date': [
            'Document Date:',
            'Doc Date:',
        ],
        'Issue Date': [
            'Issue Date:',
            'Issued On:',
        ],
        'Invoice Created On': [
            'Invoice Created On:',
            'Created On:',
        ],
        'Invoice Generation Date': [
            'Invoice Generation Date:',
            'Generated On:',
        ],
        'Transaction Date': [
            'Transaction Date:',
            'Trans Date:',
        ],
        'Order Date': [
            'Order Date:',
        ],
        'Sales Date': [
            'Sales Date:',
        ]
    }

    
    extracted_date = None

    # First attempt using date_patterns
    for date_pattern in date_patterns:
        match = re.search(date_pattern, extracted_text, re.IGNORECASE)
        if match:
            extracted_date = match.group()
            break

    # Second attempt using preceding labels
    for label, preceding_patterns in preceding_labels.items():
        if extracted_date:
            break
        for preceding_pattern in preceding_patterns:
            pattern = re.escape(preceding_pattern) + r'\s(' + '|'.join(date_patterns) + ')'
            match = re.search(pattern, extracted_text, re.IGNORECASE)
            if match:
                extracted_date = match.group(1)
                break

   # Convert date format to "dd/mm/yyyy"
    if extracted_date:
        try:
            date_obj = datetime.strptime(extracted_date, '%d-%m-%y')
            year = date_obj.strftime('%Y')
            extracted_date = date_obj.strftime('%d/%m/') + year
        except ValueError:
            try:
                date_obj = datetime.strptime(extracted_date, '%d-%b-%y')
                year = date_obj.strftime('%Y')
                extracted_date = date_obj.strftime('%d/%m/') + year
            except ValueError:
                try:
                    date_obj = datetime.strptime(extracted_date, '%m/%d/%y')
                    year = date_obj.strftime('%Y')
                    extracted_date = date_obj.strftime('%d/%m/') + year
                except ValueError:
                    try:
                        date_obj = datetime.strptime(extracted_date, '%d-%m-%y')
                        year = date_obj.strftime('%Y')
                        extracted_date = date_obj.strftime('%d/%m/') + year
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(extracted_date, '%m-%d-%y')
                            year = date_obj.strftime('%Y')
                            extracted_date = date_obj.strftime('%d/%m/') + year
                        except ValueError:
                            try:
                                date_obj = datetime.strptime(extracted_date, '%d-%b-%Y')
                                year = date_obj.strftime('%Y')
                                extracted_date = date_obj.strftime('%d/%m/') + year
                            except ValueError:
                                try:
                                    date_obj = datetime.strptime(extracted_date, '%d %b %Y')
                                    year = date_obj.strftime('%Y')
                                    extracted_date = date_obj.strftime('%d/%m/') + year
                                except ValueError:
                                    pass  # Handle other date formats here

    return extracted_date
 
       
def extract_gst_number(extracted_text):
    labels = [
        'Gstin No.:', 'GSTIN Number:', 'GSTIN/UIN:',
        'GSTIN :', 'GSTIN', 'Customer GST No ', 'GSTIN Number;',
        'GSTN:', 'GSTIN No, : ', 'GSTIN No-'
    ]

    gst_patterns = [
        r"\S+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}",
        r"\d{2}\s+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}",
        r"(?:\b|_)([0-9A-Z]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[0-9A-Z]{1})\b",
        r"\bCustomer\s+GST\s+No\s+([A-Z0-9]{15})\b",
        r"\bGST\s*Number:\s*([A-Z0-9]{15})\b",
        r"\bCustomer\s+GST\s+No\s+([A-Z0-9]{2}\s*[A-Z0-9]{10})\b"
    ]

    gst_numbers = []
    
    for pattern in gst_patterns:
        matches = re.findall(pattern, extracted_text)
        gst_numbers.extend(matches)

    for label in labels:
        label_pattern = r"\b" + re.escape(label) + r"[:\s]*([A-Z0-9]{15})\b"
        matches = re.findall(label_pattern, extracted_text)
        gst_numbers.extend(matches)

    # Remove spaces before or between GST numbers
    gst_numbers = [number.replace(" ", "") for number in gst_numbers]

    # Extract unique GST numbers
    gst_numbers = list(set(gst_numbers))

    return gst_numbers


def extract_max_amount(extracted_text):
    amount_patterns = [
        r'Total Fare\s*([\d,.]+)',
        r'\*Total Fare \(All Passenger\):\s*([\d,.]+)',
        r'Total Amount:\s*([\d,.]+)',
        r'Amount Due:\s*([\d,.]+)',
        r'Grand Total:\s*([\d,.]+)',
        r"[\d,]+\.\d{2}", 
        r"Total Payable \(INR\) [\d,]+\.\d{2}",
        r'Total Value \(in figure\)\s*([\d,.]+)',
        r'Grand\s*Total\s*\(\w+\)\s*:\s*([\d,.]+)'
        # Add more patterns as needed
    ]
    amounts = []

    for pattern in amount_patterns:
        matches = re.findall(pattern, extracted_text)
        for match in matches:
            clean_amount = float(match.replace(',', '').replace(' ', ''))
            amounts.append(clean_amount)

    if amounts:
        max_amount = max(amounts)
    else:
        max_amount = 0

    return max_amount


def extract_cgst_amount(extracted_text):
    # Approach 1: Extract CGST amount using alternative patterns
    cgst_patterns = [
        r"\[.*\|\s*([\d.,]+)\]",
        r"CGST\(\w+%\)\s+(\d+)"
    ]
    for pattern in cgst_patterns:
        matches = re.findall(pattern, extracted_text)
        # Extract the numeric values and convert to float
        cgst_amounts = [float(amount.replace(",", "")) for amount in matches]
        if cgst_amounts:
            return cgst_amounts[0]

    # Approach 2: Extract CGST amount using index and regular expression
    cgst_index = extracted_text.find("CGST")
    if cgst_index != -1:
        cgst_pattern = r'\d[\d,]+\.\d{2}'
        cgst_matches = re.findall(cgst_pattern, extracted_text[cgst_index:])
        if cgst_matches:
            cgst_amount_index_regex = float(cgst_matches[0].replace(',', ''))
            return cgst_amount_index_regex

    return None

def extract_sgst_amount(extracted_text):
    # Approach 1: Extract CGST amount using alternative patterns
    sgst_patterns = [
        r"\[.*\|\s*([\d.,]+)\]",
        r"SGST\(\w+%\)\s+(\d+)"
    ]
    for pattern in sgst_patterns:
        matches = re.findall(pattern, extracted_text)
        # Extract the numeric values and convert to float
        sgst_amounts = [float(amount.replace(",", "")) for amount in matches]
        if sgst_amounts:
            return sgst_amounts[0]

    # Approach 2: Extract CGST amount using index and regular expression
    sgst_index = extracted_text.find("SGST")
    if sgst_index != -1:
        sgst_pattern = r'\d[\d,]+\.\d{2}'
        sgst_matches = re.findall(sgst_pattern, extracted_text[sgst_index:])
        if sgst_matches:
            sgst_amount_index_regex = float(sgst_matches[0].replace(',', ''))
            return sgst_amount_index_regex

    return None


def extract_igst_amount(text):
    patterns = [
        r'IGST(\([0-9]+%\)\s+([0-9]+))',  # Pattern 1
        r'IGST\s+([0-9.]+)%\s+([0-9.]+)'  # Pattern 2
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if match.group(2):
                return float(match.group(2))
            elif match.group(1):
                return float(match.group(1))

    return None


def extract_gst_total(extracted_text):
    # Define patterns for GST total extraction
    patterns = [
        r"(?i)(?:GST|tax)\s*total[:\s]*([0-9.,]+)",
        r"(?i)total.*?(?:GST|tax)[:\s]*([0-9.,]+)",
        r"(?i)GST\s*Total\s*([0-9.,]+)"
    ]
    
    # Search for patterns in the extracted text
    for pattern in patterns:
        match = re.search(pattern, extracted_text)
        if match:
            gst_total = match.group(1)
            # Remove any commas in the number and convert to float
            gst_total = float(gst_total.replace(',', ''))
            return gst_total
    
    # If no match is found, return None
    return None
           
def extract_sgst_percentage(extracted_text):
    sgst_percentage_patterns = [
        r'SGST @ (?P<percentage>\d+(?:\.\d+)?)%',
        r'SGST\s*:\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'SGST\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'State GST\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'SGST\s*\(\w+%\)',
        r'SGST\s*@?\s*(?P<percentage>\d+(?:\.\d+)?)%'
    ]
    
    for pattern in sgst_percentage_patterns:
        match = re.search(pattern, extracted_text)
        if match and match.groupdict().get('percentage'):
            return match.groupdict().get('percentage')

    return None


def extract_cgst_percentage(extracted_text):
    cgst_percentage_patterns = [
        r'CGST @ (?P<percentage>\d+(?:\.\d+)?)%',
        r'CGST\s*:\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'CGST\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'Central GST\s*(?P<percentage>\d+(?:\.\d+)?)%',
        r'CGST\s*\(\w+%\)',
        r'CGST\s*@?\s*(?P<percentage>\d+(?:\.\d+)?)%'
    ]
    
    for pattern in cgst_percentage_patterns:
        match = re.search(pattern, extracted_text)
        if match and match.groupdict().get('percentage'):
            return match.groupdict().get('percentage')
    
    return None

     
def extract_text(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    df = pd.DataFrame(d)

    df1 = df[(df.conf != '-1') & (df.text != ' ') & (df.text != '')]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    sorted_blocks = df1.groupby('block_num').first().sort_values('top').index.tolist()
    extracted_text = ''
    for block in sorted_blocks:
        curr = df1[df1['block_num'] == block]
        sel = curr[curr.text.str.len() > 3]
        char_w = (sel.width / sel.text.str.len()).mean()
        prev_par, prev_line, prev_left = 0, 0, 0
        for ix, ln in curr.iterrows():
            # add new line when necessary
            if prev_par != ln['par_num']:
                extracted_text += '\n'
                prev_par = ln['par_num']
                prev_line = ln['line_num']
                prev_left = 0
            elif prev_line != ln['line_num']:
                extracted_text += '\n'
                prev_line = ln['line_num']
                prev_left = 0

            added = 0  # num of spaces that should be added
            if ln['left'] / char_w > prev_left + 1:
                added = int((ln['left']) / char_w) - prev_left
                extracted_text += ' ' * added
            extracted_text += ln['text'] + ' '
            prev_left += len(ln['text']) + added + 1
        extracted_text += '\n'

    print(extracted_text)
        
    data = {}
    
    invoice_no = extract_invoice_number(extracted_text)
    if invoice_no:
        print(f"Extracted Invoice Number: {invoice_no}")
        data['Invoice Number'] = invoice_no
    else:
        print("Invoice Number not found in text.")  

    date = extract_date(extracted_text)
    if date:
        print(f"Extracted date: {date}")
        data['Date'] = date
    else:
        print("Date not found in text.")  
    
    gst_numbers = extract_gst_number(extracted_text)
    if gst_numbers:
        print(f"Extracted GST Numbers: {gst_numbers}")
        for i, gst_number in enumerate(gst_numbers):
            label = f"GST Number {i+1}"
            data[label] = gst_number
    else:
        print("GST Numbers not found in text.")  
        
    max_amount = extract_max_amount(extracted_text)
    if max_amount:
        print(f"Extracted maximum amount: {max_amount}")
        data['Total'] = max_amount
    else:
        print("Total amount not found in text.")
        
    cgst_amount = extract_cgst_amount(extracted_text)
    if cgst_amount:
        print(f"Extracted CGST amount: {cgst_amount}")
        data['CGST amount'] = cgst_amount
    else:
        print("CGST amount not found in text.")
        
    sgst_amount = extract_sgst_amount(extracted_text)
    if sgst_amount:
        print(f"Extracted SGST amount: {sgst_amount}")
        data['SGST amount'] = sgst_amount
    else:
        print("SGST amount not found in text.")    
        
    igst_amount = extract_igst_amount(extracted_text)
    if igst_amount:
        print(f"Extracted IGST amount: {igst_amount}")
        data['IGST amount'] = igst_amount
    else:
        print("IGST amount not found in text.")
        
    gst_total = extract_gst_total(extracted_text)
    if gst_total:
        print(f"Extracted GST Total: {gst_total}")
        data['GST Total'] = gst_total
    else:
        print("GST Total not found in text.")        

    sgst = extract_sgst_percentage(extracted_text)
    if sgst:
        print(f"Extracted SGST Percentage: {sgst}")
        data['SGST Percentage'] = sgst
    else:
        print("SGST Percentage not found in text.")    
    
    cgst = extract_cgst_percentage(extracted_text)
    if cgst:
        print(f"Extracted CGST Percentage: {cgst}")
        data['CGST Percentage'] = cgst
    else:
        print("CGST Percentage not found in text.")  

    return data

if __name__ == '__main__':
    app.run(debug =True)

