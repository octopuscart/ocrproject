from flask import Flask, request, jsonify, render_template, send_file
from flask_restful import Resource, Api, reqparse
import re
import cv2 
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path
from num2words import num2words
import json
import base64
from io import BytesIO


app = Flask(__name__)

@app.route('/upload/ocr', methods=['POST', 'GET'])
def upload_file():
    if request.method == "POST":
       # getting input with base64_string = base64data in HTML form
       # this is code 
        base64_string = request.form.get("base64data")
        
        # decode base64 string to PIL Image object
        image = decode_base64(base64_string)
        
        # convert PIL Image object to NumPy array
        img_array = convert_to_numpy(image)
        
        # perform text detection on the image
        gray = convert_to_grayscale(img_array)
        
        thresh = apply_thresholding(gray)
        
        closing = apply_noise_removal(thresh)
        
        processed_image = perform_text_detection(closing, img_array)
        
        # save the processed image to a temporary file
        temp_file = r"D:\Projects\ocr-project\OCR_READER\Images\processed_image.jpg"
        cv2.imwrite(temp_file, processed_image)

        # read the saved image file and encode it to base64 string
        with open(temp_file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Extract text from preprocessed image
        data = extract_text(image)
        
        # save the extracted data as JSON in the static folder
        temp_json_path = r"D:\Projects\ocr-project\OCR_READER\static\extracted_data.json"
        with open(temp_json_path, "w") as json_file:
            json.dump(data, json_file)

        # render the HTML template with the extracted data and the processed image
        return render_template("detect.html", data=data, image=encoded_image)
    
    # if the request method is GET, return the upload form
    return render_template("index.html")


def decode_base64(base64_string):
    if base64_string:
        image_bytes = base64.b64decode(base64_string.split(',')[1])
        return Image.open(BytesIO(image_bytes))
    else:
        # Handle the case when base64_string is None
        return "Error: Invalid base64 string"


def convert_to_numpy(image):
    return np.array(image)

# hey!
def convert_to_grayscale(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return gray


def apply_thresholding(gray):
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh


def apply_noise_removal(thresh):
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing


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
    
    # Track the number of GST numbers encountered
    gst_count = 0
    
    # Track the number of GST number label encountered
    gstin_count = 0

    # Preceding label for GSTIN
    preceding_label = 'GSTIN'
    label_pattern = re.escape(preceding_label) + r"\b)\s*(\w+)"
 
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

            # Check if the detected text contains specific information
            text = d['text'][i]
            if 'Date' in text:
                if not date_label_encountered:
                    date_label_bbox = (x, y, w, h)
                    date_label_encountered = True
                match_date = re.search(date_pattern, text)
                if match_date and not first_date_encountered:
                    # Highlight the first encountered date
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for date
                    first_date_encountered = True

            elif 'Invoice No.' in text:
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for invoice number
                
            elif 'GSTIN' in text:
                gstin_count += 1
                if gstin_count == 2:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan color for GST number
            
            # Check if the detected text contains specific information
            elif 'Total' in text or 'Grand Total' in text:
                total_labels.append(text)
                total_label_bboxes.append((x, y, w, h))
        
            else:
                # Check for specific patterns within the text and highlight accordingly
                invoice_pattern = [r"\d{4}-\d{4}-\d{4}-\d{4}"]
                
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
                match_invoice = any(re.search(pattern, text) for pattern in invoice_pattern)

                if match_date and match_date.group() not in detected_dates:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for date
                    detected_dates.append(match_date.group())
                    
                elif match_invoice:
                    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color for invoice number

                elif match_gst:
                    gst_count += 1
                    if gst_count == 2:
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
    
   # Highlight the bounding box of the "Total" or "Grand Total" label positioned next to or above the maximum amount
    for i in range(len(total_label_bboxes)):
        (x, y, w, h) = total_label_bboxes[i]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Greean color for Total or Grand Total label

    # Return the processed image
    return img_copy


def extract_invoice_number(extracted_text):
    # Using preceding label to extract invoice number
    labels = ['Invoice Number:', 'Invoice No.', 'Invoice:', 'Inv. No.', 'Number :']
    for label in labels:
        index = extracted_text.find(label)
        if index != -1:
            invoice_number = extracted_text[index+len(label):].split()[0]
    return invoice_number


def get_invoice_nums(all_words):
    inv_nums = []
    invoice_no_re = r'^[0-9a-zA-Z-:]+$'
    for word in all_words:
        if not re.search('\d', word['text']):
            continue
        if len(word['text']) < 3:
            continue
        result = re.findall(invoice_no_re, word['text'])
        if result:
            inv_nums.append({
                'text': word['text'],
                'x1': word['left'],
                'y1': word['top'],
                'x2': word['left'] + word['width'],
                'y2': word['top'] + word['height']
            })

    return inv_nums

# extracting invoice date
def extract_date(extracted_text):
    # regular expressions for different date formats
    date_patterns = [
         r'\d[1-2][0-9][0-9][0-9][/]\d{1,2}[/]\d{1,2}',  # YYYY/MM/DD 
         r'\d[1-2][0-9][0-9][0-9][-]\d{1,2}[-]\d{1,2}', # YYYY-MM-DD
         r'[a-zA-Z]{3}[.-]\d[0-3][0-3][.-]\d[1-2][0-9][0-9][0-9]', # Mon-DD-YYYY or DD-Mon-YYYY
         r'\d{1,2}-[A-Za-z]{3}-\d[1-2]',    # Mon-DD-YY or DD-Mon-YY
         r'[a-zA-Z]{3}\s\d{1,2}\s\d[1-4]',  # Mon D YYYY
         r'\d[0-3][0-3]\s[a-zA-Z]{3}\s\d[1-4]',   # D Mon YYYY
         r'\d[1-2][0-9][0-9][0-9][.-][a-zA-Z]{3}[.-]\d{2}',  # YYYYY-Mon-DD or YYYY-Mon-DD
         r'\d{1,2}[/-][a-zA-Z]{3}[/-]\d{2,4}',  # Mon D, Yr or D Mon, Yr
         r'\d{1,2}[/]\d{1,2}[/]\d[1-4][0-9][0-9][0-9]',  # MM/DD/YYYY or DD/MM/YYYY
         r'\d{1,2}[-]\d{1,2}[-]\d[1-4][0-9][0-9][0-9]', # YYYY-MM-DD or YYYY-MM-DD
         ]
    
    extracted_date = None
    
    # check each date pattern against the extracted text
    for pattern in date_patterns:
        match = re.search(pattern, extracted_text)
        if match:
            extracted_date = match.group()
            break
    
    return extracted_date
 
    
def extract_gst_number(extracted_text):
    preceding_label = 'GSTIN :'  # Preceding label

    label_pattern = re.escape(preceding_label) + r"\s*(\w+)"

    match = re.search(label_pattern, extracted_text)
    gst_numbers = re.findall(label_pattern, extracted_text)
    if not gst_numbers:
        # Extract all potential GST numbers
        gst_pattern = [
             r"\S+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}",
             r"\d{2}\s+[A-Z]{5}\d{4}[A-Z]{1}\d{1}[Z]{1}[0-9A-Z]{1}"
        ]

        for pattern in gst_pattern:
            gst_numbers = re.findall(pattern, extracted_text)
            if not gst_numbers:
                return None
            if len(gst_numbers) <= 1:
                return gst_numbers[0]
            return gst_numbers[1]

    if len(gst_numbers) <= 1:
        return gst_numbers[0]
    return gst_numbers[1]
     
    
def extract_max_amount(extracted_text):
    amount_pattern = r"[\d,]+\.\d{2}"
    total_amounts = re.findall(amount_pattern, extracted_text)
    thisamount = 0
    
    for amount in total_amounts:
        new_string = amount.replace(",", "")
        float_amount = float(new_string)
        
        if float_amount > thisamount:
            thisamount = float_amount
    
    return thisamount

def extract_cgst_amount(extracted_text):
    # Approach 1: Extract CGST amount using alternative pattern
    cgst_pattern = r"\[\S+\s*\|\s*([\d.,]+)\]"
    matches = re.findall(cgst_pattern, extracted_text)

    # Extract the numeric values and convert to float
    cgst_amounts_pattern = [float(amount.replace(",", "")) for amount in matches]

    if cgst_amounts_pattern:
        return cgst_amounts_pattern[0]

    # Approach 2: Extract CGST amount using index and regular expression
    cgst_index = extracted_text.find("CGST")
    if cgst_index != -1:
        cgst_pattern = r'\d[\d,]+\.\d{2}'
        cgst_matches = re.findall(cgst_pattern, extracted_text[cgst_index:])
        if cgst_matches:
            cgst_amount_index_regex = float(cgst_matches[0].replace(',', ''))
            return cgst_amount_index_regex

    return None

           
def extract_sgst_percentage(extracted_text):
    # Define regex patterns to match the SGST percentage
    pattern1 = r"SGST\s*@\s*(\d+(?:\.\d+)?)%"
    pattern2 = r"(\d+(?:\.\d+)?)%\s*SGST"
    pattern3 = r"SGST\s*@\s*(\d+(?:\.\d+)?)%\s*(\d+(?:\.\d+)?)"
    pattern4 = r"SGST\s*(\d+(?:\.\d+)?)"
    
    # Try to match each pattern in order
    match = re.search(pattern1, extracted_text)
    if match:
        return int(match.group(1))
    
    match = re.search(pattern2, extracted_text)
    if match:
        return int(match.group(1))
    
    match = re.search(pattern3, extracted_text)
    if match:
        return int(match.group(1))
    
    match = re.search(pattern4, extracted_text)
    if match:
        return None # SGST without percentage value
    
    return None # No match found

def extract_cgst_percentage(extracted_text):
    # Define a list of regex patterns to match CGST tax with different formats
    patterns = [
        r'CGST @ (\d+(?:\.\d+)?)%',
        r'CGST @ (\d+(?:\.\d+)?)',
        r'CGST\s*:\s*(\d+(?:\.\d+)?)',
        r'CGST\s*(\d+(?:\.\d+)?)%',
        r'CGST\s*(\d+(?:\.\d+)?)'
    ]
    
    # Try to match each pattern in the list and return the first non-empty result
    for pattern in patterns:
        match = re.search(pattern, extracted_text)
        if match:
            return match.group(1)
    
    # Return None if no match is found
    return None

     
custom_config = r'--oem 3 --psm 6'
def extract_text(img):
    extracted_text = pytesseract.image_to_string(img, config=custom_config)
    # print(extracted_text)
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
    
    gst_number = extract_gst_number(extracted_text)
    if gst_number:
        print(f"Extracted GST number: {gst_number}")
        data['GST_number'] = gst_number
    else:
        print("GST number not found in text.") 
        
    max_amount = extract_max_amount(extracted_text)
    if max_amount:
        print(f"Extracted maximum amount: {max_amount}")
        data['Total'] = max_amount
    else:
        print("Total amount not found in text.")
        
    cgst_amount = extract_cgst_amount(extracted_text)
    if cgst_amount:
        print(f"Extracted GGST amount: {cgst_amount}")
        data['GGST amount'] = cgst_amount
    else:
        print("GGST amount not found in text.")

    sgst = extract_sgst_percentage(extracted_text)
    if max_amount:
        print(f"Extracted SGST Percentage: {sgst}")
        data['SGST Percentage'] = sgst
    else:
        print("SGST Percentage not found in text.")    
    
    cgst = extract_cgst_percentage(extracted_text)
    if max_amount:
        print(f"Extracted CGST Percentage: {sgst}")
        data['CGST Percentage'] = sgst
    else:
        print("CGST Percentage not found in text.")  

    return data

if __name__ == '__main__':
    app.run(debug =True)
