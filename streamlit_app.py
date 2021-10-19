import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import pandas as pd
import math
import matplotlib.pyplot as plt

st.write("""
# Upload receipt(s) for data extraction""")
def sort_text(receipt):
    #Text = str(open(receipt, 'r').read())

    # Text sorting
    # Com_words = ['REWE', 'EUR', 'SUMME', 'PAYBACK', 'BAR']
    # Pattern = re.compile(
    #     r'\b[A-Z]{2,15} [A-Z]{2,15} [A-Z]{2,15}\b|\b[A-Z]{2,15}. [A-Z]{2,15}\b|\b[A-Z]{2,15} [A-Z]{2,15}\b|\b[A-Z]{2,15}\b')
    # Digits = re.findall(r'\d{1,2},\d{2} A|\d{1,2},\d{2}\s*B', receipt)
    # Words = Pattern.findall(receipt)
    # Digits_filt = []
    # Words_capital = []
    # for w in Words:
    #     if w not in Com_words:
    #         Words_capital.append(w)
    # for d in Digits:
    #     d = d.strip(' AB')
    #     Digits_filt.append(d)
    #
    # return Words_capital,Digits_filt

    Com_words = re.compile(r"\bEUR\b|\bEJR\b|\bCUR\b|\bPAYBACK\b|\bSUMME\b")
    Com_words_2 = re.compile(r"\bA\b|\bB\b")
    # EJR and CUR is added to counter mis interpretation of 'EUR' Characters

    W_Pattern = re.compile(
        r"\b[A-Z]+[.]? [A-Z/]+[.]? [A-Z+]+[.]? [A-Z+]+[.]?\b|\b[A-Z]+[.]? [A-Z/]+[.]? [A-Z+]+[.]?\b|\b[A-Z]+[.]? [0-9,]*[.]? [A-Z%]*[.]?\b|\b[A-Z.!]+ [A-Z.]+\b|\b[A-Z]+\b"
    )
    D_Pattern = re.compile(r"\d{1,2},\d{2}[ ][AB]")
    receipt = re.sub(Com_words, "", receipt)
    Words = W_Pattern.findall(receipt)
    Digits = D_Pattern.findall(receipt)
    Words_capital = [ ]
    Digits_filt = [ ]
    for w in Words :
        if not Com_words_2.match(w) :
            w = re.sub(r"\d{1,2},\d{2}", "", w)
            w = re.sub(r"[ ]+[AB]\b", "", w)
            w = w.rstrip(" ")
            Words_capital.append(w)
    for d in Digits :
        d = d.rstrip(" AB")
        Digits_filt.append(d)

    return Words_capital, Digits_filt


# Takes 'UID' as a cue where the items start and 'BAR' where it ends
def items_only(words_capital, digits_filt):
    start_substring = "UID"
    string_containing_start_substring = [string for string in words_capital if start_substring in string]
    start_word = words_capital.index(string_containing_start_substring[0])
    end_substring = "BAR"
    string_containing_end_substring = [string for string in words_capital if end_substring in string]
    end_word = words_capital.index(string_containing_end_substring[0])
    words_filt = words_capital[start_word + 1:end_word]

    df_items = pd.DataFrame(words_filt, columns=['Items'])
    df_items['Price'] = pd.DataFrame(digits_filt)
    #df_items = df_items.dropna()

    return df_items

def extract_info(receipt):
    words, digits = sort_text(receipt)
    info = items_only(words, digits)

    return info

def matplot_lib(items):
    df_items = items.sort_values(by=["Price"], ascending=True).reset_index()
    df_items['Price'] = [x.replace(',', '.') for x in df_items['Price']]
    df_items["Price"] = df_items.Price.astype(float)
    price_max = math.ceil(df_items["Price"].max())
    step = 0
    if price_max < 5:
        step = 0.25
    elif 5 <= price_max < 10:
        step = 0.5
    else:
        step = 1
    price_max = price_max + step

    fig, ax = plt.subplots(figsize=[8, 6], constrained_layout=True)
    ax.barh(
        df_items.index,
        df_items["Price"],
        height=0.5,
        color="#CC071E",
    )
    plt.grid(True, "major", "x", alpha=0.3)
    ax.set_yticks(df_items.index)
    ax.set_yticklabels(df_items["Items"], fontsize=10)
    ax.set_xticks(np.arange(0, price_max, step))
    ax.set(title="Insights", xlabel="Price (€)", ylabel="Item")

    return st.pyplot(fig)


uploaded_receipts = st.file_uploader("Choose a file")
if uploaded_receipts is not None:
    receipt = Image.open(uploaded_receipts)
    img_array = np.array(receipt)

    col1, col2 = st.columns(2)

    col1.image(receipt, use_column_width=True)

    # applying binarization and noise removal
    #gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) # this line is not needed when using receiptsScan600dpi folder
    thresh = 255 - (cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    receipt_items = extract_info(text)
    col2.title("Extracted Items:")
    col2.text(receipt_items)
    col2.title("Raw OCR (beta testing)")
    col2.text(text)

    result = st.button("Get insights")
    if result:
        matplot_lib(receipt_items)



