# OCR data extraction from supermarket receipts
This project was carried out as part of the TechLabs “Digital Shaper Program” in Aachen (Summer Term 2021)
[blog post](https://techlabs-aachen.medium.com/ocr-for-extracting-information-from-supermarket-receipts-96bec1cfabfd)
## Methods
The app utilizes Google's Tesseract engine for optical character recognition. After preprocessing the input image to maximize its quality, it is run through Tesseract. Using RegEx on the output from the OCR engine, you get a clear representation of the products and their corresponding prices.
## [Demo](https://fbozhkov-ocr-app-streamlit-app-y7mvfi.streamlit.app/)

