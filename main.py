import keras_ocr
import pytesseract
from paddleocr import PaddleOCR
import easyocr
from PIL import Image
    
def keras_func(img_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    img=[img_path]
    res = pipeline.recognize(img)
    texts = [item[0] for item in res[0]]
    a = (texts, 0.86, 'keras-ocr')
    results=[]
    results.append(a)
    return results

def paddle_func(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    result = ocr.ocr(image_path, cls=True)
    results = []
    for line in result:
        if line and line[0]:
            print(line[0])
            coordinates, (text, confidence_score) = line[0]
            results.append((text, confidence_score, 'paddle-ocr'))
    print(results)
    return results

def tesseract_func(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    data = pytesseract.image_to_data(img, output_type='dict')
    confidence_scores = [int(conf) for conf in data['conf'] if conf != '-1']
    average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    return [(text.strip(), average_confidence / 100, 'tesseract')]

def easyocr_func(image_path):
    print("*"*7 + "EASY OCR" + "*"*7)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    results_lst = []
    if results:
        for (_, text, confidence) in results:
            if text.strip() != None:
                a = (text, confidence, 'easyocr')
                results_lst.append(a)
    return results_lst

def combined_ocr(image_path):
    results = []

    paddle_results = paddle_func(image_path)
    tesseract_results = tesseract_func(image_path)
    easyocr_results = easyocr_func(image_path)
    keras_results = keras_func(image_path)

    results.extend(paddle_results)
    results.extend(tesseract_results)
    results.extend(easyocr_results)
    results.extend(keras_results)
    threshold=0.85
    filtered_results = [(text, confidence, func_name) for text, confidence, func_name in results if confidence >= threshold]

    if filtered_results:
        for text, confidence, func_name in filtered_results:
            formatted_confidence_score = f"{confidence:.2f}"
            print(f"function = \"{func_name}\" text = \"{text}\" and confidence_score = \"{formatted_confidence_score}\"")
        return filtered_results
    else:
        print("No results with confidence >= 0.85 found.")
        return []

image_path = "/your_image_path"
combined_ocr(image_path)